#!/usr/bin/env python
# coding=utf-8
"""Federated learning training loop for UIE LoRA models."""
import copy
import logging
import os
import random
from collections import defaultdict
import json
import datasets
import numpy as np
import torch
import gc,contextlib
from accelerate.utils import wait_for_everyone
import math
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import torch.distributed as dist
from transformers.trainer_utils import get_last_checkpoint
from uie_collator import DataCollatorForUIE
from uie_trainer_lora import UIETrainer, DenserEvalCallback, skip_instructions, compute_fisher, compute_fisher_arithmetic
from compute_metrics import compute_metrics, compute_grouped_metrics
from model.llama import LlamaForCausalLM_with_lossmask
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from uie_dataset_lora import gen_cache_path
#from plot.data_distribution import compare
from run_uie_lora import ModelArguments, DataTrainingArguments, UIETrainingArguments, FederatedArguments
from torch.utils.data import DataLoader
from continual_fisher_client import ContinualFisherClient, ClientState
from fed_continual_state import ContinualState
from scipy.stats import pearsonr
from typing import Dict, Tuple, List
logger = logging.getLogger("federated_training")
CURRENT_DIR = os.path.dirname(__file__)

def _trainer_accelerator(trainer):
    return getattr(trainer, "accelerator", None)

def _trainer_wait_for_everyone(trainer) -> None:
    accel = _trainer_accelerator(trainer)
    if accel is not None:
        accel.wait_for_everyone()
    elif dist.is_available() and dist.is_initialized():
        dist.barrier()
    else:  # fallback for single-process execution
        wait_for_everyone()

def _trainer_unwrap_model(trainer):
    accel = _trainer_accelerator(trainer)
    if accel is not None:
        return accel.unwrap_model(trainer.model)
    return trainer.model

def partition_dataset(train_dataset, num_clients, alpha, *, base_seed: int):
    rng = np.random.RandomState(base_seed)

    num_samples = len(train_dataset)
    all_idx = np.arange(num_samples)
    rng.shuffle(all_idx)

    # 1) 保证每个 client 至少 1 个样本（若 num_samples >= num_clients）
    if num_samples >= num_clients:
        props = rng.dirichlet([alpha] * num_clients)
        # 先给每人 1 个，剩余再按 Dirichlet 分
        sizes = (props * (num_samples - num_clients)).astype(int) + 1
        # 修正最后一个，确保总和精确等于 num_samples
        sizes[-1] = num_samples - sizes[:-1].sum()
    else:
        # 样本数少于 client 数，最多只能给前 num_samples 个 client 各 1 个，其余为空
        sizes = np.zeros(num_clients, dtype=int)
        sizes[:num_samples] = 1
        rng.shuffle(sizes)  # 哪些 client 为空也固定可复现

    # 2) 额外稳健：避免出现负数或最后一个被压成 0 的边缘情况
    #    （正常不会出现，这里只是护栏）
    sizes = np.maximum(sizes, 0)
    sizes[-1] = num_samples - sizes[:-1].sum()

    # 连续切块（全局随机后的 contiguous split）
    splits = np.split(all_idx, np.cumsum(sizes)[:-1])
    client_datasets = [train_dataset.select(s.tolist()) for s in splits]
    return client_datasets

def partition_dataset_by_label(dataset, num_clients: int, alpha: float, *, base_seed: int, label_key="Dataset"):
    rng = np.random.RandomState(base_seed)

    # 1) 按标签聚合索引
    label2indices = defaultdict(list)
    for idx, example in enumerate(dataset):
        label2indices[example[label_key]].append(idx)

    client_indices = [[] for _ in range(num_clients)]

    # 2) 对每个标签做 Dirichlet，再把该标签的样本分片给各 client
    for indices in label2indices.values():
        indices = np.array(indices)
        rng.shuffle(indices)
        props = rng.dirichlet([alpha] * num_clients)
        bounds = (np.cumsum(props) * len(indices)).astype(int)[:-1]
        splits = np.split(indices, bounds)
        for cid, idxs in enumerate(splits):
            if len(idxs) > 0:
                client_indices[cid].extend(idxs.tolist())

    # 3) 防空 client：从样本最多的 client“偷”一个（可复现：用 rng 选择偷谁的哪一个）
    empty_clients = [cid for cid, idxs in enumerate(client_indices) if len(idxs) == 0]
    if len(empty_clients) > 0:
        # donor 优先选择样本最多的；若并列，按 rng 的顺序处理
        sizes = np.array([len(idxs) for idxs in client_indices])
        for cid in empty_clients:
            donors = np.where(sizes == sizes.max())[0]
            donor = donors[rng.randint(len(donors))]
            # 从 donor 随机偷一个
            stolen_pos = rng.randint(len(client_indices[donor]))
            stolen = client_indices[donor].pop(stolen_pos)
            sizes[donor] -= 1
            client_indices[cid].append(stolen)
            sizes[cid] += 1

    return [dataset.select(idxs) for idxs in client_indices]

def build_model_and_tokenizer(model_args: ModelArguments):
    """
    Unified loader for T5 (seq2seq), LLaMA-2 (decoder-only), and LLaMA-3 (decoder-only).
    Keeps PEFT-LoRA behaviors consistent with your original code.
    """

    # --------- 1) 判别模型族 ----------
    name_lower = model_args.model_name_or_path.lower()
    is_adapter = ("adapter" in name_lower)  # peft adapter 路径
    is_llama = ("llama" in name_lower)
    is_llama3 = ("llama-3" in name_lower) or ("llama3" in name_lower)
    # 只要不是 llama，就按 seq2seq（T5/FLAN-T5 等）处理
    is_seq2seq = (not is_llama)

    # --------- 2) 配置与 tokenizer ----------
    # adapter 的 base 模型名需要先从 peft config 里取出来
    if is_adapter:
        peft_cfg = PeftConfig.from_pretrained(model_args.model_name_or_path)
        base_model = peft_cfg.base_model_name_or_path
    else:
        base_model = model_args.model_name_or_path

    config = transformers.AutoConfig.from_pretrained(
        base_model,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if is_llama:
        # LLaMA-3 推荐走 AutoTokenizer；LLaMA-2 兼容 LlamaTokenizer
        if is_llama3:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                base_model,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            tokenizer = transformers.LlamaTokenizer.from_pretrained(
                base_model,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        # 与你/他脚本一致：显式设置 special ids
        config.bos_token_id = 1
        config.eos_token_id = 2
        config.pad_token_id = 1
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 1
        tokenizer.padding_side = "left"  # decoder-only 左填充，与你原来一致
    else:
        # T5/FLAN-T5 等
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # --------- 3) 选择 model class ----------
    if is_llama:
        # 你自己的自定义 LLaMA 类（保持不变）
        model_class = LlamaForCausalLM_with_lossmask
        lora_task = TaskType.CAUSAL_LM
    else:
        model_class = transformers.AutoModelForSeq2SeqLM
        lora_task = TaskType.SEQ_2_SEQ_LM

    # --------- 4) 加载模型 + 应用 LoRA/PEFT ----------
    if is_adapter:
        # 先加载 base，再把 peft adapter 套上
        model = model_class.from_pretrained(
            base_model,
            from_tf=bool(".ckpt" in base_model),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
    else:
        # 直接加载预训练模型，再用 LoRAConfig 注入训练参数
        model = model_class.from_pretrained(
            base_model,
            from_tf=bool(".ckpt" in base_model),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        peft_config = LoraConfig(
            task_type=lora_task,
            inference_mode=False,
            r=model_args.lora_dim,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)

    # --------- 5) 统一一些生成与 embedding 细节 ----------
    model.resize_token_embeddings(len(tokenizer))
    if is_llama:
        # 和你原来的做法一致
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        model.generation_config.pad_token_id = 1

    # 只训练 LoRA 权重（沿用你原来的规则）
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        elif 'shared' in name:
            param.requires_grad = False

    return model, tokenizer

def compute_fisher_diag(model, dataloader):
    """
    适配LoRA模型的对角线Fisher信息计算（修正版）
    仅计算带"lora"的可训练参数，返回字典格式
    """
    # 自动获取模型所在设备（与模型参数一致）
    device = next(model.parameters()).device
    model.eval()  # 计算梯度时使用训练模式

    # 初始化Fisher累积器（仅跟踪LoRA可训练参数，用参数名作为键）
    fisher_diag = {
        name: torch.zeros_like(param, device="cpu")  # <-- 修改
        for name, param in model.named_parameters()
        if param.requires_grad and "lora" in name  # 仅保留LoRA相关参数
    }

    total_samples = 0  # 累计样本数用于平均

    for batch in dataloader:
        # 适配字典形式的输入批次（如包含input_ids、labels等键）
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        inputs = {k: v for k, v in batch.items() if k != "labels"}  # 输入特征
        labels = batch.get("labels", None)

        if labels is None:
            continue  # 无标签数据不参与计算

        # 前向计算获取logits
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # 计算log概率（适配seq2seq模型的标签维度）
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # 对每个样本计算梯度并累积Fisher
        batch_size = logits.size(0)
        total_samples += batch_size

        for i in range(batch_size):
            # 提取当前样本的标签（忽略填充值-100）
            sample_label = labels[i]
            valid_mask = sample_label != -100
            if not valid_mask.any():
                continue  # 跳过全填充的样本

            # 计算当前样本的log概率（仅有效标签部分）
            sample_log_prob = log_probs[i, torch.arange(log_probs.size(1)), sample_label]
            sample_log_prob = sample_log_prob[valid_mask].sum()  # 累加有效位置的log概率

            # 计算梯度（创建计算图用于二阶导数）
            model.zero_grad(set_to_none=True)
            grads = torch.autograd.grad(
                sample_log_prob,
                [param for name, param in model.named_parameters() if param.requires_grad and "lora" in name],
                create_graph=True,
                retain_graph=True
            )

            # 累积梯度平方到Fisher对角线
            for (name, param), grad in zip(fisher_diag.items(), grads):
                if grad is not None:
                    # --- [修改点 2：将梯度移到 CPU 再累加] ---
                    fisher_diag[name].add_(grad.detach().cpu() ** 2)  # <-- 修改

    # 计算平均Fisher（除以总样本数）
    if total_samples > 0:
        for name in fisher_diag:
            fisher_diag[name] /= total_samples
    else:
        logger.warning("未处理有效样本，Fisher保持初始零值")

    # 归一化处理（保留原min-max归一化逻辑）
    normalized_fisher = {}
    for name, fisher in fisher_diag.items():
        x_min = fisher.min()
        x_max = fisher.max()
        if x_max - x_min > 1e-8:  # 避免除零
            normalized_fisher[name] = (fisher - x_min) / (x_max - x_min)
        else:
            normalized_fisher[name] = torch.zeros_like(fisher)

    # 转移到CPU并返回（与其他Fisher函数格式一致）
    return {k: v.detach() for k, v in normalized_fisher.items()} # <-- 修改



def run_federated_training(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: UIETrainingArguments, fed_args: FederatedArguments):
    world = max(getattr(training_args, "world_size", 1), 1)

    from accelerate.utils import set_seed
    set_seed(fed_args.federated_seed, device_specific=False)  # 仅供 Trainer/Sampler 等内部用


    def _is_main():
        return getattr(training_args, "process_index", 0) == 0

    # loading logging
    logging.basicConfig(format="%(message)s", handlers=[logging.StreamHandler()])
    logger.info("Running federated learning mode")
    # 强制将日志级别设置为 INFO，忽略 training_args 的默认值
    log_level = logging.INFO  # 直接使用 20（INFO 的级别值）
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu},"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # logger.info(f"Training/evaluation parameters {training_args}")

    # accelerator.seed_everything(fed_args.federated_seed)

    data_cache_dir = gen_cache_path(training_args.output_dir, data_args)
    raw_datasets = load_dataset(
        os.path.join(CURRENT_DIR, "uie_dataset_lora.py"),
        data_dir=data_args.data_dir,
        task_config_dir=data_args.task_config_dir,
        instruction_file=data_args.instruction_file,
        instruction_strategy=data_args.instruction_strategy,
        cache_dir=data_cache_dir,
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        num_examples=data_args.num_examples
    )
    raw_datasets.cleanup_cache_files()

    # Detecting last checkpoint (复用集中式逻辑)
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, "
                "change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    # ========== 数据集拆分 ==========
    # train dataset
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
    # eval dataset
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
    # predict dataset
    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # 按任务均匀采样的逻辑
            unique_tasks = set(predict_dataset["Dataset"])
            samples_per_task = data_args.max_predict_samples // len(unique_tasks)
            task_datasets = []
            for task in unique_tasks:
                task_data = predict_dataset.filter(lambda ex: ex["Dataset"] == task)
                task_data = task_data.shuffle(seed=training_args.seed).select(range(min(samples_per_task, len(task_data))))
                task_datasets.append(task_data)
            from datasets import concatenate_datasets
            predict_dataset = concatenate_datasets(task_datasets)

    all_metrics = {"run_name": training_args.run_name}

    # test_dataset = raw_datasets["test"] if training_args.do_predict else None

    client_datasets = partition_dataset(
                train_dataset,
                fed_args.num_clients,
                fed_args.dirichlet_alpha,
                base_seed = fed_args.federated_seed,  # 固定联邦种子
        )


    # compare(client_datasets,fed_args.dirichlet_alpha)
    model, tokenizer = build_model_and_tokenizer(model_args)

    # model = accelerator.prepare(model)

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    client_rng = random.Random(fed_args.federated_seed)

    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        # 对生成式模型的输出进行后处理
        print(type(preds), np.asarray(preds).dtype, np.asarray(preds).shape)
        decoded_preds = skip_instructions(model, preds, tokenizer)
        references = [e["Instance"]["label"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        # 按类别进行分类，考虑的是所有TC类的准确率
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                  groups=dataset["Task"])
        result.update(result_per_task)
        categories = dataset["Dataset"]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                      groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Dataset": example["Dataset"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result

    def collator_for(model):
        # 确保数据整理器始终拿到未被分布式/Accelerate 包装的原始模型，
        # 以便访问 config 以及 prepare_decoder_input_ids 等方法。
        base_model = model
        if hasattr(base_model, "module"):
            base_model = base_model.module

        # 跑LLAMA不需要
        return DataCollatorForUIE(
            tokenizer,
            model=base_model,
            padding="longest",
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            add_task_name=data_args.add_task_name,
            add_dataset_name=data_args.add_dataset_name,
            num_examples=data_args.num_examples,
            input_record_file=data_args.input_record_file,
        )

        #如果要跑T5，需要model=base_model

        # return DataCollatorForUIE(
        #     tokenizer,
        #     model=base_model,
        #     padding="longest",
        #     max_source_length=data_args.max_source_length,
        #     max_target_length=data_args.max_target_length,
        #     label_pad_token_id=label_pad_token_id,
        #     pad_to_multiple_of=8 if training_args.fp16 else None,
        #     add_task_name=data_args.add_task_name,
        #     add_dataset_name=data_args.add_dataset_name,
        #     num_examples=data_args.num_examples,
        #     input_record_file=data_args.input_record_file,
        # )

    def get_lora_trainable_keys(model):
        """
        提取模型中所有 LoRA 可训练参数的键名。
        """
        return [name for name, param in model.named_parameters() if param.requires_grad and 'lora' in name]

    def get_param_bit_width(param):
        """根据参数数据类型自动获取比特宽度"""
        dtype = param.dtype
        if dtype == torch.float32 or dtype == torch.int32:
            return 32
        elif dtype == torch.float16 or dtype == torch.int16 or dtype == torch.bfloat16:
            return 16
        elif dtype == torch.int8 or dtype == torch.uint8:
            return 8
        else:
            logger.warning(f"未知数据类型 {dtype}，默认使用32位计算")
            return 32

    def calculate_layer_packet_cost(param, packet_size=1500):
        """计算单个LoRA层所需的数据包数量"""
        num_elements = param.numel()
        bit_width = get_param_bit_width(param)
        total_bytes = (num_elements * bit_width) // 8  # 总字节数
        return (total_bytes + packet_size - 1) // packet_size  # 向上取整



    # 新增：跟踪客户端被选中的次数和最后选中轮次
    client_selection_tracker = {
        cid: {'count': 0, 'last_round': -1}
        for cid in range(fed_args.num_clients)
    }

    # -----Begin Training------
    training_args.remove_unused_columns = False
    base_args = copy.deepcopy(training_args)
    base_args.do_train = True
    base_args.do_eval = False
    base_args.do_predict = False
    method = base_args.method
    if _is_main():
        logger.info("Use method: {}".format(method))
    global_model = model
    global_model.to("cpu")  # <--- 新增：将全局模型移至CPU
    device = next(global_model.parameters()).device

    # 加载过去任务的fisher信息
    current_output_dir = training_args.output_dir
    # 解析当前任务序号（假设task_id已通过参数传入）
    current_task_id = data_args.task  # 例如：2
    prev_task_dir = None

    # 构造上一个任务的输出目录路径
    if current_task_id > 1:
        # 假设任务文件夹命名格式为 "{task_id}-{dataset_name}"
        # 提取当前目录的父路径（所有任务的共同上级目录）
        parent_dir = os.path.dirname(current_output_dir)
        # 上一个任务的序号
        prev_task_id = current_task_id - 1
        # 查找上一个任务的文件夹（匹配以f"{prev_task_id}-"开头的目录）
        for dir_name in os.listdir(parent_dir):
            if dir_name.startswith(f"{prev_task_id}-"):
                prev_task_dir = os.path.join(parent_dir, dir_name)
                break
        if prev_task_dir and not os.path.isdir(prev_task_dir):
            prev_task_dir = None

    # 新增：记录当前任务中被选中的客户端
    current_task_selected_clients = set()

    client_state_dir = os.path.join(current_output_dir, "client_states")
    os.makedirs(client_state_dir, exist_ok=True)

    # —— 任务级缓存：只记录“本任务”的 Fisher 与该任务最后一次的 θ*
    reduce_mode = getattr(fed_args, "fisher_reduce_mode", "last")  # "last" 或 "mean"
    per_task_cache = {
        cid: None  # 初始为 None，被选中后将存入 ContinualState 对象
        for cid in range(fed_args.num_clients)
    }

    lora_params = {k: p for k, p in global_model.named_parameters() if "lora" in k}
    layer_costs = {
        k: calculate_layer_packet_cost(p)
        for k, p in lora_params.items()
    }

    base_args.num_train_epochs = fed_args.local_epochs
    base_args.save_strategy = "no"
    base_args.logging_strategy = "no"
    base_args.evaluation_strategy = "no"

    if _is_main():
        logger.info("Initializing persistent DeepSpeed Trainer...")

    trainer = UIETrainer(
        model=global_model,
        args=base_args,
        train_dataset=client_datasets[0],  # <--- 占位符
        tokenizer=tokenizer,
        data_collator=collator_for(global_model),  # <--- 只创建一次
        compute_metrics=None,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
        state=None,
        comm_budget=None,
        layer_costs=None,
    )
    if _is_main():
        logger.info("DeepSpeed Trainer initialized.")




    for rnd in range(fed_args.global_rounds):
        if _is_main():
            logger.info(f"Global round {rnd + 1}/{fed_args.global_rounds}")

        # selected = client_rng.sample(
        #     range(fed_args.num_clients),
        #     min(fed_args.clients_per_round, fed_args.num_clients),
        # )

        def pick_clients(num_clients, clients_per_round, round_id, base_seed):
            rng = np.random.RandomState(base_seed + 10007 * round_id)
            k = min(clients_per_round, num_clients)
            return rng.choice(np.arange(num_clients), size=k, replace=False).tolist()

        selected = pick_clients(
                        fed_args.num_clients,
                        fed_args.clients_per_round,
                        rnd,
                        fed_args.federated_seed,
        )



        if _is_main():
            logger.info(f"Selected client {selected}")

        lora_keys = get_lora_trainable_keys(global_model)

        global_state_cpu = {
            k: v.detach().cpu()
            for k, v in global_model.state_dict().items()
            if k in lora_keys  # 只需要 LoRA 键
        }

        # 2. 将其转换为 GPU 字典，用于快速加载
        global_state_gpu = {
            k: v.to(device) for k, v in global_state_cpu.items()
        }

        aggregated = {k: torch.zeros_like(global_state_cpu[k]) for k in lora_keys}
        total = 0

        for cid in selected:
            if _is_main():
                logger.info(f"Client ID: {cid}")
                logger.info(f"Client {cid}: Resetting persistent trainer...")

            trainer.model.load_state_dict(global_state_gpu, strict=False)
            trainer.train_dataset = client_datasets[cid]

            trainer.optimizer = None
            trainer.lr_scheduler = None
            train_dataloader = trainer.get_train_dataloader()
            steps_per_epoch = len(train_dataloader) // trainer.args.gradient_accumulation_steps
            steps_per_epoch = max(steps_per_epoch, 1)  # 至少 1 步
            num_training_steps = math.ceil(trainer.args.num_train_epochs * steps_per_epoch)
            trainer.create_optimizer_and_scheduler(num_training_steps=num_training_steps)
            if hasattr(trainer, "adam_states"):
                trainer.adam_states = {}

            if method == "lora_origin":
                if _is_main():
                    logger.info(f"Client {cid}: Starting training (lora_origin)...")


                trainer.train(task_id=data_args.task)
                _trainer_wait_for_everyone(trainer)

                trained_model = _trainer_unwrap_model(trainer)
                name_to_param = dict(trained_model.named_parameters())
                delta = {}
                for k in lora_keys:
                    p = name_to_param.get(k, None)
                    if p is None:
                        continue  # 或者 log warning
                    delta[k] = global_state_cpu[k] - p.detach().cpu()

                w = len(client_datasets[cid])
                for k in lora_keys:
                    aggregated[k] += delta[k] * w
                total += w

                try:
                    del name_to_param, trained_model
                except Exception:
                    pass

            elif method == "adaptive":
                client_state = ContinualState()
                if prev_task_dir is not None:
                    prev_state_path = os.path.join(prev_task_dir, "client_states", f"client_{cid}_state.pt")
                    if os.path.exists(prev_state_path):
                        if _is_main():
                            logger.info(f"Client {cid}: Loading previous state from {prev_state_path}")
                        client_state = ContinualState.load(prev_state_path)
                    else:
                        if _is_main():
                            logger.info(f"Client {cid}: No previous state found at {prev_state_path}. Starting fresh.")
                else:
                    if _is_main():
                        logger.info(f"Client {cid}: prev_task_dir is None (Task {data_args.task}). Starting fresh.")

                if data_args.task == 1:
                    logger.info(f"Client {cid}: Starting training (adaptive task 1)...")

                    trainer.train(task_id=data_args.task)
                    _trainer_wait_for_everyone(trainer)
                    trained_model = _trainer_unwrap_model(trainer)
                    if _is_main():
                        fisher_dataloader = trainer.get_train_dataloader()

                    name_to_param = dict(trained_model.named_parameters())
                    delta = {}
                    theta_last_cpu = {}  # <-- 同时为 theta_last 做准备
                    for k in lora_keys:
                        p = name_to_param.get(k, None)
                        if p is None:
                            continue
                        p_cpu = p.detach().cpu()
                        delta[k] = global_state_cpu[k] - p_cpu
                        theta_last_cpu[k] = p_cpu  # 缓存CPU上的参数

                    w = len(client_datasets[cid])
                    for k in lora_keys:
                        aggregated[k] += delta[k] * w
                    total += w

                    if _is_main():
                        # --- [修改点 3：使用提前获取的 dataloader] ---
                        F_client = compute_fisher_diag(trained_model, fisher_dataloader)

                        theta_last = {k: theta_last_cpu[k] for k in F_client.keys() if k in theta_last_cpu}

                        # [!!! 修复 2/3: 更新 client_state 并存入 cache !!!]
                        client_state.update(F_client, theta_last)  # 此处 client_state 为空，被更新为 T=1 的状态
                        per_task_cache[cid] = client_state  # 将包含 T=1 状态的 *完整对象* 存入 cache

                        # 清理 dataloader
                        del fisher_dataloader

                    try:
                        del trained_model, name_to_param, delta, theta_last_cpu
                        if _is_main():
                            del F_client, theta_last
                    except Exception:
                        pass
                else:
                    trainer.continual_state = client_state
                    trainer.comm_budget = fed_args.comm_budget
                    trainer.layer_costs = layer_costs
                    logger.info(f"Client {cid}: Starting training (adaptive task > 1)...")
                    delta, F_client, theta_last = trainer.train(
                        task_id=data_args.task,
                        base_params={k: global_state_cpu[k] for k in lora_keys},
                        cid=cid
                    )
                    _trainer_wait_for_everyone(trainer)

                    delta = {k: delta[k].detach().cpu() for k in lora_keys}
                    w = len(client_datasets[cid])
                    for k in lora_keys:
                        aggregated[k] += delta[k] * w
                    total += w

                    if _is_main():
                        # [!!! 修复 2/3: 更新 client_state 并存入 cache !!!]
                        client_state.update(F_client, theta_last)  # 此处 client_state 包含 T-1 历史，被更新为 T 的状态
                        per_task_cache[cid] = client_state  # 将包含 T 状态的 *完整对象* 存入 cache

                    try:
                        del delta, F_client, theta_last
                    except Exception:
                        pass


        for cid in selected:
            client_selection_tracker[cid]['count'] += 1
            client_selection_tracker[cid]['last_round'] = rnd
            current_task_selected_clients.add(cid)

        for k in lora_keys:
            mu = aggregated[k] / max(total, 1)
            global_state_cpu[k] = global_state_cpu[k] - mu

        global_model.load_state_dict(global_state_cpu, strict=False)

        wait_for_everyone()

        try:
            del aggregated, global_state_cpu, global_state_gpu
        except Exception:
            pass
        gc.collect()


    # ===== 任务结束：将“本任务”信息并入历史 =====
    if method == "adaptive" and data_args.task >= 1:
        # [!!! 修复 3/3: 直接从 cache 保存已更新的 state !!!]
        # (原有的 gamma 和 reduce_mode 逻辑现在由 client_state.update() 内部处理，这里不再需要)

        for cid in current_task_selected_clients:
            client_state_updated = per_task_cache[cid]  # 获取已更新的完整状态对象 (T=1, or T=2, ...)

            # 没被选中的客户端，或 (仅 main 进程) 状态未被正确更新
            if client_state_updated is None:
                continue

            # 直接保存这个已包含(历史+当前)所有信息的对象
            # client_state_dir 是在 (约 647 行) 定义的 *当前任务* 输出目录
            client_state_path = os.path.join(client_state_dir, f"client_{cid}_state.pt")

            if _is_main():
                logger.info(f"Saving updated state for client {cid} to {client_state_path}")
                client_state_updated.save(client_state_path)

        wait_for_everyone()


    # ========== 保存 Adapter ==========
    peft_model_id = os.path.join(training_args.output_dir, "adapter")

    if _is_main():
        global_model.save_pretrained(peft_model_id)
        tokenizer.save_pretrained(peft_model_id)
        logger.info(f"Saved LoRA adapter/tokenizer to {peft_model_id}")
    all_metrics.update({"adapter_saved": peft_model_id})
    wait_for_everyone()
    # logger.info(f"Saved LoRA adapter/tokenizer to {peft_model_id}")

    # ========== 最终预测 & 指标记录 ==========
    if training_args.do_predict:
        if _is_main():
            logger.info("Initializing evaluator trainer for final prediction...")
            global_model.to(device)  # <--- 将最终的 CPU 模型移到 GPU
        eval_trainer = UIETrainer(
            model=global_model,
            args=training_args,
            train_dataset=None,
            eval_dataset=predict_dataset,
            tokenizer=tokenizer,
            data_collator=collator_for(global_model),
            compute_metrics=compute_rouge_metrics,
            callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
        )

        predict_results = eval_trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_new_tokens=training_args.generation_max_length or data_args.max_target_length,
            num_beams=data_args.num_beams or training_args.generation_num_beams,
            repetition_penalty=data_args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )
        metrics = predict_results.metrics
        metrics["predict_samples"] = min(data_args.max_predict_samples or len(predict_dataset),
                                         len(predict_dataset))
        eval_trainer.log_metrics("predict", metrics)
        eval_trainer.save_metrics("predict", metrics)
        all_metrics.update(metrics)
        logger.info(f"Final federated evaluation metrics: {metrics}")

    wait_for_everyone()

    return all_metrics

