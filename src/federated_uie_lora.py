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

import math
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import matplotlib.pyplot as plt
import transformers
from accelerate import PartialState
from transformers import set_seed
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

def partition_dataset(dataset, num_clients: int, alpha: float):
    label_key = "Dataset"
    label2indices = defaultdict(list)
    for idx, example in enumerate(dataset):
        label2indices[example[label_key]].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    for indices in label2indices.values():
        np.random.shuffle(indices)
        props = np.random.dirichlet([alpha] * num_clients)
        # 累计比例映射到具体的 split 位置
        bounds = (np.cumsum(props) * len(indices)).astype(int)[:-1]
        splits = np.split(np.array(indices), bounds)
        for cid, idxs in enumerate(splits):
            client_indices[cid].extend(idxs.tolist())

    # 修复空 client：从样本最多的 client 那里“偷”一个
    for cid, idxs in enumerate(client_indices):
        if len(idxs) == 0:
            donor = max(range(num_clients), key=lambda x: len(client_indices[x]))
            stolen = client_indices[donor].pop()
            client_indices[cid].append(stolen)

    return [dataset.select(idxs) for idxs in client_indices]

def build_model_and_tokenizer(model_args: ModelArguments):
    if 'adapter' in model_args.model_name_or_path:
        config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        if 'llama' in model_args.model_name_or_path.lower():
            tokenizer = transformers.LlamaTokenizer.from_pretrained(config.base_model_name_or_path)
            config.bos_token_id = 1
            config.eos_token_id = 2
            config.pad_token_id = 1
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
            tokenizer.pad_token_id = 1
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    elif 'llama' in model_args.model_name_or_path.lower():
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        config.bos_token_id = 1
        config.eos_token_id = 2
        config.pad_token_id = 1
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 1
    else:
        config = transformers.AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if 'llama' in model_args.model_name_or_path.lower():
        model_class = LlamaForCausalLM_with_lossmask
        tokenizer.padding_side = 'left'
    else:
        model_class = transformers.AutoModelForSeq2SeqLM

    if 'adapter' in model_args.model_name_or_path:
        model = model_class.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
    elif 'llama' in model_args.model_name_or_path.lower():
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool('.ckpt' in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None
        )
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=model_args.lora_dim, lora_alpha=32, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
    else:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool('.ckpt' in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=model_args.lora_dim, lora_alpha=32, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)

    model.resize_token_embeddings(len(tokenizer))

    if 'llama' in model_args.model_name_or_path.lower():
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        model.generation_config.pad_token_id = 1

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
        name: torch.zeros_like(param, device=device)
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
                    fisher_diag[name].add_(grad.detach() ** 2)

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
    return {k: v.detach().cpu() for k, v in normalized_fisher.items()}



def filter_lora_parameters(fisher_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """过滤仅保留包含"lora"的参数（与compute_fisher_diag保持一致）"""
    return {
        name: tensor for name, tensor in fisher_dict.items()
        if "lora" in name.lower()
    }


def normalize_fisher(fisher_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """对compute_fisher的结果执行min-max归一化（与compute_fisher_diag对齐）"""
    normalized = {}
    for name, tensor in fisher_dict.items():
        x_min = tensor.min()
        x_max = tensor.max()
        if x_max - x_min > 1e-8:
            normalized[name] = (tensor - x_min) / (x_max - x_min)
        else:
            normalized[name] = torch.zeros_like(tensor)
    return normalized


def compute_difference_metrics(
        f1: Dict[str, torch.Tensor],
        f2: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, float], float, float, float, float]:
    """计算两个Fisher结果的差异指标（支持字典格式输入）"""
    per_layer_metrics = {}
    all_f1 = []
    all_f2 = []

    # 获取共有的参数名（确保对比的是相同参数）
    common_keys = f1.keys() & f2.keys()
    if not common_keys:
        logging.warning("未找到共有的LoRA参数，无法计算差异指标")
        return {}, 0.0, 0.0, 0.0, 0.0

    for name in common_keys:
        tensor1 = f1[name].flatten().cpu().numpy()
        tensor2 = f2[name].flatten().cpu().numpy()

        # 层级指标
        abs_error = np.mean(np.abs(tensor1 - tensor2))
        rel_error = np.mean(np.abs(tensor1 - tensor2) / (np.abs(tensor1) + 1e-8))  # 避免除零
        per_layer_metrics[name] = {"abs_error": abs_error, "rel_error": rel_error}

        # 收集全局数据
        all_f1.extend(tensor1)
        all_f2.extend(tensor2)

    # 全局指标
    all_f1 = np.array(all_f1)
    all_f2 = np.array(all_f2)
    mean_abs_error = np.mean(np.abs(all_f1 - all_f2))
    mean_rel_error = np.mean(np.abs(all_f1 - all_f2) / (np.abs(all_f1) + 1e-8))
    l2_distance = np.sqrt(np.sum((all_f1 - all_f2) ** 2))
    pearson_corr, _ = pearsonr(all_f1, all_f2) if len(all_f1) > 1 else (0.0, 0.0)

    return per_layer_metrics, mean_abs_error, mean_rel_error, l2_distance, pearson_corr


def visualize_comparison(
        f1: Dict[str, torch.Tensor],
        f2: Dict[str, torch.Tensor],
        sample_layers: int = 3,
        save_dir: str = "."
) -> None:
    """可视化对比结果（直方图和散点图）"""
    common_keys = list(f1.keys() & f2.keys())
    if not common_keys:
        logging.warning("无共有的LoRA参数，跳过可视化")
        return

    # 随机选择样本层（最多sample_layers个）
    layers_to_plot = common_keys[:min(sample_layers, len(common_keys))]

    # 1. 直方图对比
    plt.figure(figsize=(15, 5))
    for i, name in enumerate(layers_to_plot):
        plt.subplot(1, sample_layers, i + 1)
        tensor1 = f1[name].flatten().cpu().numpy()
        tensor2 = f2[name].flatten().cpu().numpy()
        plt.hist(tensor1, bins=50, alpha=0.5, label="compute_fisher")
        plt.hist(tensor2, bins=50, alpha=0.5, label="compute_fisher_diag")
        plt.title(f"Layer: {name.split('.')[-1]}")  # 显示短名称
        plt.xlabel("Fisher Value")
        plt.ylabel("Count")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fisher_histogram_comparison.png")
    plt.close()

    # 2. 散点图（全局分布）
    plt.figure(figsize=(8, 8))
    all_f1 = np.concatenate([f1[name].flatten().cpu().numpy() for name in common_keys])
    all_f2 = np.concatenate([f2[name].flatten().cpu().numpy() for name in common_keys])
    plt.scatter(all_f1, all_f2, alpha=0.5, s=1)
    plt.plot([0, 1], [0, 1], 'r--')  # 归一化后的理想对角线（0-1范围）
    plt.xlabel("compute_fisher (Normalized)")
    plt.ylabel("compute_fisher_diag (Normalized)")
    plt.title("Fisher Value Scatter Plot (LoRA Parameters Only)")
    plt.savefig(f"{save_dir}/fisher_scatter_comparison.png")
    plt.close()


def compare_fishers(
        model: torch.nn.Module,
        dataloader: DataLoader,
        alpha: float = 0.5,
        save_dir: str = ".",
        sample_layers: int = 3
) -> None:
    """主函数：对比两个Fisher计算函数的结果"""
    # 1. 计算两个Fisher矩阵
    logging.info("开始计算Fisher矩阵...")
    fisher = compute_fisher(model, dataloader, alpha=alpha)  # EMA版本（所有可训练参数）
    fisher_diag = compute_fisher_diag(model, dataloader)  # 样本平均版本（仅LoRA参数）

    # 2. 过滤compute_fisher的结果：仅保留LoRA参数（与fisher_diag对齐）
    fisher_lora = filter_lora_parameters(fisher)
    logging.info(f"compute_fisher中LoRA参数数量: {len(fisher_lora)}")
    logging.info(f"compute_fisher_diag中LoRA参数数量: {len(fisher_diag)}")

    # 3. 归一化compute_fisher的LoRA参数（与fisher_diag的归一化方式一致）
    # fisher_lora_norm = normalize_fisher(fisher_lora)

    # 4. 计算差异指标
    per_layer_metrics, mean_abs, mean_rel, l2, corr = compute_difference_metrics(
        fisher_lora, fisher_diag
    )

    # 5. 输出量化结果
    logging.info("\n===== Fisher对比量化指标 =====")
    logging.info(f"共有的LoRA参数数量: {len(per_layer_metrics)}")
    logging.info(f"全局平均绝对误差: {mean_abs:.6f}")
    logging.info(f"全局平均相对误差: {mean_rel:.6f}")
    logging.info(f"全局L2距离: {l2:.6f}")
    logging.info(f"全局Pearson相关系数: {corr:.6f}")

    # 打印前5层的详细误差
    if per_layer_metrics:
        logging.info("\n===== 部分LoRA层的误差详情 =====")
        for name in list(per_layer_metrics.keys())[:5]:
            logging.info(
                f"层 {name}: 绝对误差={per_layer_metrics[name]['abs_error']:.6f}, "
                f"相对误差={per_layer_metrics[name]['rel_error']:.6f}"
            )

    # 6. 可视化
    visualize_comparison(fisher_lora, fisher_diag, sample_layers, save_dir)
    logging.info(f"可视化结果已保存至 {save_dir}")






def run_federated_training(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: UIETrainingArguments, fed_args: FederatedArguments):


    distributed_state = PartialState()
    world_size = getattr(distributed_state, "num_processes", 1)
    process_index = getattr(distributed_state, "process_index", 0)
    main_process = getattr(distributed_state, "main_process", 0)
    if isinstance(main_process, (list, tuple)):
        main_process = main_process[0]
    effective_world_size = max(int(world_size), 1)
    process_index = int(process_index) % effective_world_size


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

    set_seed(training_args.seed)

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

    client_datasets = partition_dataset(train_dataset, fed_args.num_clients, fed_args.dirichlet_alpha)
    # compare(client_datasets,fed_args.dirichlet_alpha)
    model, tokenizer = build_model_and_tokenizer(model_args)

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
        return DataCollatorForUIE(
            tokenizer,
            model=model,
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
    logger.info("Use method: {}".format(method))
    global_model = model
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
        cid: {"F_sum": None, "count": 0, "F_last": None, "theta_last": None}
        for cid in range(fed_args.num_clients)
    }

    lora_params = {k: p for k, p in global_model.named_parameters() if "lora" in k}
    # 预计算所有LoRA层的成本（只需要计算一次，全局共享）
    layer_costs = {
        k: calculate_layer_packet_cost(p)
        for k, p in lora_params.items()
    }
    logger.info(f"预计算的LoRA层通信成本: "
                f"{layer_costs['base_model.model.encoder.block.0.layer.0.SelfAttention.q.lora_A.default.weight']}")
    import torch.distributed as dist
    for rnd in range(fed_args.global_rounds):
        logger.info(f"Global round {rnd + 1}/{fed_args.global_rounds}")

        # selected = client_rng.sample(
        #     range(fed_args.num_clients),
        #     min(fed_args.clients_per_round, fed_args.num_clients),
        # )

        selected = None
        if distributed_state.is_main_process:
            selected = client_rng.sample(
                range(fed_args.num_clients),
                min(fed_args.clients_per_round, fed_args.num_clients),
            )

        if dist.is_available() and dist.is_initialized():
            selected_container = [selected]
            dist.broadcast_object_list(selected_container, src=main_process)
            selected = selected_container[0]
        if selected is None:
            selected = client_rng.sample(
                range(fed_args.num_clients),
                min(fed_args.clients_per_round, fed_args.num_clients),
            )

        # logger.info(f"Selected client {selected}")
        if distributed_state.is_main_process:
            logger.info(f"Selected client {selected}")

        lora_keys = get_lora_trainable_keys(global_model)
        global_state_cpu = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        aggregated = {k: torch.zeros_like(global_state_cpu[k]) for k in lora_keys}
        total = 0

        # for cid in selected:
        #     logger.info(f"Client ID: {cid}")
        #     # 更新客户端选择跟踪
        #     client_selection_tracker[cid]['count'] += 1
        #     client_selection_tracker[cid]['last_round'] = rnd
        #     current_task_selected_clients.add(cid)

        local_selected = [cid for idx, cid in enumerate(selected) if idx % effective_world_size == process_index]
        local_payloads = []

        for cid in local_selected:
            logger.info(f"Client ID: {cid}")
            local_model = copy.deepcopy(global_model)
            local_args = copy.deepcopy(base_args)
            local_args.resume_from_checkpoint = None
            local_args.num_train_epochs = fed_args.local_epochs
            local_args.save_strategy = "no"
            local_args.logging_strategy = "no"
            local_args.evaluation_strategy = "no"

            # TODO 考虑如何把deepspeed兼容
            if method == "lora_origin":
                local_args.deepspeed = None
                trainer = UIETrainer(
                    model=local_model,
                    args=local_args,
                    train_dataset=client_datasets[cid],
                    tokenizer=tokenizer,
                    data_collator=collator_for(local_model),
                    compute_metrics=None,
                    callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
                )
                trainer.train(task_id=data_args.task)

                state_dict = local_model.state_dict()
                delta = {
                    k: global_state_cpu[k] - state_dict[k].detach().cpu()
                    for k in lora_keys
                }

                payload = {
                    "cid": cid,
                    "weight": len(client_datasets[cid]),
                    "delta": delta,
                    "cache": None,
                }

            elif method == "adaptive":
                client_state = ContinualState()
                if prev_task_dir is not None:
                    prev_state_path = os.path.join(prev_task_dir, "client_states", f"client_{cid}_state.pt")
                    if os.path.exists(prev_state_path):
                        client_state = ContinualState.load(prev_state_path)

                local_args.deepspeed = None
                if data_args.task == 1:
                    trainer = UIETrainer(
                        model=local_model,
                        args=local_args,
                        train_dataset=client_datasets[cid],
                        tokenizer=tokenizer,
                        data_collator=collator_for(local_model),
                        compute_metrics=None,
                        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
                    )
                    trainer.train(task_id=data_args.task)

                    deepspeed_engine = trainer.deepspeed if trainer.deepspeed is not None else None

                    state_dict = local_model.state_dict()
                    delta = {
                        k: global_state_cpu[k] - state_dict[k].detach().cpu()
                        for k in lora_keys
                    }

                    # 新增：选择Fisher计算方式（通过参数控制，默认用原有EMA）
                    # use_arithmetic_fisher = getattr(fed_args, "use_arithmetic_fisher", False)
                    # if use_arithmetic_fisher:
                    #     # 调用新的算术平均Fisher函数
                    #     F_client = compute_fisher_arithmetic(local_model, trainer.get_train_dataloader())
                    # else:
                    #     # 保留原有EMA方式
                    # F_client = compute_fisher(local_model, trainer.get_train_dataloader())
                    F_client = compute_fisher_diag(local_model, trainer.get_train_dataloader())
                    # 确保F_client与base_params对齐（原有逻辑不变）
                    # F_client = {k: F_client.get(k, torch.zeros_like(global_state_cpu[k])) for k in lora_keys}
                    theta_last = {k: state_dict[k].detach().cpu() for k in F_client.keys()}

                    payload = {
                        "cid": cid,
                        "weight": len(client_datasets[cid]),
                        "delta": delta,
                        "cache": {"F_client": F_client, "theta_last": theta_last},
                    }



                    # ---------------------- 新增：两种Fisher计算方法对比 ----------------------
                    # 1. 计算算术平均方式的Fisher（假设compute_fisher_diag已实现）
                    # F_client_arithmetic = compute_fisher_diag(local_model, trainer.get_train_dataloader())

                    # # 2. 确保两种Fisher的参数键对齐（只对比共同参数）
                    # common_keys = set(F_client.keys()) & set(F_client_arithmetic.keys())
                    # if not common_keys:
                    #     logger.warning("两种Fisher计算方法无共同参数，无法对比")
                    # else:
                    #     # 3. 计算并记录对比指标
                    #     fisher_comparison = {}
                    #     for key in common_keys:
                    #         f_ema = F_client[key]
                    #         f_arith = F_client_arithmetic[key]
                    #
                    #         # 计算数值差异（绝对值、相对值）
                    #         abs_diff = torch.abs(f_ema - f_arith)
                    #         rel_diff = abs_diff / (torch.abs(f_ema) + 1e-8)  # 避免除以0
                    #
                    #         # 记录关键统计量
                    #         fisher_comparison[key] = {
                    #             "ema_mean": f_ema.mean().item(),
                    #             "arith_mean": f_arith.mean().item(),
                    #             "abs_diff_mean": abs_diff.mean().item(),
                    #             "rel_diff_mean": rel_diff.mean().item(),
                    #             "ema_max": f_ema.max().item(),
                    #             "arith_max": f_arith.max().item()
                    #         }
                    #
                    #     # 4. 输出对比结果（日志+文件）
                    #     logger.info(f"客户端 {cid} 两种Fisher计算方法对比（共同参数数：{len(common_keys)}）：")
                    #     for key, stats in fisher_comparison.items():
                    #         logger.info(
                    #             f"参数 {key}：EMA均值={stats['ema_mean']:.6f}, "
                    #             f"算术均值={stats['arith_mean']:.6f}, "
                    #             f"平均绝对差={stats['abs_diff_mean']:.6f}, "
                    #             f"平均相对差={stats['rel_diff_mean']:.6f}"
                    #         )
                    #
                    #     # 保存详细对比结果到文件
                    #     comparison_dir = os.path.join(training_args.output_dir, "fisher_comparison")
                    #     os.makedirs(comparison_dir, exist_ok=True)
                    #     comparison_path = os.path.join(comparison_dir, f"client_{cid}_round_{rnd}_fisher_compare.json")
                    #     with open(comparison_path, "w") as f:
                    #         json.dump(fisher_comparison, f, indent=2)
                    #     logger.info(f"Fisher对比结果已保存至：{comparison_path}")
                    # -------------------------------------------------------------------------
                    # compare_fishers(local_model, trainer.get_train_dataloader(),0.5,comparison_dir,3)

                else:
                    trainer = UIETrainer(
                        model=local_model,
                        args=local_args,
                        train_dataset=client_datasets[cid],
                        tokenizer=tokenizer,
                        data_collator=collator_for(local_model),
                        compute_metrics=None,
                        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
                        state=client_state,
                        comm_budget=fed_args.comm_budget,
                        layer_costs=layer_costs,
                    )
                    delta, F_client, theta_last = trainer.train(
                        task_id=data_args.task,
                        base_params={k: global_state_cpu[k] for k in lora_keys},
                        cid=cid
                    )

                # —— 轮内：累计/覆盖本任务 Fisher
                # acc = per_task_cache[cid]
                    payload = {
                        "cid": cid,
                        "weight": len(client_datasets[cid]),
                        "delta": delta,
                        "cache": {"F_client": F_client, "theta_last": theta_last},
                    }
            else:
                payload = {
                    "cid": cid,
                    "weight": len(client_datasets[cid]),
                    "delta": delta,
                    "cache": None,
                }
            local_payloads.append(payload)
                # if reduce_mode == "last":
                #     # 只保留最新一轮
                #     acc["F_last"] = {k: v.clone() for k, v in F_client.items()}
                # else:  # "mean"
                #     if acc["F_sum"] is None:
                #         acc["F_sum"] = {k: v.clone() for k, v in F_client.items()}
                #         acc["count"] = 1
                #     else:
                #         for k in F_client:
                #             acc["F_sum"][k] += F_client[k]
                #         acc["count"] += 1
        if dist.is_available() and dist.is_initialized():
            gathered_payloads = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_payloads, local_payloads)
            all_payloads = []
            for chunk in gathered_payloads:
                if chunk:
                    all_payloads.extend(chunk)
        else:
            all_payloads = list(local_payloads)

        for payload in all_payloads:
            cid = payload["cid"]
            client_selection_tracker[cid]['count'] += 1
            client_selection_tracker[cid]['last_round'] = rnd
            current_task_selected_clients.add(cid)

            delta = payload["delta"]
            weight = payload["weight"]

            # Server update
            # weight = len(client_datasets[cid])
            for k in lora_keys:
                aggregated[k] += delta[k] * weight
            total += weight

            if method == "adaptive" and payload.get("cache") is not None:
                F_client = payload["cache"].get("F_client")
                theta_last = payload["cache"].get("theta_last")
                if F_client is not None and theta_last is not None:
                    acc = per_task_cache[cid]
                    acc["theta_last"] = theta_last
                    if reduce_mode == "last":
                        acc["F_last"] = {k: v.clone() for k, v in F_client.items()}
                    else:
                        if acc["F_sum"] is None:
                            acc["F_sum"] = {k: v.clone() for k, v in F_client.items()}
                            acc["count"] = 1
                        else:
                            for k in F_client:
                                acc["F_sum"][k] += F_client[k]
                            acc["count"] += 1
        for k in lora_keys:
            mu = aggregated[k] / max(total, 1)
            global_state_cpu[k] = global_state_cpu[k] - mu

        update_dict = {k: global_state_cpu[k].to(device) for k in lora_keys}
        global_model.load_state_dict(update_dict, strict=False)
        distributed_state.wait_for_everyone()

    # ===== 任务结束：将“本任务”信息并入历史 =====
    if method == "adaptive" and data_args.task >= 1:
        gamma = 0.9
        for cid in current_task_selected_clients:
            cache = per_task_cache[cid]

            # 没被选中的客户端：跳过
            if (reduce_mode == "last" and cache["F_last"] is None) or \
                    (reduce_mode == "mean" and (cache["F_sum"] is None or cache["count"] == 0)):
                continue

            # 生成该客户端的 F_task 与 θ*
            if reduce_mode == "last":
                F_task = cache["F_last"]
            else:  # mean
                F_task = {k: cache["F_sum"][k] / cache["count"] for k in cache["F_sum"]}

            theta_star = cache["theta_last"]  # 该客户端本任务最后一次的本地参数

            # 载入历史（若有），执行指数累计
            client_state_path = os.path.join(client_state_dir, f"client_{cid}_state.pt")
            client_state = ContinualState.load(client_state_path)
            client_state.update(F_task, theta_star)
            # client_state.save(client_state_path)
            if distributed_state.is_main_process:
                client_state.save(client_state_path)
        distributed_state.wait_for_everyone()

    # ========== 保存 Adapter ==========
    peft_model_id = os.path.join(training_args.output_dir, "adapter")
    # global_model.save_pretrained(peft_model_id)
    # tokenizer.save_pretrained(peft_model_id)
    if distributed_state.is_main_process:
        global_model.save_pretrained(peft_model_id)
        tokenizer.save_pretrained(peft_model_id)
        logger.info(f"Saved LoRA adapter/tokenizer to {peft_model_id}")
    all_metrics.update({"adapter_saved": peft_model_id})
    # logger.info(f"Saved LoRA adapter/tokenizer to {peft_model_id}")

    # ========== 最终预测 & 指标记录 ==========
    if training_args.do_predict:
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

    # import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    distributed_state.wait_for_everyone()
    return all_metrics

