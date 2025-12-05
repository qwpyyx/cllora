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
import gc
from accelerate.utils import wait_for_everyone
import math
from datasets import load_dataset
import transformers
import torch.distributed as dist
from transformers.trainer_utils import get_last_checkpoint
from uie_collator import DataCollatorForUIE
from uie_trainer_lora import UIETrainer, DenserEvalCallback, skip_instructions
from compute_metrics import compute_metrics, compute_grouped_metrics
# from model.llama import LlamaForCausalLM_with_lossmask
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from uie_dataset_lora import gen_cache_path
from run_uie_lora import ModelArguments, DataTrainingArguments, UIETrainingArguments, FederatedArguments
from fed_continual_state import ContinualState
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,  # <-- 确保引入这个
    LlamaTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    GenerationConfig
)

os.environ['WANDB_DISABLED'] = "True"
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


def build_model_and_tokenizer(model_args):
    """
    Unified loader for T5 and Llama in Federated Learning.
    - T5: Uses standard loading.
    - Llama: Uses Flash Attention 2 + BF16 + Custom Tokenizer Settings.
    """

    # --------- 1) 判别模型族 ----------
    name_lower = model_args.model_name_or_path.lower()
    is_adapter = ("adapter" in name_lower) or ("peft" in name_lower)
    is_llama = ("llama" in name_lower) or ("vicuna" in name_lower)

    print(f"[Build Model] Loading: {model_args.model_name_or_path} | Is Llama: {is_llama} | Is Adapter: {is_adapter}")

    # --------- 2) 准备 Config 和 Tokenizer ----------
    if is_adapter:
        peft_cfg = PeftConfig.from_pretrained(model_args.model_name_or_path)
        base_model_path = peft_cfg.base_model_name_or_path
    else:
        base_model_path = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(
        base_model_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # [通用设置] 训练时关闭 cache 以节省显存
    config.use_cache = False

    if is_llama:
        # Llama Tokenizer (新环境/旧环境都兼容)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # 1. 补全 pad_token (如果缺失)
        # Llama 原生通常没有 pad_token，优先使用 unk_token (id=0)
        if tokenizer.pad_token is None:
            if tokenizer.unk_token_id is not None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
                tokenizer.pad_token = tokenizer.unk_token
            else:
                # 兜底策略
                tokenizer.pad_token_id = 0
                tokenizer.pad_token = "<unk>"

        # 2. 强制修正 ID (避免 Pad=1 与 BOS=1 冲突)
        # 这是解决训练不收敛和预测乱码的关键
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0  # 必须是 0

        # 3. 设置左填充 (Left Padding)
        # Decoder-only 模型做生成任务时必须左填充，否则输出不对齐
        tokenizer.padding_side = "left"

        # 同步更新 Config，防止生成时报 Warning
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id

    else:
        # [T5 路径] 标准加载，完全兼容旧环境
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # T5 默认 pad_token_id=0, padding_side='right'，无需修改

    # --------- 3) 准备模型加载参数 ----------
    model_load_kwargs = {
        "config": config,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    # [Llama 专属优化]
    if is_llama:
        # 1. 精度选择: 优先 BF16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            model_load_kwargs["torch_dtype"] = torch.bfloat16
            print("[Build Model] Using bfloat16 for Llama.")
        else:
            model_load_kwargs["torch_dtype"] = torch.float16
            print("[Build Model] Using float16 for Llama.")

        # 2. Flash Attention 2 加速 (如果安装了)
        try:
            import flash_attn
            config._attn_implementation = "flash_attention_2"
            print("[Build Model] >>> USING FLASH ATTENTION 2 <<<")
        except ImportError:
            print("[Build Model] Flash Attention 2 not found, using default attention.")

    # --------- 4) 加载模型 ----------
    if is_llama:
        model_class = AutoModelForCausalLM
        lora_task_type = TaskType.CAUSAL_LM
    else:
        # T5 使用标准 Seq2Seq 类
        model_class = AutoModelForSeq2SeqLM
        lora_task_type = TaskType.SEQ_2_SEQ_LM

    # 加载 Base Model
    model = model_class.from_pretrained(
        base_model_path,
        from_tf=bool(".ckpt" in base_model_path),
        **model_load_kwargs
    )

    # [梯度检查点支持]
    # 开启 input_require_grads 以支持 gradient_checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # --------- 5) 应用 PEFT / LoRA ----------
    if is_adapter:
        print(f"[Build Model] Loading existing adapter: {model_args.model_name_or_path}")
        model = PeftModel.from_pretrained(
            model,
            model_args.model_name_or_path,
            torch_dtype=model_load_kwargs.get("torch_dtype", "auto")
        )
    else:
        print(f"[Build Model] Initializing new LoRA adapter (r={model_args.lora_dim})")
        peft_config = LoraConfig(
            task_type=lora_task_type,
            inference_mode=False,
            r=model_args.lora_dim,
            lora_alpha=32,
            lora_dropout=0.1,
            # Llama 需要指定 target_modules，T5 通常不需要(默认q,v)
            target_modules=["q_proj", "v_proj"] if is_llama else None
        )
        model = get_peft_model(model, peft_config)

    # --------- 6) 后处理 ----------
    # 调整 Embedding 大小以匹配 Tokenizer (防止 special tokens 越界)
    model.resize_token_embeddings(len(tokenizer))

    # 确保 LoRA 参数可训练 (双重保险)
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True

    # 打印可训练参数
    model.print_trainable_parameters()
    return model, tokenizer


def compute_fisher_diag(model, dataloader):
    """
    适配LoRA模型的对角线Fisher信息计算（显存优化修正版）
    修复了 LlamaModel 不接受 input_ids_wo_label 的问题
    """
    # 自动获取模型所在设备
    device = next(model.parameters()).device
    model.eval()

    # 初始化Fisher累积器 (CPU上)
    fisher_diag = {
        name: torch.zeros_like(param, device="cpu")
        for name, param in model.named_parameters()
        if param.requires_grad and "lora" in name
    }

    total_samples = 0

    for batch in dataloader:
        # --- [修改点 1] 数据清洗与迁移 ---
        # 定义模型 forward 不接受的参数列表
        keys_to_ignore = ["input_ids_wo_label", "loss_mask", "labels"]

        # 构建 inputs：只包含模型 forward 需要的参数 (input_ids, attention_mask 等)
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k not in keys_to_ignore
        }

        # 获取 labels 并移至设备 (计算 log_prob 需要)
        labels = batch.get("labels", None)
        if labels is None:
            continue
        labels = labels.to(device)

        # ---------------------------------

        # 前向传播 (构造计算图)
        # 现在 inputs 里已经没有 input_ids_wo_label 了，不会报错
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # 计算 Log Softmax
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        batch_size = logits.size(0)
        total_samples += batch_size

        for i in range(batch_size):
            sample_label = labels[i]
            # 忽略 padding 部分 (-100)
            valid_mask = sample_label != -100
            if not valid_mask.any():
                continue

            # 提取单个样本的 Log Prob
            # 注意：这里假设 log_probs 和 labels 长度是对齐的
            # 对于 CausalLM，通常 labels 会发生 shift，但如果你的 collator 已经处理好了对齐，这里就不动
            # 如果是标准 HuggingFace 输出，logits 长度通常等于 input_ids 长度

            # 截取有效长度防止越界
            seq_len = min(log_probs.size(1), sample_label.size(0))
            sample_log_prob_seq = log_probs[i, :seq_len, :]
            sample_label_seq = sample_label[:seq_len]
            valid_mask_seq = valid_mask[:seq_len]

            # Gather 正确类别的概率
            # sample_log_prob_seq: [seq_len, vocab_size] -> 选出 target token 的概率
            selected_log_probs = sample_log_prob_seq[torch.arange(seq_len, device=device), sample_label_seq]

            # 求和得到该样本的 log(p(y|x))
            sample_total_log_prob = selected_log_probs[valid_mask_seq].sum()

            model.zero_grad(set_to_none=True)

            # [显存优化关键点]
            # 1. create_graph=False: 我们只需要梯度值，不需要二阶导
            # 2. retain_graph: 只有在处理 Batch 中非最后一个样本时才需要保留图
            retain_graph = (i < batch_size - 1)

            try:
                grads = torch.autograd.grad(
                    sample_total_log_prob,
                    [param for name, param in model.named_parameters() if param.requires_grad and "lora" in name],
                    create_graph=False,
                    retain_graph=retain_graph,
                    allow_unused=True  # 防止部分 LoRA 层未参与计算报错
                )

                # 累积到 CPU
                for (name, param), grad in zip(fisher_diag.items(), grads):
                    if grad is not None:
                        # Fisher = E[grad^2]
                        fisher_diag[name].add_(grad.detach().cpu().pow(2))

            except RuntimeError as e:
                # 捕获可能的 OOM 或计算图错误，防止整个训练中断
                logger.warning(f"Skipping sample due to error: {e}")
                continue

        # 显式清理，防止显存碎片
        del outputs, logits, log_probs
        if 'grads' in locals(): del grads
        torch.cuda.empty_cache()

    # 平均化
    if total_samples > 0:
        for name in fisher_diag:
            fisher_diag[name] /= total_samples
    else:
        logger.warning("未处理有效样本，Fisher保持初始零值")

    # 归一化
    normalized_fisher = {}
    for name, fisher in fisher_diag.items():
        x_min = fisher.min()
        x_max = fisher.max()
        if x_max - x_min > 1e-8:
            normalized_fisher[name] = (fisher - x_min) / (x_max - x_min)
        else:
            normalized_fisher[name] = torch.zeros_like(fisher)

    return {k: v.detach() for k, v in normalized_fisher.items()}



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
    # === 修改开始: 使用上下文管理器确保只有主进程先执行数据生成 ===
    with training_args.main_process_first(desc="loading dataset"):
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
    # === 修改结束 ===

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

    # [修改] 集中管理 Gradient Checkpointing 逻辑
    if training_args.gradient_checkpointing:
        logger.info("Gradient Checkpointing enabled.")

        # 1. 开启 GC
        if hasattr(model, "gradient_checkpointing_enable"):
            # [修改] 旧版 API 不接受参数，直接调用
            model.gradient_checkpointing_enable()
            logger.info("Gradient Checkpointing enabled (Legacy Mode).")

        # 2. [关键补充] 只有开 GC 时，才强制开启输入层梯度
        # 这解决了 "does not have a grad_fn" 报错，同时不影响不开 GC 的情况
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    else:
        logger.info("Gradient Checkpointing DISABLED (per arguments).")


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
        radius=training_args.radius,
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

