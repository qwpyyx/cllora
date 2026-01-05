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
from datasets import concatenate_datasets
from lorm_utils import lorm_aggregate, LormGlobalState



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
    # is_llama = ("llama" in name_lower) or ("vicuna" in name_lower)
    if "t5" in name_lower:
        is_llama = False
    else:
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
    [修改版] DDP 专用 Fisher 计算：
    1. 只计算本地数据的梯度平方和，不平均，不归一化。
    2. 包含 Llama/GPT 的 Logits Shift 修正。
    """
    device = next(model.parameters()).device
    model.eval()

    # 初始化累积矩阵 (CPU)
    fisher_sum = {
        name: torch.zeros_like(param, device="cpu")
        for name, param in model.named_parameters()
        if param.requires_grad and "lora" in name
    }

    local_samples = 0
    model.zero_grad()

    for step, batch in enumerate(dataloader):
        # 1. 数据移到 GPU
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ["input_ids_wo_label", "labels"]}
        if "labels" in batch:
            labels = batch["labels"].to(device)

        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # 2. [关键] Logits Shift (针对 Llama/GPT)
        is_causal = False
        if "llama" in getattr(model.config, "_name_or_path", "").lower() or getattr(model.config, "is_decoder", False):
            if not getattr(model.config, "is_encoder_decoder", False):
                is_causal = True

        if is_causal:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        # 3. 计算梯度
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        batch_size = logits.size(0)
        local_samples += batch_size

        # 假设 dataloader batch_size=1，直接取 [0]
        # 如果 batch_size > 1，这里需要循环处理，但为了显存通常设为 1
        for i in range(batch_size):
            sample_label = labels[i]
            valid_mask = sample_label != -100
            if not valid_mask.any(): continue

            seq_len = min(log_probs.size(1), sample_label.size(0))
            sample_log_prob_seq = log_probs[i, :seq_len, :]
            sample_label_seq = sample_label[:seq_len]

            # Gather log probs
            selected_log_probs = sample_log_prob_seq[torch.arange(seq_len, device=device), sample_label_seq]
            sample_total_log_prob = selected_log_probs[valid_mask[:seq_len]].sum()

            model.zero_grad()
            sample_total_log_prob.backward()

            # 4. [关键] 累积到 CPU
            for name, param in model.named_parameters():
                if param.requires_grad and "lora" in name and param.grad is not None:
                    fisher_sum[name] += param.grad.detach().cpu().pow(2)

            model.zero_grad(set_to_none=True)

    # 返回 原始平方和 和 样本数量
    return fisher_sum, local_samples

def select_layers_random(layer_names, layer_costs, budget, seed):
    """
    Randomly select layers until budget is exhausted.
    """
    rng = random.Random(seed)
    # Shuffle indices
    indices = list(range(len(layer_names)))
    rng.shuffle(indices)

    selected_layers = set()
    current_cost = 0

    for idx in indices:
        name = layer_names[idx]
        cost = layer_costs.get(name, 0)
        if current_cost + cost <= budget:
            selected_layers.add(name)
            current_cost += cost

    return selected_layers, current_cost


def select_layers_topk(delta_dict, layer_costs, budget):
    """
    Select layers based on L2 norm of the update (Top-K importance) until budget is exhausted.
    """
    # 1. Calculate Importance Score (L2 Norm of Delta)
    layer_scores = []
    for name, tensor in delta_dict.items():
        score = torch.norm(tensor.float()).item()
        layer_scores.append((name, score))

    # 2. Sort by score descending
    layer_scores.sort(key=lambda x: x[1], reverse=True)

    # 3. Greedy selection
    selected_layers = set()
    current_cost = 0

    for name, score in layer_scores:
        cost = layer_costs.get(name, 0)
        if current_cost + cost <= budget:
            selected_layers.add(name)
            current_cost += cost

    return selected_layers, current_cost

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

    client_ewc_states = None
    client_replay_buffers = None
    baseline_save_dir = os.path.join(current_output_dir, "baseline_states")

    lorm_global_state = None
    if method == "lorm":
        lorm_global_state = LormGlobalState()
        if prev_task_dir:
            prev_lorm_path = os.path.join(prev_task_dir, "baseline_states", "lorm_global.pt")
            lorm_global_state.load(prev_lorm_path)

    if method == "lorm" and prev_task_dir:
        prev_lorm_path = os.path.join(prev_task_dir, "baseline_states", "lorm_global.pt")
        if os.path.exists(prev_lorm_path):
            logger.info(f"[LoRM] Recovering Backbone from {prev_lorm_path}...")
            state = LormGlobalState()
            state.load(prev_lorm_path)
            # 这会将历史累积的 DeltaW 加载到当前新初始化的 Backbone 中
            state.merge_and_apply(model, device=device)



    if method == "ewc":
        client_ewc_states = defaultdict(lambda: {"fisher": None, "params": None})
        # 如果找到了上个任务目录，尝试加载 EWC 状态
        if prev_task_dir:
            prev_baseline_dir = os.path.join(prev_task_dir, "baseline_states")
            if os.path.exists(prev_baseline_dir):
                if _is_main(): logger.info(f"Loading EWC states from {prev_baseline_dir}")
                for cid in range(fed_args.num_clients):
                    p = os.path.join(prev_baseline_dir, f"ewc_{cid}.pt")
                    if os.path.exists(p):
                        client_ewc_states[cid] = torch.load(p, map_location="cpu")
            else:
                if _is_main(): logger.warning(f"Previous task found but no baseline_states dir at {prev_baseline_dir}")


    elif method in ["replay", "gem"]:
        client_replay_buffers = defaultdict(list)
        # 如果找到了上个任务目录，尝试加载 Replay Buffer
        if prev_task_dir:
            prev_baseline_dir = os.path.join(prev_task_dir, "baseline_states")
            if os.path.exists(prev_baseline_dir):
                from datasets import load_from_disk
                if _is_main(): logger.info(f"Loading Replay buffers from {prev_baseline_dir}")
                for cid in range(fed_args.num_clients):
                    p = os.path.join(prev_baseline_dir, f"replay_{cid}")
                    if os.path.exists(p):
                        try:
                            ds = load_from_disk(p)
                            client_replay_buffers[cid] = [ds]
                        except Exception as e:
                            logger.warning(f"Failed to load replay buffer for client {cid}: {e}")



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

    current_task_ewc_cache = {}
    current_task_replay_cache = {}

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

        if method == "lorm":
            # 偶数轮 (0, 2...) 训练 B，奇数轮 (1, 3...) 训练 A
            target_matrix = "B" if (rnd % 2 == 0) else "A"
            freeze_target = "A" if target_matrix == "B" else "B"

            if _is_main():
                logger.info(f"Global Round {rnd + 1} [LoRM]: Training '{target_matrix}', Freezing '{freeze_target}'")

            # 设置 Global Model 的参数冻结状态 (作为 Client 的初始状态)
            for n, p in global_model.named_parameters():
                if "lora_" + freeze_target in n:
                    p.requires_grad = False
                elif "lora_" + target_matrix in n:
                    p.requires_grad = True

            # 初始化收集列表
            lorm_client_updates = []

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

            if method == "lorm":
                # 1. 训练前准备
                if _is_main():
                    logger.info(f"Client {cid}: Starting training (Method: {method})...")

                # 设置冻结状态
                for n, p in trainer.model.named_parameters():
                    if "lora_" + freeze_target in n:
                        p.requires_grad = False
                    elif "lora_" + target_matrix in n:
                        p.requires_grad = True

                trainer.optimizer = None
                trainer.lr_scheduler = None

                # 2. 标准训练 (自动触发 Gram 计算)
                trainer.train(task_id=data_args.task, cid=cid)

                # 3. [后处理] 提取全量参数 & Grams
                model_to_save = _trainer_unwrap_model(trainer)

                # 只取本轮训练的目标矩阵 (A 或 B)
                # 类似于你 EWC 取 params
                client_params = {
                    k: v.detach().cpu().clone()
                    for k, v in model_to_save.state_dict().items()
                    if "lora" in k and target_matrix in k
                }
                client_grams = getattr(trainer, "lorm_grams", {})

                # 4. [通信控制] 稀疏化筛选
                if fed_args.comm_budget is not None and fed_args.comm_budget > 0:

                    # 准备 Cost 字典 (Parameter + Gram)
                    # 我们需要手动构建这个 layer_costs，因为 select_layers_* 函数只认 params
                    # 但 LoRM 的 Cost 是 Param + Gram

                    lorm_layer_costs = {}
                    candidate_keys = list(client_params.keys())

                    for k in candidate_keys:
                        # 基础 Cost: 参数本身
                        cost = calculate_layer_packet_cost(client_params[k])

                        # 额外 Cost: 对应的 Gram
                        layer_prefix = k.split("lora_")[0]
                        for g_key, g_val in client_grams.items():
                            if layer_prefix in g_key:
                                cost += calculate_layer_packet_cost(g_val)
                                break
                        lorm_layer_costs[k] = cost

                    # 调用你的通用筛选函数
                    selected_layers = set()
                    selection_cost = 0

                    if training_args.random_layer_selection:
                        # Random Selection (完全复用你的逻辑)
                        seed = fed_args.federated_seed + rnd + cid
                        # 注意：select_layers_random 的第一个参数通常是 key 列表
                        selected_layers, selection_cost = select_layers_random(
                            candidate_keys, lorm_layer_costs, fed_args.comm_budget, seed
                        )
                        if _is_main():
                            logger.info(
                                f"Client {cid} [LoRM Random]: Selected {len(selected_layers)} layers, cost {selection_cost}/{fed_args.comm_budget}")

                    else:
                        # Top-K (Norm-based)
                        # 注意：select_layers_topk 通常接受 (delta, costs)
                        # 这里我们传 (client_params, costs)，因为它会算 value 的 norm
                        selected_layers, selection_cost = select_layers_topk(
                            client_params, lorm_layer_costs, fed_args.comm_budget
                        )
                        if _is_main():
                            logger.info(
                                f"Client {cid} [LoRM Top-K]: Selected {len(selected_layers)} layers, cost {selection_cost}/{fed_args.comm_budget}")

                    # 5. [执行稀疏化] 剔除未选中的层
                    # 区别于 FedAvg 的 Mask(置0)，这里直接删除 Key
                    keys_to_drop = [k for k in client_params.keys() if k not in selected_layers]
                    for k in keys_to_drop:
                        del client_params[k]

                    # Gram 矩阵也要对应筛选
                    grams_to_keep = {}
                    for k in selected_layers:
                        layer_prefix = k.split("lora_")[0]
                        for g_key, g_val in client_grams.items():
                            if layer_prefix in g_key:
                                grams_to_keep[g_key] = g_val
                                break
                    client_grams = grams_to_keep

                # 5. 存入队列 (等待聚合)
                lorm_client_updates.append({
                    "state_dict": client_params,
                    "grams": client_grams
                })

                # 显式清理
                try:
                    del model_to_save
                except:
                    pass

                continue



            if method in ["lora_origin", "ewc", "replay", "gem"]:
                if _is_main():
                    logger.info(f"Client {cid}: Starting training (lora_origin)...")

                # --- [Replay] 数据增强: 拼接旧数据 ---
                if method == "replay" and client_replay_buffers is not None:
                    if len(client_replay_buffers[cid]) > 0:
                        # 拼接所有历史 buffer
                        replay_ds = concatenate_datasets(client_replay_buffers[cid])
                        # 混合当前数据 + 回放数据
                        combined_ds = concatenate_datasets([client_datasets[cid], replay_ds])
                        # 打乱并赋值给 trainer
                        trainer.train_dataset = combined_ds.shuffle(seed=training_args.seed)
                        if _is_main():
                            logger.info(
                                f"Client {cid} [Replay]: Merged buffer. Size {len(client_datasets[cid])} -> {len(trainer.train_dataset)}")
                    else:
                        if _is_main(): logger.info(
                            f"Client {cid} [Replay]: Buffer empty, training on current task only.")

                elif method == "gem" and client_replay_buffers is not None:
                    if len(client_replay_buffers[cid]) > 0:
                        gem_memory_ds = concatenate_datasets(client_replay_buffers[cid])
                        # GEM 核心：独立设置 gem_dataset
                        trainer.gem_dataset = gem_memory_ds
                        trainer.init_gem_loader()  # <--- 手动触发 Loader 初始化
                        if _is_main():
                            logger.info(f"✅ Client {cid} [GEM]: Loaded memory constraint. Size: {len(gem_memory_ds)}")
                    else:
                        # 确保清理状态
                        trainer.gem_dataset = None
                        trainer.gem_loader = None
                        if _is_main():
                            logger.info(f"⚠️ Client {cid} [GEM]: Memory empty (Normal for Task 1). Running standard training.")
                else:
                    # 确保 GEM 状态被清理，防止污染其他算法
                    trainer.gem_dataset = None
                    trainer.gem_loader = None

                # --- [EWC] 注入状态: 设置 Fisher 和 Theta_star ---
                if method == "ewc" and client_ewc_states is not None:
                    state = client_ewc_states[cid]

                    # =========== [新增 Log 开始] ===========
                    if _is_main():
                        if state["fisher"] is None:
                            # 只有 Task 1 应该是这种情况
                            logger.info(f"Client {cid} [EWC Status]: ⚪ No Fisher matrix found (Normal for Task 1).")
                        else:
                            # 检查 Fisher 是否全为 0，或者是否有值
                            fisher_keys = list(state["fisher"].keys())
                            sample_key = fisher_keys[0]
                            sample_val = state["fisher"][sample_key].mean().item()
                            total_params = len(fisher_keys)
                            logger.info(f"Client {cid} [EWC Status]: 🟢 Fisher Loaded! "
                                        f"Count: {total_params} layers | "
                                        f"Sample Layer ({sample_key}) Mean: {sample_val:.6f}")
                            # 再次确认 params 是否存在
                            if state["params"] is None:
                                logger.error(
                                    f"Client {cid} [EWC Error]: 🔴 Fisher exists but Reference Params (theta*) is None!")
                            # =========== [新增 Log 结束] ===========

                            # 将状态注入 Trainer (Trainer.compute_loss 会用到)
                    trainer.ewc_fisher = state["fisher"]
                    trainer.ewc_params = state["params"]
                    if state["fisher"] is not None and _is_main():
                        logger.info(f"Client {cid} [EWC]: Loaded regularization constraints.")

                trainer.train(task_id=data_args.task)
                _trainer_wait_for_everyone(trainer)
                trained_model = _trainer_unwrap_model(trainer)

                if method == "ewc":
                    if _is_main():
                        logger.info(
                            f"Client {cid} [EWC]: Computing Fisher Matrix (Round {rnd + 1}) [Accelerate Distributed]...")

                    # 1. 数据采样 (所有 Rank 必须一致)
                    FISHER_SAMPLE_LIMIT = 500
                    fisher_ds = client_datasets[cid]
                    if len(fisher_ds) > FISHER_SAMPLE_LIMIT:
                        rng = np.random.RandomState(fed_args.federated_seed + cid + rnd)
                        indices = rng.choice(len(fisher_ds), FISHER_SAMPLE_LIMIT, replace=False)
                        fisher_ds = fisher_ds.select(indices)

                    # 2. 分布式 DataLoader (batch_size=1 保显存)
                    fisher_sampler = torch.utils.data.distributed.DistributedSampler(
                        fisher_ds, shuffle=False, drop_last=False
                    )
                    fisher_collator = collator_for(trained_model)
                    fisher_loader = torch.utils.data.DataLoader(
                        fisher_ds, batch_size=1, sampler=fisher_sampler, collate_fn=fisher_collator
                    )

                    # 3. 并行计算局部 Fisher
                    local_fisher_sum, local_count = compute_fisher_diag(trained_model, fisher_loader)

                    # 4. 聚合 (Reduce Sum)
                    local_count_tensor = torch.tensor(local_count, device=trainer.accelerator.device)
                    total_count_tensor = trainer.accelerator.reduce(local_count_tensor, reduction="sum")
                    total_samples = total_count_tensor.item()

                    sorted_keys = sorted(local_fisher_sum.keys())
                    final_fisher = {}

                    for name in sorted_keys:
                        local_tensor = local_fisher_sum[name].to(trainer.accelerator.device)
                        global_sum_tensor = trainer.accelerator.reduce(local_tensor, reduction="sum")

                        # 计算平均值
                        if total_samples > 0:
                            avg_fisher = global_sum_tensor / total_samples
                        else:
                            avg_fisher = global_sum_tensor

                        final_fisher[name] = avg_fisher.cpu()

                    # 5. 归一化 & 存入缓存 (仅主进程)
                    if _is_main():
                        all_values = torch.cat([f.flatten() for f in final_fisher.values()])
                        x_min, x_max = all_values.min(), all_values.max()

                        normalized_fisher = {}
                        for k, v in final_fisher.items():
                            if x_max - x_min > 1e-8:
                                normalized_fisher[k] = (v - x_min) / (x_max - x_min)
                            else:
                                normalized_fisher[k] = torch.zeros_like(v)

                        # [关键] 覆写缓存：保留该 Client 在本任务最后一次的状态
                        curr_params = {k: p.detach().cpu().clone() for k, p in trained_model.named_parameters() if
                                       p.requires_grad and "lora" in k}
                        current_task_ewc_cache[cid] = {
                            "fisher": normalized_fisher,
                            "params": curr_params
                        }
                        logger.info(f"Client {cid} [EWC]: Distributed Fisher finished. Total samples: {total_samples}")

                    # 清理
                    del fisher_loader, local_fisher_sum, final_fisher
                    torch.cuda.empty_cache()
                    trainer.accelerator.wait_for_everyone()

                # --- [Replay] 后处理: 采样并存入 Buffer ---
                if method in ["replay", "gem"] and client_replay_buffers is not None:
                    current_ds = client_datasets[cid]
                    buffer_size = training_args.replay_buffer_size

                    # 1. 采样逻辑 (保持不变)
                    if len(current_ds) > buffer_size:
                        # 使用 seed 确保可复现，这里加上 rnd 防止每一轮采的一模一样(虽然后面会覆写，但保持随机性更好)
                        rng = np.random.RandomState(fed_args.federated_seed + rnd + cid)
                        indices = rng.choice(len(current_ds), buffer_size, replace=False)
                        sampled_ds = current_ds.select(indices)
                    else:
                        sampled_ds = current_ds

                    # 2. [关键修改] 存入 Cache，而不是直接 append 到持久化 buffer
                    # 效果：
                    # A. 如果 Client 在本任务中多次被选中，旧的采样会被新的覆盖 -> 保证每个任务只存一份数据
                    # B. 只要被选中过一次，就会被记录 -> 解决了遗漏问题
                    current_task_replay_cache[cid] = sampled_ds

                    if _is_main():
                        logger.info(f"Client {cid} [{method.upper()}]: Cached {len(sampled_ds)} examples for next task.")


                name_to_param = dict(trained_model.named_parameters())
                delta = {}

                # 1. Calculate Full Delta first
                for k in lora_keys:
                    p = name_to_param.get(k, None)
                    if p is None:
                        continue  # 或者 log warning
                    delta[k] = global_state_cpu[k] - p.detach().cpu()

                # 2. Apply Selection Strategy (Compressed Upload)
                if fed_args.comm_budget is not None and fed_args.comm_budget > 0:
                    selected_layers = set()
                    selection_cost = 0

                    if training_args.random_layer_selection:
                        # Random Selection
                        seed = fed_args.federated_seed + rnd + cid
                        selected_layers, selection_cost = select_layers_random(
                            lora_keys, layer_costs, fed_args.comm_budget, seed
                        )
                        if _is_main():
                            logger.info(
                                f"Client {cid} [Random]: Selected {len(selected_layers)} layers, cost {selection_cost}/{fed_args.comm_budget}")
                    else:
                        # Top-K (Norm-based) Selection
                        selected_layers, selection_cost = select_layers_topk(
                            delta, layer_costs, fed_args.comm_budget
                        )
                        if _is_main():
                            logger.info(
                                f"Client {cid} [Top-K]: Selected {len(selected_layers)} layers, cost {selection_cost}/{fed_args.comm_budget}")

                    # 3. Mask unselected layers (set delta to 0)
                    for k in delta:
                        if k not in selected_layers:
                            delta[k] = torch.zeros_like(delta[k])




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

                    # =========================================================================
                    # [修改点] Adaptive Task 1: 分布式 Fisher 计算 (替换原有的 _is_main 块)
                    # =========================================================================
                    if _is_main():
                        logger.info(f"Client {cid} [Adaptive Task 1]: Computing Fisher (Distributed)...")

                    ADAPTIVE_SAMPLE_LIMIT = 500
                    fisher_ds = client_datasets[cid]
                    if len(fisher_ds) > ADAPTIVE_SAMPLE_LIMIT:
                        # 确保所有 rank 随机种子一致，这样大家看到的“全集”是一样的
                        rng = np.random.RandomState(fed_args.federated_seed + cid + 1)
                        indices = rng.choice(len(fisher_ds), ADAPTIVE_SAMPLE_LIMIT, replace=False)
                        fisher_ds = fisher_ds.select(indices)

                    # 2. 分布式 DataLoader (所有进程都执行 !!!)
                    # DistributedSampler 会根据当前 rank 自动分发属于它的数据切片
                    fisher_sampler = torch.utils.data.distributed.DistributedSampler(
                        fisher_ds, shuffle=False, drop_last=False
                    )
                    fisher_collator = collator_for(trained_model)
                    fisher_loader = torch.utils.data.DataLoader(
                        fisher_ds,
                        batch_size=1,
                        sampler=fisher_sampler,
                        collate_fn=fisher_collator
                    )

                    # 3. 并行计算局部 Fisher (所有进程都执行 !!!)
                    local_fisher_sum, local_count = compute_fisher_diag(trained_model, fisher_loader)

                    # 4. 使用 Accelerate 聚合结果 (所有进程都执行 !!!)
                    # A. 聚合样本总数
                    local_count_tensor = torch.tensor(local_count, device=trainer.accelerator.device)
                    # reduce 是集合通信操作，所有进程必须同时到达这里
                    total_count_tensor = trainer.accelerator.reduce(local_count_tensor, reduction="sum")
                    total_samples = total_count_tensor.item()

                    # B. 聚合 Fisher 矩阵 (所有进程都执行 !!!)
                    sorted_keys = sorted(local_fisher_sum.keys())
                    final_fisher = {}

                    for name in sorted_keys:
                        local_tensor = local_fisher_sum[name].to(trainer.accelerator.device)
                        # reduce 是集合通信操作
                        global_sum_tensor = trainer.accelerator.reduce(local_tensor, reduction="sum")

                        # 计算平均值 (每个进程都算一份，虽然最后只有 Rank 0 用，但为了逻辑对称没问题)
                        if total_samples > 0:
                            avg_fisher = (global_sum_tensor / total_samples).cpu()
                        else:
                            avg_fisher = global_sum_tensor.cpu()

                        final_fisher[name] = avg_fisher

                    # 5. [仅主进程] 归一化 -> 更新状态 -> 存入 Cache
                    # 这里才需要缩进
                    if _is_main():
                        # =========== [关键补丁] 手动执行 Min-Max 归一化 ===========
                        all_values = torch.cat([f.flatten() for f in final_fisher.values()])
                        x_min, x_max = all_values.min(), all_values.max()

                        normalized_fisher = {}
                        for k, v in final_fisher.items():
                            if x_max - x_min > 1e-8:
                                normalized_fisher[k] = (v - x_min) / (x_max - x_min)
                            else:
                                normalized_fisher[k] = torch.zeros_like(v)
                        # ========================================================

                        theta_last = {k: theta_last_cpu[k] for k in normalized_fisher.keys() if k in theta_last_cpu}

                        client_state.update(normalized_fisher, theta_last)
                        per_task_cache[cid] = client_state

                        logger.info(
                            f"Client {cid} [Adaptive Task 1]: State initialized (Normalized) and cached. Samples: {total_samples}")

                    # 6. 清理资源 & 同步
                    del fisher_loader, local_fisher_sum, final_fisher, fisher_sampler
                    torch.cuda.empty_cache()
                    # 必须等待 Rank 0 存完 Cache
                    trainer.accelerator.wait_for_everyone()

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

        if method == "lorm" and len(lorm_client_updates) > 0:
            if _is_main():
                logger.info(f"[LoRM] Aggregating {target_matrix} Matrix (Spatial)...")

                # 1. 空间聚合 (Spatial Aggregation): 合并 Client 参数到 Global Model
                # 这一步让 Global Model 在当前任务上性能最优
                new_state = lorm_aggregate(
                    lorm_client_updates,
                    global_model,
                    target_matrix=target_matrix,
                    device=device
                )
                global_model.load_state_dict(new_state, strict=False)

                # 同步 CPU 状态供下轮分发
                global_state_cpu = {
                    k: v.cpu() for k, v in global_model.state_dict().items()
                    if "lora" in k
                }

                # 2. 时间聚合 (Temporal Aggregation): 更新全局历史记忆
                # [关键] 必须在最后一轮 (Last Round) 执行，否则数据就被 del 了
                if rnd == fed_args.global_rounds - 1:
                    logger.info("[LoRM] End of Task: Updating Global History (Temporal)...")

                    # (A) 收集本任务的 Gram 矩阵统计量
                    # 我们使用最后一轮所有参与 Client 的 Gram 矩阵之和作为本任务数据的近似
                    task_grams = {}
                    for client in lorm_client_updates:
                        for k, g in client['grams'].items():
                            # 累加到 CPU 避免显存溢出
                            if k not in task_grams:
                                task_grams[k] = g.cpu()
                            else:
                                task_grams[k] += g.cpu()

                    # (B) 更新全局状态 (LormGlobalState)
                    # 这会计算 DeltaW = B*A，并累积 Sum(DeltaW * G) 和 Sum(G)
                    lorm_global_state.update(global_model, task_grams, device=device)



                    # (D) 保存状态到磁盘 (供下一个 Task 加载)
                    # 路径: .../outputs/task_id/baseline_states/lorm_global.pt
                    os.makedirs(baseline_save_dir, exist_ok=True)
                    save_path = os.path.join(baseline_save_dir, "lorm_global.pt")
                    lorm_global_state.save(save_path)
                    logger.info(f"[LoRM] Global state saved to {save_path}")

            # 等待所有进程完成
            wait_for_everyone()

            # 清理内存 (这也是为什么不能在循环外做的原因)
            del lorm_client_updates
            torch.cuda.empty_cache()

        if method != "lorm":
            for k in lora_keys:
                mu = aggregated[k] / max(total, 1)
                global_state_cpu[k] = global_state_cpu[k] - mu

            global_model.load_state_dict(global_state_cpu, strict=False)




        # for k in lora_keys:
        #     mu = aggregated[k] / max(total, 1)
        #     global_state_cpu[k] = global_state_cpu[k] - mu
        #
        # global_model.load_state_dict(global_state_cpu, strict=False)

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


    if _is_main() and (method == "ewc" or method == "replay" or method == "gem"):
        # baseline_save_dir 在循环外已定义 (current_output_dir/baseline_states)
        os.makedirs(baseline_save_dir, exist_ok=True)

        # [EWC 保存逻辑重写]
        if method == "ewc":
            logger.info(f"Saving EWC states to {baseline_save_dir}...")

            # 遍历所有客户端
            for cid in range(fed_args.num_clients):

                # 1. 获取旧状态 (Task T-1)
                # 如果 client_ewc_states 是 None (Task 1)，则视为全空
                old_state = {"fisher": None, "params": None}
                if client_ewc_states is not None and cid in client_ewc_states:
                    old_state = client_ewc_states[cid]

                # 2. 获取当前任务的新状态 (Task T)
                new_update = current_task_ewc_cache.get(cid, None)

                # 3. 合并逻辑
                final_fisher = None
                final_params = None

                # 如果该 Client 在本轮任务中从未被选中 (new_update is None)
                # 则它没有学到新知识，保留旧的约束（或者你可以决定是否要衰减）
                # 这里我们假设保留旧约束

                if new_update is not None:
                    # Client 参与了当前任务，更新状态
                    curr_fisher = new_update["fisher"]
                    curr_params = new_update["params"]

                    # Fisher 累加: F_total = F_old + F_new
                    if old_state["fisher"] is None:
                        final_fisher = curr_fisher
                    else:
                        final_fisher = {}
                        # 确保 keys 对齐 (LoRA 结构不变)
                        for k in curr_fisher:
                            if k in old_state["fisher"]:
                                final_fisher[k] = old_state["fisher"][k] + curr_fisher[k]
                            else:
                                final_fisher[k] = curr_fisher[k]

                    # Params 更新: θ* 更新为当前任务的最优解
                    final_params = curr_params

                else:
                    # Client 未参与当前任务，保持原样
                    final_fisher = old_state["fisher"]
                    final_params = old_state["params"]

                # 4. 保存到磁盘
                if final_fisher is not None:
                    save_dict = {"fisher": final_fisher, "params": final_params}
                    torch.save(save_dict, os.path.join(baseline_save_dir, f"ewc_{cid}.pt"))

            logger.info("EWC states saved successfully.")



        elif method in ["replay", "gem"] and client_replay_buffers is not None:

            logger.info(f"Saving {method.upper()} buffers to {baseline_save_dir}...")
            # 遍历所有客户端
            for cid in range(fed_args.num_clients):
                # 1. 获取该 Client 在当前任务中的采样数据
                new_data = current_task_replay_cache.get(cid, None)
                # 2. 如果存在新数据，将其加入该 Client 的历史 Buffer 列表
                if new_data is not None:
                    client_replay_buffers[cid].append(new_data)
                # 3. 保存到磁盘
                # 注意：即使当前任务没被选中(new_data is None)，也需要把旧的 Buffer 保存下来传递给下一个任务
                buffers = client_replay_buffers[cid]
                if buffers:
                    try:
                        # 合并所有历史任务的 buffer (Task 1 + Task 2 + ... + Task T)
                        full_replay_ds = concatenate_datasets(buffers)
                        full_replay_ds.save_to_disk(os.path.join(baseline_save_dir, f"replay_{cid}"))
                    except Exception as e:
                        logger.error(f"Error saving buffer for client {cid}: {e}")
            logger.info(f"{method.upper()} buffers saved successfully.")

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

