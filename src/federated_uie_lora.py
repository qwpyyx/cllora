#!/usr/bin/env python
# coding=utf-8
"""Federated learning training loop for UIE LoRA models."""
import copy
import logging
import os
import random
import math
import json
import re
import datasets
import numpy as np
import torch
import gc
import transformers
import torch.distributed as dist
from itertools import combinations
from datasets import load_dataset
from collections import defaultdict, Counter
from accelerate.utils import wait_for_everyone
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
from pilora_utils import (
    load_pilora_ref,
    save_pilora_ref,
    extract_pilora_ref_from_model,
)
from gsm8k.gsm8k_metrics import compute_gsm8k_metrics, extract_gsm8k_final_answer
from baseline_compressors import BaselineCompressor, apply_residual_to_lora_state


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



def summarize_client_partitions(client_datasets, *, label_key="Dataset"):
    """
    Build lightweight client-partition statistics for checking whether the split is
    quantity-skew or label/category-skew. The result is JSON-serializable.
    """
    summary = []
    for cid, ds in enumerate(client_datasets):
        labels = []
        if hasattr(ds, "column_names") and label_key in ds.column_names:
            try:
                labels = [str(x) for x in ds[label_key]]
            except Exception:
                labels = []
        cnt = Counter(labels)
        total = int(len(ds))
        if total > 0 and len(cnt) > 0:
            probs = np.array(list(cnt.values()), dtype=np.float64) / float(total)
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
            dominant_label, dominant_count = cnt.most_common(1)[0]
            dominant_ratio = float(dominant_count / total)
        else:
            entropy = 0.0
            dominant_label = None
            dominant_ratio = 0.0

        summary.append({
            "cid": int(cid),
            "num_samples": total,
            "num_labels": int(len(cnt)),
            "label_counts": dict(sorted(cnt.items(), key=lambda x: (-x[1], x[0]))),
            "label_entropy": entropy,
            "dominant_label": dominant_label,
            "dominant_label_ratio": dominant_ratio,
        })
    return summary


def make_client_datasets(train_dataset, fed_args):
    """
    Construct FL client datasets.

    quantity: old behavior. Dirichlet controls client sample counts only after a
              global shuffle, so each client is approximately IID in label/category.
    label:    label/category-skew Dirichlet. For each label/category, split its
              examples across clients using Dirichlet(alpha). This is the setting
              needed for semantic non-IID Dolly experiments.
    """
    strategy = str(getattr(fed_args, "partition_strategy", "quantity") or "quantity").lower()
    label_key = str(getattr(fed_args, "partition_label_key", "Dataset") or "Dataset")

    if strategy in ("quantity", "quantity_skew", "size", "size_skew"):
        client_datasets = partition_dataset(
            train_dataset,
            fed_args.num_clients,
            fed_args.dirichlet_alpha,
            base_seed=fed_args.federated_seed,
        )
    elif strategy in ("label", "label_skew", "semantic", "category", "category_skew"):
        if not (hasattr(train_dataset, "column_names") and label_key in train_dataset.column_names):
            raise ValueError(
                f"partition_strategy={strategy} requires label_key='{label_key}', "
                f"but train_dataset columns are {getattr(train_dataset, 'column_names', None)}"
            )
        client_datasets = partition_dataset_by_label(
            train_dataset,
            fed_args.num_clients,
            fed_args.dirichlet_alpha,
            base_seed=fed_args.federated_seed,
            label_key=label_key,
        )
    else:
        raise ValueError(
            f"Unknown partition_strategy={strategy}. "
            "Use quantity or label."
        )

    return client_datasets


def build_model_and_tokenizer(model_args):
    """
    Isolated loader for T5 / LLaMA / Qwen in Federated Learning.
    - T5 path: keep old behavior.
    - LLaMA path: keep old behavior.
    - Qwen path: new decoder-only branch, but do NOT hard-code 1/2/0.
    """

    name_lower = model_args.model_name_or_path.lower()
    is_adapter = ("adapter" in name_lower) or ("peft" in name_lower)

    is_t5 = "t5" in name_lower
    is_llama = ("llama" in name_lower) or ("vicuna" in name_lower)
    is_qwen = "qwen" in name_lower

    if not (is_t5 or is_llama or is_qwen):
        raise ValueError(
            f"Unsupported model family for federated mode: {model_args.model_name_or_path}. "
            "Currently only T5 / LLaMA / Qwen are explicitly supported."
        )

    print(
        f"[Build Model] Loading: {model_args.model_name_or_path} | "
        f"is_t5={is_t5}, is_llama={is_llama}, is_qwen={is_qwen}, is_adapter={is_adapter}"
    )

    # --------- resolve base model path ----------
    if is_adapter:
        peft_cfg = PeftConfig.from_pretrained(model_args.model_name_or_path)
        base_model_path = peft_cfg.base_model_name_or_path
    else:
        base_model_path = model_args.model_name_or_path

    # --------- config ----------
    config = AutoConfig.from_pretrained(
        base_model_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # 若你的 transformers 较老、Qwen 报错，再打开这一行
        # trust_remote_code=True,
    )
    config.use_cache = False

    # --------- tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # 若你的 transformers 较老、Qwen 报错，再打开这一行
        # trust_remote_code=True,
    )

    # ===== T5: 完全保持旧逻辑 =====
    if is_t5:
        pass

    # ===== LLaMA / Vicuna =====
    elif is_llama:
        # Do NOT hard-code LLaMA-2 token ids (1/2/0).
        # Llama-3.x uses a different special-token space, and its EOS can be a list
        # in the model/generation config.  Keep model-native BOS/EOS and choose a
        # PAD token that is distinct from EOS whenever possible.
        eos_ids = getattr(config, "eos_token_id", None)
        if isinstance(eos_ids, (list, tuple, set)):
            eos_id_set = set(int(x) for x in eos_ids)
        elif eos_ids is not None:
            eos_id_set = {int(eos_ids)}
        else:
            eos_id_set = set()

        def _valid_existing_token(tok: str) -> bool:
            try:
                tid = tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                return False
            if tid is None:
                return False
            if tokenizer.unk_token_id is not None and tid == tokenizer.unk_token_id:
                return False
            return int(tid) >= 0 and int(tid) not in eos_id_set

        if tokenizer.pad_token is None or (tokenizer.pad_token_id is not None and int(tokenizer.pad_token_id) in eos_id_set):
            chosen_pad = None
            for cand in ("<|finetune_right_pad_id|>", "<|reserved_special_token_0|>", "<|pad|>"):
                if _valid_existing_token(cand):
                    chosen_pad = cand
                    break
            if chosen_pad is not None:
                tokenizer.pad_token = chosen_pad
            elif tokenizer.unk_token is not None and (tokenizer.unk_token_id is None or int(tokenizer.unk_token_id) not in eos_id_set):
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        tokenizer.padding_side = "left"

        # Preserve config.bos/eos when the model provides them, especially list EOS
        # for Llama-3.x.  Only fill missing values from the tokenizer.
        if getattr(config, "bos_token_id", None) is None and tokenizer.bos_token_id is not None:
            config.bos_token_id = tokenizer.bos_token_id
        if getattr(config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
            config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id

    # ===== Qwen: 新增逻辑，只对 Qwen 生效 =====
    elif is_qwen:
        # Qwen 不要硬写 1/2/0，直接使用它自己的 special tokens
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        tokenizer.padding_side = "left"

        if tokenizer.pad_token_id is not None:
            config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.bos_token_id is not None:
            config.bos_token_id = tokenizer.bos_token_id
        if tokenizer.eos_token_id is not None:
            config.eos_token_id = tokenizer.eos_token_id

    # --------- model load kwargs ----------
    model_load_kwargs = {
        "config": config,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    # decoder-only 模型（LLaMA / Qwen）用半精度与 FlashAttention
    if is_llama or is_qwen:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            model_load_kwargs["torch_dtype"] = torch.bfloat16
            print("[Build Model] Using bfloat16 for decoder-only model.")
        else:
            model_load_kwargs["torch_dtype"] = torch.float16
            print("[Build Model] Using float16 for decoder-only model.")

        if model_args.ues_flash_attention:
            try:
                import flash_attn  # noqa: F401
                config._attn_implementation = "flash_attention_2"
                print("[Build Model] >>> USING FLASH ATTENTION 2 <<<")
            except ImportError:
                print("[Build Model] Flash Attention 2 not found, using default attention.")

    # --------- model class ----------
    if is_t5:
        model_class = AutoModelForSeq2SeqLM
        lora_task_type = TaskType.SEQ_2_SEQ_LM
    else:
        model_class = AutoModelForCausalLM
        lora_task_type = TaskType.CAUSAL_LM

    model = model_class.from_pretrained(
        base_model_path,
        from_tf=bool(".ckpt" in base_model_path),
        **model_load_kwargs,
        # 若你的 transformers 较老、Qwen 报错，再打开这一行
        # trust_remote_code=True,
    )

    # Keep generation_config consistent with model/tokenizer native special tokens.
    # This is required for Llama-3.x, whose EOS ids are not the Llama-2 ids 1/2.
    if not is_t5 and hasattr(model, "generation_config"):
        if getattr(model.generation_config, "bos_token_id", None) is None:
            model.generation_config.bos_token_id = getattr(config, "bos_token_id", None) or tokenizer.bos_token_id
        if getattr(model.generation_config, "eos_token_id", None) is None:
            model.generation_config.eos_token_id = getattr(config, "eos_token_id", None) or tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("[Build Model] tokenizer.bos/eos/pad =", tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    print("[Build Model] config.bos/eos/pad =", getattr(config, "bos_token_id", None), getattr(config, "eos_token_id", None), getattr(config, "pad_token_id", None))
    if hasattr(model, "generation_config"):
        print("[Build Model] generation.bos/eos/pad =", getattr(model.generation_config, "bos_token_id", None), getattr(model.generation_config, "eos_token_id", None), getattr(model.generation_config, "pad_token_id", None))

    # gradient checkpointing support
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # --------- PEFT / LoRA ----------
    if is_adapter:
        print(f"[Build Model] Loading existing adapter: {model_args.model_name_or_path}")
        model = PeftModel.from_pretrained(
            model,
            model_args.model_name_or_path,
            torch_dtype=model_load_kwargs.get("torch_dtype", "auto")
        )
    else:
        print(f"[Build Model] Initializing new LoRA adapter (r={model_args.lora_dim})")

        # T5 保持旧逻辑；LLaMA/Qwen 都显式打在 q/v projection 上
        target_modules = ["q_proj", "v_proj"] if (is_llama or is_qwen) else None

        peft_config = LoraConfig(
            task_type=lora_task_type,
            inference_mode=False,
            r=model_args.lora_dim,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)

    model.resize_token_embeddings(len(tokenizer))

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    model.print_trainable_parameters()

    # print("tokenizer.name_or_path =", tokenizer.name_or_path)
    # print("tokenizer.bos/eos/pad =", tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    # print("tokenizer.padding_side =", tokenizer.padding_side)
    # print("config.is_encoder_decoder =", getattr(config, "is_encoder_decoder", None))

    return model, tokenizer


# def build_model_and_tokenizer(model_args):
#     """
#     Unified loader for T5 and Llama in Federated Learning.
#     - T5: Uses standard loading.
#     - Llama: Uses Flash Attention 2 + BF16 + Custom Tokenizer Settings.
#     """
#
#     # --------- 1) 判别模型族 ----------
#     name_lower = model_args.model_name_or_path.lower()
#     is_adapter = ("adapter" in name_lower) or ("peft" in name_lower)
#     # is_llama = ("llama" in name_lower) or ("vicuna" in name_lower)
#     if "t5" in name_lower:
#         is_llama = False
#     else:
#         is_llama = ("llama" in name_lower) or ("vicuna" in name_lower)
#     print(f"[Build Model] Loading: {model_args.model_name_or_path} | Is Llama: {is_llama} | Is Adapter: {is_adapter}")
#
#     # --------- 2) 准备 Config 和 Tokenizer ----------
#     if is_adapter:
#         peft_cfg = PeftConfig.from_pretrained(model_args.model_name_or_path)
#         base_model_path = peft_cfg.base_model_name_or_path
#     else:
#         base_model_path = model_args.model_name_or_path
#
#     config = AutoConfig.from_pretrained(
#         base_model_path,
#         cache_dir=model_args.cache_dir,
#         revision=model_args.model_revision,
#         use_auth_token=True if model_args.use_auth_token else None,
#     )
#
#     # [通用设置] 训练时关闭 cache 以节省显存
#     config.use_cache = False
#
#     if is_llama:
#         # Llama Tokenizer (新环境/旧环境都兼容)
#         tokenizer = AutoTokenizer.from_pretrained(
#             base_model_path,
#             cache_dir=model_args.cache_dir,
#             use_fast=model_args.use_fast_tokenizer,
#             revision=model_args.model_revision,
#             use_auth_token=True if model_args.use_auth_token else None,
#         )
#
#         # 1. 补全 pad_token (如果缺失)
#         # Llama 原生通常没有 pad_token，优先使用 unk_token (id=0)
#         if tokenizer.pad_token is None:
#             if tokenizer.unk_token_id is not None:
#                 tokenizer.pad_token_id = tokenizer.unk_token_id
#                 tokenizer.pad_token = tokenizer.unk_token
#             else:
#                 # 兜底策略
#                 tokenizer.pad_token_id = 0
#                 tokenizer.pad_token = "<unk>"
#
#         # 2. 强制修正 ID (避免 Pad=1 与 BOS=1 冲突)
#         # 这是解决训练不收敛和预测乱码的关键
#         tokenizer.bos_token_id = 1
#         tokenizer.eos_token_id = 2
#         tokenizer.pad_token_id = 0  # 必须是 0
#
#         # 3. 设置左填充 (Left Padding)
#         # Decoder-only 模型做生成任务时必须左填充，否则输出不对齐
#         tokenizer.padding_side = "left"
#
#         # 同步更新 Config，防止生成时报 Warning
#         config.bos_token_id = tokenizer.bos_token_id
#         config.eos_token_id = tokenizer.eos_token_id
#         config.pad_token_id = tokenizer.pad_token_id
#
#     else:
#         # [T5 路径] 标准加载，完全兼容旧环境
#         tokenizer = AutoTokenizer.from_pretrained(
#             base_model_path,
#             cache_dir=model_args.cache_dir,
#             use_fast=model_args.use_fast_tokenizer,
#             revision=model_args.model_revision,
#             use_auth_token=True if model_args.use_auth_token else None,
#         )
#         # T5 默认 pad_token_id=0, padding_side='right'，无需修改
#
#     # --------- 3) 准备模型加载参数 ----------
#     model_load_kwargs = {
#         "config": config,
#         "cache_dir": model_args.cache_dir,
#         "revision": model_args.model_revision,
#         "use_auth_token": True if model_args.use_auth_token else None,
#     }
#
#     # [Llama 专属优化]
#     if is_llama:
#         # 1. 精度选择: 优先 BF16
#         if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
#             model_load_kwargs["torch_dtype"] = torch.bfloat16
#             print("[Build Model] Using bfloat16 for Llama.")
#         else:
#             model_load_kwargs["torch_dtype"] = torch.float16
#             print("[Build Model] Using float16 for Llama.")
#
#         # 2. Flash Attention 2 加速 (如果安装了)
#         if model_args.ues_flash_attention:
#             try:
#                 import flash_attn
#                 config._attn_implementation = "flash_attention_2"
#                 print("[Build Model] >>> USING FLASH ATTENTION 2 <<<")
#             except ImportError:
#                 print("[Build Model] Flash Attention 2 not found, using default attention.")
#
#     # --------- 4) 加载模型 ----------
#     if is_llama:
#         model_class = AutoModelForCausalLM
#         lora_task_type = TaskType.CAUSAL_LM
#     else:
#         # T5 使用标准 Seq2Seq 类
#         model_class = AutoModelForSeq2SeqLM
#         lora_task_type = TaskType.SEQ_2_SEQ_LM
#
#     # 加载 Base Model
#     model = model_class.from_pretrained(
#         base_model_path,
#         from_tf=bool(".ckpt" in base_model_path),
#         **model_load_kwargs
#     )
#
#     # [梯度检查点支持]
#     # 开启 input_require_grads 以支持 gradient_checkpointing
#     if hasattr(model, "enable_input_require_grads"):
#         model.enable_input_require_grads()
#     else:
#         def make_inputs_require_grad(module, input, output):
#             output.requires_grad_(True)
#
#         model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
#
#     # --------- 5) 应用 PEFT / LoRA ----------
#     if is_adapter:
#         print(f"[Build Model] Loading existing adapter: {model_args.model_name_or_path}")
#         model = PeftModel.from_pretrained(
#             model,
#             model_args.model_name_or_path,
#             torch_dtype=model_load_kwargs.get("torch_dtype", "auto")
#         )
#     else:
#         print(f"[Build Model] Initializing new LoRA adapter (r={model_args.lora_dim})")
#         peft_config = LoraConfig(
#             task_type=lora_task_type,
#             inference_mode=False,
#             r=model_args.lora_dim,
#             lora_alpha=32,
#             lora_dropout=0.1,
#             # Llama 需要指定 target_modules，T5 通常不需要(默认q,v)
#             target_modules=["q_proj", "v_proj"] if is_llama else None
#         )
#         model = get_peft_model(model, peft_config)
#
#     # --------- 6) 后处理 ----------
#     # 调整 Embedding 大小以匹配 Tokenizer (防止 special tokens 越界)
#     model.resize_token_embeddings(len(tokenizer))
#
#     # 确保 LoRA 参数可训练 (双重保险)
#     for name, param in model.named_parameters():
#         if 'lora_' in name:
#             param.requires_grad = True
#
#     # 打印可训练参数
#     model.print_trainable_parameters()
#     return model, tokenizer

def compute_selection_overlap_stats(client_to_layers, all_candidate_layers):
    """Compute overlap statistics for per-client selected layer sets within a round."""
    client_ids = sorted(client_to_layers.keys())
    sets = [set(client_to_layers[cid]) for cid in client_ids]
    num_clients = len(sets)
    if num_clients == 0:
        return None

    selected_sizes = [len(s) for s in sets]
    stats = {
        "num_clients": num_clients,
        "candidate_layers": len(all_candidate_layers),
        "mean_selected_layers": float(np.mean(selected_sizes)),
        "min_selected_layers": int(min(selected_sizes)),
        "max_selected_layers": int(max(selected_sizes)),
    }

    # Pairwise Jaccard / overlap
    if num_clients >= 2:
        jaccards = []
        intersections = []
        unions = []
        for a, b in combinations(sets, 2):
            inter = len(a & b)
            union = len(a | b)
            j = inter / union if union > 0 else 1.0
            jaccards.append(j)
            intersections.append(inter)
            unions.append(union)
        stats.update({
            "pairwise_jaccard_mean": float(np.mean(jaccards)),
            "pairwise_jaccard_min": float(np.min(jaccards)),
            "pairwise_jaccard_max": float(np.max(jaccards)),
            "pairwise_intersection_mean": float(np.mean(intersections)),
            "pairwise_union_mean": float(np.mean(unions)),
        })
    else:
        stats.update({
            "pairwise_jaccard_mean": 1.0,
            "pairwise_jaccard_min": 1.0,
            "pairwise_jaccard_max": 1.0,
            "pairwise_intersection_mean": float(len(sets[0])),
            "pairwise_union_mean": float(len(sets[0])),
        })

    # Layer-level coverage
    layer_counts = defaultdict(int)
    for s in sets:
        for name in s:
            layer_counts[name] += 1

    coverage_ratios = [cnt / num_clients for cnt in layer_counts.values()]
    stats.update({
        "union_selected_layers": int(len(layer_counts)),
        "mean_layer_coverage_ratio": float(np.mean(coverage_ratios)) if coverage_ratios else 0.0,
        "fully_shared_layers": int(sum(cnt == num_clients for cnt in layer_counts.values())),
        "singleton_layers": int(sum(cnt == 1 for cnt in layer_counts.values())),
        "coverage_histogram": {str(k): int(v) for k, v in sorted(
            ((cnt, sum(1 for x in layer_counts.values() if x == cnt)) for cnt in set(layer_counts.values())),
            key=lambda x: int(x[0])
        )},
        "top_shared_layers": sorted(layer_counts.items(), key=lambda x: (-x[1], x[0]))[:10],
    })
    return stats



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
        # is_causal = False
        # if "llama" in getattr(model.config, "_name_or_path", "").lower() or getattr(model.config, "is_decoder", False):
        #     if not getattr(model.config, "is_encoder_decoder", False):
        #         is_causal = True
        is_causal = not getattr(model.config, "is_encoder_decoder", False)


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


def get_atomic_unit_id(name: str, atomic_mode: str) -> str:
    """
    Map one LoRA tensor name to an upload unit id.

    atomic_mode:
      - tensor: original behavior, each tensor is one unit
      - ab_pair: lora_A and lora_B of the same module are one atomic unit
      - qv_block: q_proj/v_proj LoRA tensors in the same transformer layer are one atomic unit
    """
    if atomic_mode == "tensor":
        return name

    if atomic_mode == "ab_pair":
        if ".lora_A." in name:
            return name.replace(".lora_A.", ".lora_PAIR.")
        if ".lora_B." in name:
            return name.replace(".lora_B.", ".lora_PAIR.")
        return name

    if atomic_mode == "qv_block":
        # Example:
        # base_model.model.model.layers.32.self_attn.q_proj.lora_A.default.weight
        # base_model.model.model.layers.32.self_attn.v_proj.lora_B.default.weight
        # -> base_model.model.model.layers.32.self_attn.QV_BLOCK
        if ".self_attn." in name and (".q_proj." in name or ".v_proj." in name):
            layer_prefix = name.split(".self_attn.")[0]
            return layer_prefix + ".self_attn.QV_BLOCK"
        # fallback: at least keep A/B atomicity
        if ".lora_A." in name:
            return name.replace(".lora_A.", ".lora_PAIR.")
        if ".lora_B." in name:
            return name.replace(".lora_B.", ".lora_PAIR.")
        return name

    raise ValueError(f"Unknown atomic_mode: {atomic_mode}")


def build_atomic_units(layer_names, atomic_mode: str):
    """
    Return:
      units: dict unit_id -> list[tensor_name]
    """
    units = defaultdict(list)
    for name in layer_names:
        uid = get_atomic_unit_id(name, atomic_mode)
        units[uid].append(name)
    return units


def select_units_random(layer_names, layer_costs, budget, seed, atomic_mode: str):
    """
    Randomly select atomic upload units until budget is exhausted.
    Return selected tensor names, total cost, selected unit ids.
    """
    units = build_atomic_units(layer_names, atomic_mode)
    unit_items = list(units.items())

    rng = random.Random(seed)
    rng.shuffle(unit_items)

    selected_layers = set()
    selected_units = set()
    current_cost = 0

    for uid, members in unit_items:
        unit_cost = sum(layer_costs.get(k, 0) for k in members)
        if current_cost + unit_cost <= budget:
            selected_units.add(uid)
            selected_layers.update(members)
            current_cost += unit_cost

    return selected_layers, current_cost, selected_units


def select_units_topk(
        delta_dict,
        layer_costs,
        budget,
        atomic_mode: str,
        *,
        allowed_units=None,
        coverage_counts=None,
        coverage_penalty_beta: float = 0.0,
):
    """
    Select atomic upload units by update norm.

    Score of one atomic unit = sqrt(sum(||delta_k||_2^2)) over its member tensors.
    Cost of one atomic unit = sum(packet cost of its member tensors).

    Optional system-level diversity controls:
      - allowed_units: if not None, only units in this set can be selected.
      - coverage_counts: per-round number of previous clients that have selected each unit.
      - coverage_penalty_beta: adjusted_score = raw_score / (1 + beta * coverage_count).
    """
    units = build_atomic_units(list(delta_dict.keys()), atomic_mode)

    allowed_units = set(allowed_units) if allowed_units is not None else None
    coverage_counts = coverage_counts if coverage_counts is not None else {}

    unit_scores = []
    for uid, members in units.items():
        if allowed_units is not None and uid not in allowed_units:
            continue

        sq_sum = 0.0
        unit_cost = 0
        for k in members:
            if k not in delta_dict:
                continue
            n = torch.norm(delta_dict[k].float()).item()
            sq_sum += n * n
            unit_cost += layer_costs.get(k, 0)

        raw_score = math.sqrt(sq_sum)
        cover_cnt = int(coverage_counts.get(uid, 0))
        adjusted_score = raw_score / (1.0 + float(coverage_penalty_beta) * cover_cnt)
        unit_scores.append((uid, adjusted_score, unit_cost, members, raw_score, cover_cnt))

    unit_scores.sort(key=lambda x: x[1], reverse=True)

    selected_layers = set()
    selected_units = set()
    current_cost = 0

    for uid, adjusted_score, unit_cost, members, raw_score, cover_cnt in unit_scores:
        if current_cost + unit_cost <= budget:
            selected_units.add(uid)
            selected_layers.update(members)
            current_cost += unit_cost

    return selected_layers, current_cost, selected_units


def _lowrank_fro_norm_sq(x_parts, y_parts, eps: float = 1e-12) -> float:
    """
    Compute || X Y ||_F^2 without materializing the large matrix XY.

    X = concat(x_parts, dim=1), shape [d_out, k]
    Y = concat(y_parts, dim=0), shape [k, d_in]

    ||XY||_F^2 = tr((X^T X)(Y Y^T)).
    """
    xs = [x.detach().float().cpu() for x in x_parts if x is not None]
    ys = [y.detach().float().cpu() for y in y_parts if y is not None]

    if len(xs) == 0 or len(ys) == 0:
        return 0.0

    X = torch.cat(xs, dim=1)
    Y = torch.cat(ys, dim=0)

    gx = X.transpose(0, 1).matmul(X)
    gy = Y.matmul(Y.transpose(0, 1))

    val = (gx * gy).sum().item()
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return max(float(val), 0.0)


def _get_ab_key_pair(members):
    """Return (A_key, B_key) from one ab_pair unit."""
    a_key, b_key = None, None
    for k in members:
        if ".lora_A." in k:
            a_key = k
        elif ".lora_B." in k:
            b_key = k
    return a_key, b_key


def _get_ab_key_pairs_from_unit_members(members):
    """
    Return all valid (A_key, B_key) pairs contained in an upload unit.

    For atomic_mode=ab_pair, the unit contains one pair.
    For atomic_mode=qv_block, the unit normally contains two pairs from the
    same attention layer: q_proj and v_proj.  We treat the block effective
    update as a direct sum of its pair-level effective updates, so its group
    inner product is the sum of pair-level Frobenius inner products.
    """
    by_pair = defaultdict(list)
    for k in members:
        by_pair[get_atomic_unit_id(k, "ab_pair")].append(k)

    pairs = []
    for pair_uid in sorted(by_pair.keys()):
        a_key, b_key = _get_ab_key_pair(by_pair[pair_uid])
        if a_key is not None and b_key is not None:
            pairs.append((a_key, b_key))
    return pairs


def _safe_sqrt_ratio(num_sq: float, den_sq: float, eps: float = 1e-12) -> float:
    return math.sqrt(max(num_sq, 0.0) / (max(den_sq, 0.0) + eps))




def _parse_float_list(value, default_values):
    """Parse comma-separated floats used by diagnostics."""
    if value is None:
        return list(default_values)
    if isinstance(value, (list, tuple)):
        vals = [float(x) for x in value]
    else:
        vals = []
        for x in str(value).split(','):
            x = x.strip()
            if x:
                vals.append(float(x))
    vals = [x for x in vals if math.isfinite(x) and x > 0]
    return vals if len(vals) > 0 else list(default_values)


def _rank_dict_desc(score_map):
    """Return 1-based descending ranks. Ties are broken deterministically by uid."""
    items = sorted(score_map.items(), key=lambda x: (-float(x[1]), x[0]))
    return {uid: idx + 1 for idx, (uid, _) in enumerate(items)}


def _spearman_from_score_maps(score_a, score_b):
    common = sorted(set(score_a.keys()) & set(score_b.keys()))
    n = len(common)
    if n < 2:
        return None
    rank_a = _rank_dict_desc({k: score_a[k] for k in common})
    rank_b = _rank_dict_desc({k: score_b[k] for k in common})
    xa = np.array([rank_a[k] for k in common], dtype=np.float64)
    xb = np.array([rank_b[k] for k in common], dtype=np.float64)
    xa = xa - xa.mean()
    xb = xb - xb.mean()
    den = float(np.sqrt((xa * xa).sum()) * np.sqrt((xb * xb).sum()))
    if den <= 0:
        return None
    return float((xa * xb).sum() / den)


def _select_unit_ids_by_score(unit_rows, budget, score_key):
    """Greedy budgeted unit selection by a chosen score key."""
    selected = set()
    current_cost = 0
    sorted_rows = sorted(
        unit_rows,
        key=lambda r: (-float(r.get(score_key, 0.0)), str(r.get("uid", "")))
    )
    for r in sorted_rows:
        cost = int(r.get("unit_cost", 0))
        if budget is None or budget <= 0 or current_cost + cost <= budget:
            selected.add(r["uid"])
            current_cost += cost
    return selected, current_cost


def _selection_overlap(a, b):
    a = set(a)
    b = set(b)
    inter = len(a & b)
    union = len(a | b)
    return {
        "intersection": int(inter),
        "union": int(union),
        "jaccard": float(inter / union) if union > 0 else 1.0,
        "overlap_to_a": float(inter / len(a)) if len(a) > 0 else 1.0,
        "overlap_to_b": float(inter / len(b)) if len(b) > 0 else 1.0,
    }


def _sum_sq_for_units(unit_rows, selected_units, score_key):
    selected_units = set(selected_units)
    return float(sum(float(r.get(score_key, 0.0)) ** 2 for r in unit_rows if r["uid"] in selected_units))


def _build_ab_pair_saliency_rows(global_state_cpu, upload_candidate_update, lora_keys, layer_costs):
    """
    Build per-A/B-pair diagnostic rows.

    factor_score = sqrt(||U_A||^2 + ||U_B||^2), matching current A/B-pair Top-K.
    effective_score = ||B U_A + U_B A + U_B U_A||_F, computed without materializing the full matrix.
    """
    rows = []
    units = build_atomic_units(lora_keys, "ab_pair")

    for uid, members in units.items():
        a_key, b_key = _get_ab_key_pair(members)
        if a_key is None or b_key is None:
            continue
        if a_key not in global_state_cpu or b_key not in global_state_cpu:
            continue
        if a_key not in upload_candidate_update or b_key not in upload_candidate_update:
            continue

        A = global_state_cpu[a_key]
        B = global_state_cpu[b_key]
        U_A = upload_candidate_update[a_key]
        U_B = upload_candidate_update[b_key]

        ua_norm = torch.norm(U_A.float()).item()
        ub_norm = torch.norm(U_B.float()).item()
        factor_sq = ua_norm * ua_norm + ub_norm * ub_norm

        # Phi_t(U) = B U_A + U_B A + U_B U_A = [B, U_B] [U_A; A + U_A]
        eff_sq = _lowrank_fro_norm_sq([B, U_B], [U_A, A + U_A])

        rows.append({
            "uid": uid,
            "a_key": a_key,
            "b_key": b_key,
            "unit_cost": int(sum(layer_costs.get(k, 0) for k in members)),
            "factor_score": float(math.sqrt(max(factor_sq, 0.0))),
            "effective_score": float(math.sqrt(max(eff_sq, 0.0))),
            "ua_norm": float(ua_norm),
            "ub_norm": float(ub_norm),
        })

    return rows




def select_ab_pair_effective_topk(
        global_state_cpu,
        upload_candidate_update,
        lora_keys,
        layer_costs,
        budget,
        *,
        allowed_units=None,
        coverage_counts=None,
        coverage_penalty_beta: float = 0.0,
):
    """
    Select A/B-pair upload units by effective-update norm.

    Score of one A/B pair:
        || B U_A + U_B A + U_B U_A ||_F
    computed by the same low-rank Frobenius routine used in diagnostics.

    This function is used for the actual training path when
    --upload_score_mode effective_norm and --upload_atomic_mode ab_pair.
    """
    rows = _build_ab_pair_saliency_rows(
        global_state_cpu=global_state_cpu,
        upload_candidate_update=upload_candidate_update,
        lora_keys=lora_keys,
        layer_costs=layer_costs,
    )

    allowed_units = set(allowed_units) if allowed_units is not None else None
    coverage_counts = coverage_counts if coverage_counts is not None else {}

    unit_scores = []
    for r in rows:
        uid = r["uid"]
        if allowed_units is not None and uid not in allowed_units:
            continue
        raw_score = float(r.get("effective_score", 0.0))
        cover_cnt = int(coverage_counts.get(uid, 0))
        adjusted_score = raw_score / (1.0 + float(coverage_penalty_beta) * cover_cnt)
        members = [r["a_key"], r["b_key"]]
        unit_scores.append((uid, adjusted_score, int(r["unit_cost"]), members, raw_score, cover_cnt))

    unit_scores.sort(key=lambda x: (-x[1], x[0]))

    selected_layers = set()
    selected_units = set()
    current_cost = 0

    for uid, adjusted_score, unit_cost, members, raw_score, cover_cnt in unit_scores:
        if budget is None or budget <= 0 or current_cost + unit_cost <= budget:
            selected_units.add(uid)
            selected_layers.update(members)
            current_cost += unit_cost

    return selected_layers, current_cost, selected_units


def _lowrank_fro_inner(x1_parts, y1_parts, x2_parts, y2_parts) -> float:
    """
    Compute <X1 Y1, X2 Y2>_F without materializing the large matrices.

    X1 = concat(x1_parts, dim=1), Y1 = concat(y1_parts, dim=0)
    X2 = concat(x2_parts, dim=1), Y2 = concat(y2_parts, dim=0)

    <X1Y1, X2Y2>_F = sum((X1^T X2) * (Y1 Y2^T)).
    """
    xs1 = [x.detach().float().cpu() for x in x1_parts if x is not None]
    ys1 = [y.detach().float().cpu() for y in y1_parts if y is not None]
    xs2 = [x.detach().float().cpu() for x in x2_parts if x is not None]
    ys2 = [y.detach().float().cpu() for y in y2_parts if y is not None]

    if len(xs1) == 0 or len(ys1) == 0 or len(xs2) == 0 or len(ys2) == 0:
        return 0.0

    X1 = torch.cat(xs1, dim=1)
    Y1 = torch.cat(ys1, dim=0)
    X2 = torch.cat(xs2, dim=1)
    Y2 = torch.cat(ys2, dim=0)

    gx = X1.transpose(0, 1).matmul(X2)
    gy = Y1.matmul(Y2.transpose(0, 1))
    val = (gx * gy).sum().item()
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return float(val)


class _SNMaxCostFlow:
    """Small max-cost flow solver for the client-module assignment graph."""

    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]

    def add_edge(self, v, to, cap, cost, meta=None):
        fwd = [to, cap, float(cost), len(self.g[to]), meta]
        rev = [v, 0, -float(cost), len(self.g[v]), None]
        self.g[v].append(fwd)
        self.g[to].append(rev)
        return len(self.g[v]) - 1

    def max_cost_flow(self, s, t, maxf):
        flow = 0
        cost = 0.0
        n = self.n
        while flow < maxf:
            dist_arr = [-float('inf')] * n
            inq = [False] * n
            pv = [-1] * n
            pe = [-1] * n
            dist_arr[s] = 0.0
            q = [s]
            inq[s] = True
            head = 0
            while head < len(q):
                v = q[head]
                head += 1
                inq[v] = False
                for ei, e in enumerate(self.g[v]):
                    if e[1] <= 0:
                        continue
                    to = e[0]
                    nd = dist_arr[v] + e[2]
                    if nd > dist_arr[to] + 1e-12:
                        dist_arr[to] = nd
                        pv[to] = v
                        pe[to] = ei
                        if not inq[to]:
                            q.append(to)
                            inq[to] = True
            if pv[t] == -1:
                break
            add = maxf - flow
            v = t
            while v != s:
                e = self.g[pv[v]][pe[v]]
                add = min(add, e[1])
                v = pv[v]
            v = t
            while v != s:
                e = self.g[pv[v]][pe[v]]
                e[1] -= add
                self.g[v][e[3]][1] += add
                cost += add * e[2]
                v = pv[v]
            flow += add
        return flow, cost



def _sn_parse_layer_index(uid: str):
    """Extract transformer layer index from a LoRA unit id if available."""
    m = re.search(r"\.layers\.(\d+)\.", str(uid))
    if m is None:
        m = re.search(r"layers\.(\d+)", str(uid))
    return int(m.group(1)) if m is not None else None


def _sn_depth_group_from_layer(layer_idx, max_layer_idx):
    """Map a layer index to lower/middle/upper depth group."""
    if layer_idx is None or max_layer_idx is None or max_layer_idx < 0:
        return "unknown"
    n_layers = int(max_layer_idx) + 1
    # Three approximately equal depth groups.
    if layer_idx < n_layers / 3.0:
        return "lower"
    if layer_idx < 2.0 * n_layers / 3.0:
        return "middle"
    return "upper"


def _sn_rank_normalize(values_by_uid, uids, eps: float = 1e-6):
    """
    Rank-normalize positive scalars to (0, 1].  This preserves order inside the
    comparison set but removes raw scale differences across depth/projection groups.
    """
    uids = list(uids)
    n = len(uids)
    if n == 0:
        return {}
    ordered = sorted(uids, key=lambda u: (float(values_by_uid.get(u, 0.0)), str(u)))
    out = {}
    if n == 1:
        out[ordered[0]] = 1.0
        return out
    for rank, uid in enumerate(ordered, start=1):
        out[uid] = max(float(rank) / float(n), float(eps))
    return out


def _sn_parse_depth_ratios(depth_group_ratios: str):
    try:
        vals = [float(x.strip()) for x in str(depth_group_ratios).split(',') if x.strip() != '']
    except Exception:
        vals = []
    if len(vals) != 3 or any(v < 0 for v in vals) or sum(vals) <= 0:
        vals = [1.0, 1.0, 2.0]
    return {"lower": vals[0], "middle": vals[1], "upper": vals[2]}


def _sn_group_slot_targets(total_slots: int, ratios_by_group, available_groups):
    """Integer slot targets for depth-balanced P1, with exact total sum."""
    total_slots = int(total_slots)
    groups = [g for g in ["lower", "middle", "upper", "unknown"] if g in available_groups]
    if total_slots <= 0 or len(groups) == 0:
        return {g: 0 for g in groups}
    weights = {g: float(ratios_by_group.get(g, 1.0)) for g in groups}
    if sum(weights.values()) <= 0:
        weights = {g: 1.0 for g in groups}
    raw = {g: total_slots * weights[g] / sum(weights.values()) for g in groups}
    base = {g: int(math.floor(raw[g])) for g in groups}
    remain = total_slots - sum(base.values())
    frac_order = sorted(groups, key=lambda g: (-(raw[g] - base[g]), g))
    for g in frac_order[:remain]:
        base[g] += 1
    return base

def _run_signal_noise_p1_p2_schedule(
        *,
        client_records,
        global_state_cpu,
        lora_keys,
        layer_costs,
        budget,
        atomic_mode: str = "ab_pair",
        gap_eta: float = 1.0,
        force_full_budget: bool = False,
        min_eps: float = 1e-12,
        p1_norm_mode: str = "raw",
        depth_group_ratios: str = "1,1,2",
):
    """
    Server-side signal-noise sparse upload scheduler for structured FedLoRA atoms.

    atomic_mode can be:
      - ab_pair: each LoRA A/B pair is one upload unit;
      - qv_block: q_proj and v_proj A/B pairs in the same attention layer are
        grouped as one upload unit.

    Steps:
      1) compute exact low-rank effective-update inner products for each upload unit;
      2) solve P1 by selecting positive diminishing-return marginal gains;
      3) solve gap-aware P2-L by max-cost flow under equal unit costs.

    This implementation uses exact low-rank statistics for the main-result runs.
    A later sketch ablation can replace the inner-product matrix by Rademacher sketches.
    """
    if budget is None or budget <= 0:
        raise ValueError("sn_p1p2 requires a positive per-client comm_budget.")

    K = len(client_records)
    if K <= 1:
        raise ValueError("sn_p1p2 requires at least two participating clients for leave-one-out signal estimation.")

    atomic_mode = str(atomic_mode or "ab_pair").lower()
    if atomic_mode not in ("ab_pair", "qv_block"):
        raise ValueError(f"sn_p1p2 supports atomic_mode=ab_pair or qv_block, got {atomic_mode}.")

    units = build_atomic_units(lora_keys, atomic_mode)
    unit_infos = []
    for uid in sorted(units.keys()):
        members = sorted(units[uid])
        pairs = _get_ab_key_pairs_from_unit_members(members)
        if len(pairs) == 0:
            continue
        valid = True
        for a_key, b_key in pairs:
            if a_key not in global_state_cpu or b_key not in global_state_cpu:
                valid = False
                break
            if any(a_key not in rec["upload_candidate_update"] or b_key not in rec["upload_candidate_update"] for rec in client_records):
                valid = False
                break
        if not valid:
            continue
        cost = int(sum(layer_costs.get(k, 0) for k in members))
        if cost <= 0 or cost > int(budget):
            continue
        unit_infos.append({"uid": uid, "members": members, "pairs": pairs, "cost": cost})

    if len(unit_infos) == 0:
        return {rec["cid"]: {"selected_layers": set(), "selected_units": set(), "selection_cost": 0} for rec in client_records}, {
            "num_units": 0,
            "reason": "no_valid_ab_pair_units",
        }

    costs = sorted(set(int(u["cost"]) for u in unit_infos))
    equal_cost = (len(costs) == 1)

    # For Qwen2.5 with GQA, q_proj and v_proj LoRA A/B pairs can have
    # different packet costs (e.g., 220 vs. 132).  In that case P1 is solved
    # by a cost-aware marginal-gain greedy rule under the total packet budget,
    # and P2 is solved by a feasible generalized-assignment greedy repair.
    # The equal-cost case still uses the exact max-cost-flow solver.
    if equal_cost:
        unit_cost = int(costs[0])
        client_capacity = int(budget) // unit_cost
        if client_capacity <= 0:
            raise ValueError(f"sn_p1p2 budget={budget} is smaller than one A/B-pair unit cost={unit_cost}.")
        total_slot_budget = client_capacity * K
        total_cost_budget = total_slot_budget * unit_cost
        p2_solver = "exact_max_cost_flow_equal_cost"
    else:
        unit_cost = None
        client_capacity = None
        total_slot_budget = None
        total_cost_budget = int(budget) * K
        p2_solver = "greedy_generalized_assignment_heterogeneous_cost"

    # Build exact effective-update Gram matrices for each unit.
    # For qv_block, the unit effective update is a direct-sum group of the
    # q_proj and v_proj pair-level effective updates; therefore the group
    # inner product is the sum of pair-level inner products.  This avoids
    # invalid cross terms between q/v matrices with different output shapes.
    unit_stats = {}
    for u in unit_infos:
        uid = u["uid"]
        group_parts = []
        q_vals = []
        for rec in client_records:
            per_pair_parts = []
            q_total = 0.0
            for a_key, b_key in u["pairs"]:
                A = global_state_cpu[a_key]
                B = global_state_cpu[b_key]
                U_A = rec["upload_candidate_update"][a_key]
                U_B = rec["upload_candidate_update"][b_key]
                # Phi(U) = B U_A + U_B A + U_B U_A = [B, U_B] [U_A; A + U_A]
                x_parts = [B, U_B]
                y_parts = [U_A, A + U_A]
                per_pair_parts.append((x_parts, y_parts))
                q_total += _lowrank_fro_norm_sq(x_parts, y_parts)
            group_parts.append(per_pair_parts)
            q_vals.append(q_total)

        gram = np.zeros((K, K), dtype=np.float64)
        # Per-pair dimension normalization factors (for GQA: q dominates v otherwise)
        pair_norms = []
        for pair_idx, (a_key, b_key) in enumerate(u["pairs"]):
            d_out = global_state_cpu[b_key].shape[0]  # B rows
            d_in = global_state_cpu[a_key].shape[1]   # A cols
            pair_norms.append(float(d_out * d_in))
        total_norm = sum(pair_norms)
        # Normalize by total dimension product so q/v contribute equally per-element
        for i in range(K):
            gram[i, i] = float(q_vals[i]) / total_norm
        for i in range(K):
            for j in range(i + 1, K):
                val = 0.0
                for pair_idx in range(len(u["pairs"])):
                    val += _lowrank_fro_inner(
                        group_parts[i][pair_idx][0], group_parts[i][pair_idx][1],
                        group_parts[j][pair_idx][0], group_parts[j][pair_idx][1],
                    )
                gram[i, j] = val / total_norm
                gram[j, i] = val / total_norm

        if K > 1:
            a_hat = float((gram.sum() - np.trace(gram)) / (K * (K - 1)))
        else:
            a_hat = 0.0
        q_hat = float(np.mean(np.diag(gram)))
        a_hat = max(a_hat, 0.0)
        b_hat = max(q_hat - a_hat, float(min_eps))
        unit_stats[uid] = {
            "gram": gram,
            "q": np.asarray(q_vals, dtype=np.float64),
            "a_hat": a_hat,
            "b_hat": b_hat,
            "q_hat": q_hat,
            "unit_info": u,
            "num_pairs": int(len(u.get("pairs", []))),
        }

    # P1: diminishing-return module quota allocation.
    # We support scale/depth normalization for P1 only.  P2 still uses the exact
    # leave-one-out effective-update alignment.  This addresses a practical issue
    # observed on Qwen2.5-14B: raw Frobenius signal can be strongly biased toward
    # upper layers even after q/v are grouped into equal-cost qv-blocks.
    p1_norm_mode = str(p1_norm_mode or "raw").lower()
    if p1_norm_mode == "none":
        p1_norm_mode = "raw"

    # Depth metadata for diagnostics and depth-aware modes.
    layer_indices = {uid: _sn_parse_layer_index(uid) for uid in unit_stats.keys()}
    valid_layers = [x for x in layer_indices.values() if x is not None]
    max_layer_idx = max(valid_layers) if valid_layers else None
    depth_groups = {uid: _sn_depth_group_from_layer(layer_indices.get(uid), max_layer_idx) for uid in unit_stats.keys()}

    raw_a = {uid: float(unit_stats[uid]["a_hat"]) for uid in unit_stats.keys()}
    raw_b = {uid: float(unit_stats[uid]["b_hat"]) for uid in unit_stats.keys()}
    p1_a = dict(raw_a)
    p1_b = dict(raw_b)

    if p1_norm_mode in ("rank", "global_rank"):
        p1_a = _sn_rank_normalize(raw_a, unit_stats.keys(), eps=float(min_eps))
        p1_b = _sn_rank_normalize(raw_b, unit_stats.keys(), eps=float(min_eps))
    elif p1_norm_mode in ("depth_rank", "layer_rank", "depth_balanced"):
        p1_a, p1_b = {}, {}
        for g in sorted(set(depth_groups.values())):
            group_uids = [uid for uid in unit_stats.keys() if depth_groups.get(uid) == g]
            p1_a.update(_sn_rank_normalize(raw_a, group_uids, eps=float(min_eps)))
            p1_b.update(_sn_rank_normalize(raw_b, group_uids, eps=float(min_eps)))
    elif p1_norm_mode in ("mean", "global_mean"):
        mean_a = float(np.mean(list(raw_a.values()))) if raw_a else 1.0
        mean_b = float(np.mean(list(raw_b.values()))) if raw_b else 1.0
        p1_a = {uid: raw_a[uid] / max(mean_a, float(min_eps)) for uid in raw_a}
        p1_b = {uid: raw_b[uid] / max(mean_b, float(min_eps)) for uid in raw_b}
    elif p1_norm_mode != "raw":
        raise ValueError(f"Unknown sn_p1_norm_mode={p1_norm_mode}. Use raw, rank, depth_rank, or depth_balanced.")

    for uid in unit_stats.keys():
        unit_stats[uid]["p1_a_hat"] = float(max(p1_a.get(uid, raw_a[uid]), 0.0))
        unit_stats[uid]["p1_b_hat"] = float(max(p1_b.get(uid, raw_b[uid]), float(min_eps)))
        unit_stats[uid]["layer_idx"] = layer_indices.get(uid)
        unit_stats[uid]["depth_group"] = depth_groups.get(uid, "unknown")

    def _build_marginals(uid_subset=None):
        uid_subset = sorted(unit_stats.keys()) if uid_subset is None else sorted(uid_subset)
        out = []
        for uid in uid_subset:
            a_hat = unit_stats[uid]["p1_a_hat"]
            b_hat = unit_stats[uid]["p1_b_hat"]
            unit_c = int(unit_stats[uid]["unit_info"]["cost"])
            for k in range(K):
                gain = ((2 * (K - k) - 1) * a_hat - b_hat) / (K * K)
                if force_full_budget or gain > 0.0:
                    density = float(gain) / max(float(unit_c), 1.0)
                    out.append((float(gain), density, uid, k, unit_c))
        return out

    quotas = {uid: 0 for uid in unit_stats.keys()}
    used_total_cost = 0

    if p1_norm_mode == "depth_balanced" and equal_cost:
        ratios_by_group = _sn_parse_depth_ratios(depth_group_ratios)
        group_targets = _sn_group_slot_targets(int(total_slot_budget), ratios_by_group, set(depth_groups.values()))
        all_chosen_keys = set()
        for g, group_slots in group_targets.items():
            if group_slots <= 0:
                continue
            group_uids = [uid for uid in unit_stats.keys() if depth_groups.get(uid) == g]
            group_marginals = _build_marginals(group_uids)
            group_marginals.sort(key=lambda x: (-x[0], x[2], x[3]))
            chosen = group_marginals[:int(group_slots)]
            for gain, density, uid, k, unit_c in chosen:
                quotas[uid] += 1
                used_total_cost += int(unit_c)
                all_chosen_keys.add((uid, k))
        # If some group lacks positive candidates and we still have remaining slots,
        # backfill globally.  With force_full_budget=True this exactly exhausts the
        # same total slot budget; otherwise it only backfills positive marginal gains.
        remaining_slots = int(total_slot_budget) - int(sum(quotas.values()))
        if remaining_slots > 0:
            global_marginals = [m for m in _build_marginals(None) if (m[2], m[3]) not in all_chosen_keys]
            global_marginals.sort(key=lambda x: (-x[0], x[2], x[3]))
            for gain, density, uid, k, unit_c in global_marginals[:remaining_slots]:
                quotas[uid] += 1
                used_total_cost += int(unit_c)
    else:
        marginals = _build_marginals(None)
        if equal_cost:
            marginals.sort(key=lambda x: (-x[0], x[2], x[3]))
            chosen = marginals[:int(total_slot_budget)]
            for gain, density, uid, k, unit_c in chosen:
                quotas[uid] += 1
                used_total_cost += int(unit_c)
        else:
            # Multiple-choice marginal knapsack approximation.  Each marginal copy
            # has the corresponding module cost.  This keeps P1 cost-aware without
            # introducing a large DP table into the training loop.
            marginals.sort(key=lambda x: (-x[1], -x[0], x[2], x[3]))
            for gain, density, uid, k, unit_c in marginals:
                if used_total_cost + int(unit_c) <= int(total_cost_budget):
                    quotas[uid] += 1
                    used_total_cost += int(unit_c)

    quotas = {uid: int(k) for uid, k in quotas.items() if int(k) > 0}
    required_flow = int(sum(quotas.values()))

    if required_flow == 0:
        empty = {rec["cid"]: {"selected_layers": set(), "selected_units": set(), "selection_cost": 0} for rec in client_records}
        diag = {
            "num_clients": int(K),
            "atomic_mode": str(atomic_mode),
            "num_units": int(len(unit_stats)),
            "equal_cost": bool(equal_cost),
            "unit_cost": int(unit_cost) if unit_cost is not None else None,
            "unit_costs": [int(x) for x in costs],
            "client_capacity_units": int(client_capacity) if client_capacity is not None else None,
            "total_slot_budget": int(total_slot_budget) if total_slot_budget is not None else None,
            "total_cost_budget": int(total_cost_budget),
            "scheduled_units_total": 0,
            "scheduled_cost_total": 0,
            "p2_solver": p2_solver,
            "reason": "no_positive_marginal_gain",
        }
        return empty, diag

    # P2-L score with leave-one-out shared direction and positive-interaction gap penalty.
    active_uids = sorted(quotas.keys())
    score = {}
    interaction_gap_pos = {}
    for uid in active_uids:
        g = unit_stats[uid]["gram"]
        q = unit_stats[uid]["q"]
        for i, rec in enumerate(client_records):
            loo_alignment = float((g[i, :].sum() - g[i, i]) / max(K - 1, 1))
            d_pos = float(sum(max(float(g[i, j]), 0.0) for j in range(K) if j != i))
            s_im = (2.0 / K) * loo_alignment - (1.0 / (K * K)) * float(q[i]) - (float(gap_eta) / (K * K)) * d_pos
            score[(i, uid)] = float(s_im)
            interaction_gap_pos[(i, uid)] = d_pos

    selected_by_client = {rec["cid"]: {"selected_layers": set(), "selected_units": set(), "selection_cost": 0} for rec in client_records}
    unit_selected_clients = defaultdict(list)
    flow_score = 0.0
    p2_unfilled_quotas = {}

    if equal_cost:
        # Max-cost flow: source -> clients -> modules -> sink.
        n_clients = K
        n_units = len(active_uids)
        src = 0
        client_offset = 1
        unit_offset = client_offset + n_clients
        sink = unit_offset + n_units
        mf = _SNMaxCostFlow(sink + 1)

        for i in range(n_clients):
            mf.add_edge(src, client_offset + i, int(client_capacity), 0.0)

        edge_lookup = {}
        for i in range(n_clients):
            for uidx, uid in enumerate(active_uids):
                eidx = mf.add_edge(client_offset + i, unit_offset + uidx, 1, score[(i, uid)], meta=(i, uid))
                edge_lookup[(i, uid)] = (client_offset + i, eidx)

        for uidx, uid in enumerate(active_uids):
            mf.add_edge(unit_offset + uidx, sink, quotas[uid], 0.0)

        flow, flow_score = mf.max_cost_flow(src, sink, required_flow)
        if flow != required_flow:
            raise RuntimeError(f"sn_p1p2 assignment infeasible: flow={flow}, required={required_flow}.")

        for (i, uid), (v, eidx) in edge_lookup.items():
            # forward edge cap becomes 0 if it was used once.
            if mf.g[v][eidx][1] == 0:
                rec = client_records[i]
                cid = rec["cid"]
                u = unit_stats[uid]["unit_info"]
                selected_by_client[cid]["selected_units"].add(uid)
                selected_by_client[cid]["selected_layers"].update(u["members"])
                selected_by_client[cid]["selection_cost"] += int(unit_cost)
                unit_selected_clients[uid].append(int(cid))
    else:
        # Greedy generalized assignment for heterogeneous A/B-pair costs.
        # It respects every client's packet budget and every module quota as far
        # as feasible.  If a quota cannot be filled because of heterogeneous
        # budgets, we relax it and record the unfilled quota in diagnostics.
        remaining_budget = [int(budget) for _ in range(K)]
        remaining_quota = {uid: int(quotas[uid]) for uid in active_uids}
        candidates = []
        for uid in active_uids:
            unit_c = int(unit_stats[uid]["unit_info"]["cost"])
            for i in range(K):
                val = float(score[(i, uid)])
                density = val / max(float(unit_c), 1.0)
                candidates.append((density, val, i, uid, unit_c))
        candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))

        for density, val, i, uid, unit_c in candidates:
            if remaining_quota.get(uid, 0) <= 0:
                continue
            if remaining_budget[i] < unit_c:
                continue
            rec = client_records[i]
            cid = rec["cid"]
            if uid in selected_by_client[cid]["selected_units"]:
                continue
            u = unit_stats[uid]["unit_info"]
            selected_by_client[cid]["selected_units"].add(uid)
            selected_by_client[cid]["selected_layers"].update(u["members"])
            selected_by_client[cid]["selection_cost"] += int(unit_c)
            unit_selected_clients[uid].append(int(cid))
            remaining_budget[i] -= int(unit_c)
            remaining_quota[uid] -= 1
            flow_score += float(val)

        p2_unfilled_quotas = {uid: int(v) for uid, v in remaining_quota.items() if int(v) > 0}
        # Use actual scheduled flow after feasible repair.
        required_flow = int(sum(len(v["selected_units"]) for v in selected_by_client.values()))

    quota_values = list(quotas.values())
    diag_units = []
    # Save compact top diagnostics only to avoid huge files.
    for uid in sorted(unit_stats.keys(), key=lambda x: (-quotas.get(x, 0), -unit_stats[x]["a_hat"], x))[:50]:
        st = unit_stats[uid]
        diag_units.append({
            "uid": uid,
            "quota": int(quotas.get(uid, 0)),
            "unit_cost": int(st["unit_info"].get("cost", 0)),
            "num_pairs": int(st.get("num_pairs", 1)),
            "layer_idx": None if st.get("layer_idx") is None else int(st.get("layer_idx")),
            "depth_group": str(st.get("depth_group", "unknown")),
            "a_hat": float(st["a_hat"]),
            "b_hat": float(st["b_hat"]),
            "q_hat": float(st["q_hat"]),
            "p1_a_hat": float(st.get("p1_a_hat", st["a_hat"])),
            "p1_b_hat": float(st.get("p1_b_hat", st["b_hat"])),
            "snr": float(st["a_hat"] / (st["b_hat"] + 1e-12)),
            "p1_snr": float(st.get("p1_a_hat", st["a_hat"]) / (st.get("p1_b_hat", st["b_hat"]) + 1e-12)),
            "selected_clients": unit_selected_clients.get(uid, []),
        })

    scheduled_cost_total = int(sum(int(v["selection_cost"]) for v in selected_by_client.values()))
    diag = {
        "num_clients": int(K),
        "atomic_mode": str(atomic_mode),
        "num_units": int(len(unit_stats)),
        "active_units": int(len(active_uids)),
        "equal_cost": bool(equal_cost),
        "unit_cost": int(unit_cost) if unit_cost is not None else None,
        "unit_costs": [int(x) for x in costs],
        "client_capacity_units": int(client_capacity) if client_capacity is not None else None,
        "per_client_budget": int(budget),
        "total_slot_budget": int(total_slot_budget) if total_slot_budget is not None else None,
        "total_cost_budget": int(total_cost_budget),
        "used_total_cost_p1": int(used_total_cost),
        "scheduled_units_total": int(required_flow),
        "scheduled_cost_total": scheduled_cost_total,
        "p2_solver": p2_solver,
        "p2_unfilled_quota_total": int(sum(p2_unfilled_quotas.values())) if p2_unfilled_quotas else 0,
        "p2_unfilled_quotas": {str(k): int(v) for k, v in p2_unfilled_quotas.items()},
        "flow_score": float(flow_score),
        "gap_eta": float(gap_eta),
        "force_full_budget": bool(force_full_budget),
        "p1_norm_mode": str(p1_norm_mode),
        "depth_group_ratios": str(depth_group_ratios),
        "quota_by_depth_group": {str(g): int(sum(int(quotas.get(uid, 0)) for uid in quotas if depth_groups.get(uid) == g)) for g in sorted(set(depth_groups.values()))},
        "mean_quota": float(np.mean(quota_values)) if quota_values else 0.0,
        "max_quota": int(max(quota_values)) if quota_values else 0,
        "num_quota_units": int(len(quota_values)),
        "client_selected_units": {str(cid): int(len(v["selected_units"])) for cid, v in selected_by_client.items()},
        "client_selection_cost": {str(cid): int(v["selection_cost"]) for cid, v in selected_by_client.items()},
        "top_units": diag_units,
    }
    return selected_by_client, diag


def _diagnose_ab_pair_saliency(
        *,
        cid,
        rnd,
        selected_units,
        selection_cost,
        budget,
        global_state_cpu,
        upload_candidate_update,
        lora_keys,
        layer_costs,
        reparam_scales,
        seed,
        save_top_units=False,
        top_n=20,
):
    """
    Diagnostic only: does not affect selection, residuals, or aggregation.

    Diagnosis 1: equivalent LoRA reparameterization changes factor-norm Top-K.
    Diagnosis 2: factor-norm ranking mismatches effective-update ranking.
    """
    rows = _build_ab_pair_saliency_rows(
        global_state_cpu=global_state_cpu,
        upload_candidate_update=upload_candidate_update,
        lora_keys=lora_keys,
        layer_costs=layer_costs,
    )

    if len(rows) == 0:
        return None

    factor_score_map = {r["uid"]: r["factor_score"] for r in rows}
    eff_score_map = {r["uid"]: r["effective_score"] for r in rows}

    factor_selected, factor_cost = _select_unit_ids_by_score(rows, budget, "factor_score")
    eff_selected, eff_cost = _select_unit_ids_by_score(rows, budget, "effective_score")

    # Use the actual selection from the training path as well, because diversity/random modes may differ.
    actual_selected = set(selected_units)

    eff_mass_factor = _sum_sq_for_units(rows, factor_selected, "effective_score")
    eff_mass_eff = _sum_sq_for_units(rows, eff_selected, "effective_score")
    eff_mass_actual = _sum_sq_for_units(rows, actual_selected, "effective_score")

    factor_mass_factor = _sum_sq_for_units(rows, factor_selected, "factor_score")
    factor_mass_eff = _sum_sq_for_units(rows, eff_selected, "factor_score")

    out = {
        "global_round": int(rnd + 1),
        "cid": int(cid),
        "budget": int(budget) if budget is not None else None,
        "actual_selection_cost": int(selection_cost),
        "num_ab_pairs": int(len(rows)),
        "num_actual_selected_units": int(len(actual_selected)),
        "num_factor_selected_units": int(len(factor_selected)),
        "num_effective_selected_units": int(len(eff_selected)),
        "factor_selection_cost": int(factor_cost),
        "effective_selection_cost": int(eff_cost),

        # Diagnosis 2: ranking mismatch between current factor score and effective update score.
        "spearman_factor_vs_effective": _spearman_from_score_maps(factor_score_map, eff_score_map),
        "factor_vs_effective_selection_overlap": _selection_overlap(factor_selected, eff_selected),
        "actual_vs_effective_selection_overlap": _selection_overlap(actual_selected, eff_selected),
        "effective_mass_factor_selected": float(eff_mass_factor),
        "effective_mass_effective_selected": float(eff_mass_eff),
        "effective_mass_actual_selected": float(eff_mass_actual),
        "effective_mass_ratio_factor_to_effective": float(eff_mass_factor / (eff_mass_eff + 1e-12)),
        "effective_mass_ratio_actual_to_effective": float(eff_mass_actual / (eff_mass_eff + 1e-12)),
        "factor_mass_factor_selected": float(factor_mass_factor),
        "factor_mass_effective_selected": float(factor_mass_eff),
    }

    # Mean absolute rank gap between factor and effective rankings.
    factor_rank = _rank_dict_desc(factor_score_map)
    eff_rank = _rank_dict_desc(eff_score_map)
    common = sorted(set(factor_rank) & set(eff_rank))
    if len(common) > 0:
        rank_gaps = [abs(factor_rank[u] - eff_rank[u]) for u in common]
        out["mean_abs_rank_gap_factor_effective"] = float(np.mean(rank_gaps))
        out["median_abs_rank_gap_factor_effective"] = float(np.median(rank_gaps))

    # Diagnosis 1a: same scalar gauge transformation for all modules.
    scale_records = []
    for c in reparam_scales:
        c = float(c)
        reparam_rows = []
        for r in rows:
            reparam_score = math.sqrt((c * r["ua_norm"]) ** 2 + (r["ub_norm"] / c) ** 2)
            rr = dict(r)
            rr["reparam_factor_score"] = float(reparam_score)
            reparam_rows.append(rr)

        reparam_score_map = {r["uid"]: r["reparam_factor_score"] for r in reparam_rows}
        reparam_selected, reparam_cost = _select_unit_ids_by_score(
            reparam_rows, budget, "reparam_factor_score"
        )
        eff_mass_reparam = _sum_sq_for_units(reparam_rows, reparam_selected, "effective_score")

        scale_records.append({
            "scheme": "global_scale",
            "scale": float(c),
            "num_selected_units": int(len(reparam_selected)),
            "selection_cost": int(reparam_cost),
            "spearman_factor_vs_reparam_factor": _spearman_from_score_maps(factor_score_map, reparam_score_map),
            "overlap_with_original_factor_selection": _selection_overlap(factor_selected, reparam_selected),
            "overlap_with_effective_selection": _selection_overlap(eff_selected, reparam_selected),
            "effective_mass_reparam_selected": float(eff_mass_reparam),
            "effective_mass_ratio_reparam_to_effective": float(eff_mass_reparam / (eff_mass_eff + 1e-12)),
        })

    # Diagnosis 1b: independent per-module gauge transformation.
    rng = random.Random(int(seed))
    scale_choices = [float(x) for x in reparam_scales if float(x) > 0]
    if len(scale_choices) > 0:
        per_unit_scales = {r["uid"]: scale_choices[rng.randrange(len(scale_choices))] for r in rows}
        reparam_rows = []
        max_eff_relerr = 0.0

        row_by_uid = {r["uid"]: r for r in rows}
        for r in rows:
            c = float(per_unit_scales[r["uid"]])
            reparam_score = math.sqrt((c * r["ua_norm"]) ** 2 + (r["ub_norm"] / c) ** 2)
            rr = dict(r)
            rr["reparam_factor_score"] = float(reparam_score)
            reparam_rows.append(rr)

            # Verify effective-update invariance under A' = cA, B' = B/c, U_A' = cU_A, U_B' = U_B/c.
            A = global_state_cpu[r["a_key"]]
            B = global_state_cpu[r["b_key"]]
            U_A = upload_candidate_update[r["a_key"]]
            U_B = upload_candidate_update[r["b_key"]]
            eff_reparam_sq = _lowrank_fro_norm_sq([B / c, U_B / c], [c * U_A, c * (A + U_A)])
            eff_reparam = math.sqrt(max(eff_reparam_sq, 0.0))
            denom = abs(float(r["effective_score"])) + 1e-12
            max_eff_relerr = max(max_eff_relerr, abs(eff_reparam - float(r["effective_score"])) / denom)

        reparam_score_map = {r["uid"]: r["reparam_factor_score"] for r in reparam_rows}
        reparam_selected, reparam_cost = _select_unit_ids_by_score(
            reparam_rows, budget, "reparam_factor_score"
        )
        eff_mass_reparam = _sum_sq_for_units(reparam_rows, reparam_selected, "effective_score")
        scale_records.append({
            "scheme": "per_unit_random_scale",
            "scale_choices": scale_choices,
            "num_selected_units": int(len(reparam_selected)),
            "selection_cost": int(reparam_cost),
            "spearman_factor_vs_reparam_factor": _spearman_from_score_maps(factor_score_map, reparam_score_map),
            "overlap_with_original_factor_selection": _selection_overlap(factor_selected, reparam_selected),
            "overlap_with_effective_selection": _selection_overlap(eff_selected, reparam_selected),
            "effective_mass_reparam_selected": float(eff_mass_reparam),
            "effective_mass_ratio_reparam_to_effective": float(eff_mass_reparam / (eff_mass_eff + 1e-12)),
            "max_effective_score_relative_error_after_reparam": float(max_eff_relerr),
        })

    out["reparameterization_diagnostics"] = scale_records

    if save_top_units:
        top_n = max(1, int(top_n))
        top_factor = sorted(rows, key=lambda r: (-r["factor_score"], r["uid"]))[:top_n]
        top_eff = sorted(rows, key=lambda r: (-r["effective_score"], r["uid"]))[:top_n]
        out["top_factor_units"] = [
            {"uid": r["uid"], "factor_score": r["factor_score"], "effective_score": r["effective_score"]}
            for r in top_factor
        ]
        out["top_effective_units"] = [
            {"uid": r["uid"], "factor_score": r["factor_score"], "effective_score": r["effective_score"]}
            for r in top_eff
        ]

    return out

def _compute_residual_pre_diagnostics(
    *,
    cid,
    rnd,
    atomic_mode,
    selected_layers,
    full_update,
    global_state_cpu,
    lora_keys,
):
    """
    Compute quantities available before global aggregation:
      - full effective update norm ||Phi_t(U)||^2
      - residual effective update norm ||Phi_t(R)||^2
      - missing effective update norm ||H||^2
      - split error norm ||S||^2

    Also keep per-module C/R factors for post-aggregation drift diagnostics.
    """
    units = build_atomic_units(lora_keys, "ab_pair")

    full_eff_sq = 0.0
    residual_eff_sq = 0.0
    missing_eff_sq = 0.0
    split_sq = 0.0

    modules = []

    for uid, members in units.items():
        a_key, b_key = _get_ab_key_pair(members)
        if a_key is None or b_key is None:
            continue
        if a_key not in full_update or b_key not in full_update:
            continue
        if a_key not in global_state_cpu or b_key not in global_state_cpu:
            continue

        A = global_state_cpu[a_key]
        B = global_state_cpu[b_key]

        U_A = full_update[a_key]
        U_B = full_update[b_key]

        zero_A = torch.zeros_like(U_A)
        zero_B = torch.zeros_like(U_B)

        # C: uploaded part; R: unuploaded residual part.
        C_A = U_A if a_key in selected_layers else zero_A
        C_B = U_B if b_key in selected_layers else zero_B
        R_A = zero_A if a_key in selected_layers else U_A
        R_B = zero_B if b_key in selected_layers else U_B

        # Phi_t(U) = [B, U_B] [U_A; A + U_A]
        full_eff_sq += _lowrank_fro_norm_sq(
            [B, U_B],
            [U_A, A + U_A],
        )

        # Phi_t(R) = [B, R_B] [R_A; A + R_A]
        residual_eff_sq += _lowrank_fro_norm_sq(
            [B, R_B],
            [R_A, A + R_A],
        )

        # H = Phi_t(U) - Phi_t(C)
        #   = (B + C_B) R_A + R_B (A + U_A)
        #   = [B + C_B, R_B] [R_A; A + U_A]
        missing_eff_sq += _lowrank_fro_norm_sq(
            [B + C_B, R_B],
            [R_A, A + U_A],
        )

        # S = C_B R_A + R_B C_A = [C_B, R_B] [R_A; C_A]
        split_sq += _lowrank_fro_norm_sq(
            [C_B, R_B],
            [R_A, C_A],
        )

        # Store for post-aggregation drift/error computation.
        modules.append({
            "a_key": a_key,
            "b_key": b_key,
            "C_A": C_A.detach().cpu(),
            "C_B": C_B.detach().cpu(),
            "R_A": R_A.detach().cpu(),
            "R_B": R_B.detach().cpu(),
        })

    return {
        "cid": int(cid),
        "global_round": int(rnd + 1),
        "atomic_mode": atomic_mode,
        "num_modules": int(len(modules)),
        "full_eff_sq": float(full_eff_sq),
        "residual_eff_sq": float(residual_eff_sq),
        "missing_eff_sq": float(missing_eff_sq),
        "split_sq": float(split_sq),
        "modules": modules,
    }


def _finish_residual_post_diagnostics(pre_record, global_update_cpu):
    """
    Compute drift and total compensation error after global aggregation.

    global_update_cpu[k] = A^{t+1} - A^t or B^{t+1} - B^t.
    """
    drift_sq = 0.0
    comp_sq = 0.0

    for m in pre_record["modules"]:
        a_key = m["a_key"]
        b_key = m["b_key"]

        if a_key not in global_update_cpu or b_key not in global_update_cpu:
            continue

        dA = global_update_cpu[a_key]
        dB = global_update_cpu[b_key]

        C_A = m["C_A"]
        C_B = m["C_B"]
        R_A = m["R_A"]
        R_B = m["R_B"]

        # D = dB R_A + R_B dA = [dB, R_B] [R_A; dA]
        drift_sq += _lowrank_fro_norm_sq(
            [dB, R_B],
            [R_A, dA],
        )

        # E = D - S = (dB - C_B) R_A + R_B (dA - C_A)
        #     = [dB - C_B, R_B] [R_A; dA - C_A]
        comp_sq += _lowrank_fro_norm_sq(
            [dB - C_B, R_B],
            [R_A, dA - C_A],
        )

    eps = 1e-12
    split_ratio_full = _safe_sqrt_ratio(pre_record["split_sq"], pre_record["full_eff_sq"], eps)
    split_ratio_missing = _safe_sqrt_ratio(pre_record["split_sq"], pre_record["missing_eff_sq"], eps)
    drift_ratio_residual = _safe_sqrt_ratio(drift_sq, pre_record["residual_eff_sq"], eps)
    comp_error_ratio_missing = _safe_sqrt_ratio(comp_sq, pre_record["missing_eff_sq"], eps)

    out = {
        "cid": pre_record["cid"],
        "global_round": pre_record["global_round"],
        "atomic_mode": pre_record["atomic_mode"],
        "num_modules": pre_record["num_modules"],

        "full_eff_norm": math.sqrt(max(pre_record["full_eff_sq"], 0.0)),
        "residual_eff_norm": math.sqrt(max(pre_record["residual_eff_sq"], 0.0)),
        "missing_eff_norm": math.sqrt(max(pre_record["missing_eff_sq"], 0.0)),
        "split_norm": math.sqrt(max(pre_record["split_sq"], 0.0)),
        "drift_norm": math.sqrt(max(drift_sq, 0.0)),
        "comp_error_norm": math.sqrt(max(comp_sq, 0.0)),

        "split_ratio_to_full": float(split_ratio_full),
        "split_ratio_to_missing": float(split_ratio_missing),
        "drift_ratio_to_residual": float(drift_ratio_residual),
        "comp_error_ratio_to_missing": float(comp_error_ratio_missing),
    }

    # Do not serialize tensors.
    return out


def _make_state_delta_cpu(anchor_state_cpu, current_state_cpu, lora_keys):
    """
    Return current_state - anchor_state for LoRA tensors.
    This corresponds to A^s - A^t and B^s - B^t.
    """
    out = {}
    for k in lora_keys:
        if k in anchor_state_cpu and k in current_state_cpu:
            out[k] = (current_state_cpu[k] - anchor_state_cpu[k]).detach().cpu()
    return out


def _move_batch_to_device_for_loss(batch, device):
    out = {}
    for k, v in batch.items():
        if k == "input_ids_wo_label":
            continue
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def _estimate_loss_on_batches(model, dataloader, device, max_batches: int):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    count = 0

    for step, batch in enumerate(dataloader):
        if step >= max_batches:
            break

        inputs = _move_batch_to_device_for_loss(batch, device)

        try:
            outputs = model(**inputs)
            loss = getattr(outputs, "loss", None)
            if loss is None:
                if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                    loss = outputs[0]
                else:
                    continue

            loss_val = float(loss.detach().float().item())
            if math.isfinite(loss_val):
                total_loss += loss_val
                count += 1
        except Exception as e:
            # 诊断逻辑不要影响主训练
            logger.warning(f"[ReplayGain] loss estimation failed at batch {step}: {e}")
            continue

    if was_training:
        model.train()

    if count == 0:
        return None

    return total_loss / count


def _set_lora_state_inplace(model, state_cpu, lora_keys, device):
    name_to_param = dict(model.named_parameters())
    for k in lora_keys:
        if k in name_to_param and k in state_cpu:
            p = name_to_param[k]
            p.data.copy_(state_cpu[k].to(device=p.device, dtype=p.dtype))


def _apply_lora_residual_inplace(model, residual_cpu, scale: float):
    name_to_param = dict(model.named_parameters())
    for k, r in residual_cpu.items():
        if k in name_to_param:
            p = name_to_param[k]
            p.data.add_(r.to(device=p.device, dtype=p.dtype), alpha=scale)


def _diagnose_residual_replay_gain(
        *,
        model,
        global_state_cpu,
        lora_keys,
        client_dataset,
        collator,
        device,
        cid,
        rnd,
        client_residuals,
        client_residual_ages,
        batch_size,
        max_batches,
        min_age,
        max_age,
        scale,
        seed,
):
    """
    Bucket-level residual replay gain.

    For each residual age bucket:
      1. Reset LoRA to current global state.
      2. Compute baseline loss on a small local mini-batch.
      3. Add residuals of this age bucket to global LoRA.
      4. Compute replay loss.
      5. replay_gain = baseline_loss - replay_loss.

    Positive replay_gain means the historical residual still reduces current loss.
    """

    if client_residuals is None or len(client_residuals) == 0:
        return []

    # group residual tensors by age
    age_to_residual = defaultdict(dict)
    age_to_factor_norm_sq = defaultdict(float)

    for k, r in client_residuals.items():
        if k not in lora_keys:
            continue

        age = int(client_residual_ages.get(k, 0))
        if age < min_age:
            continue
        if max_age >= 0 and age > max_age:
            continue

        if r is None:
            continue

        r_cpu = r.detach().cpu()
        r_norm = torch.norm(r_cpu.float()).item()
        if r_norm <= 0:
            continue

        age_to_residual[age][k] = r_cpu
        age_to_factor_norm_sq[age] += r_norm * r_norm

    if len(age_to_residual) == 0:
        return []

    # build a small dataloader for replay-gain estimation
    g = torch.Generator()
    g.manual_seed(int(seed))

    dataloader = torch.utils.data.DataLoader(
        client_dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        generator=g,
        collate_fn=collator,
    )

    # baseline: current global LoRA
    _set_lora_state_inplace(model, global_state_cpu, lora_keys, device)
    baseline_loss = _estimate_loss_on_batches(
        model, dataloader, device, max_batches=max_batches
    )

    if baseline_loss is None:
        _set_lora_state_inplace(model, global_state_cpu, lora_keys, device)
        return []

    records = []

    for age in sorted(age_to_residual.keys()):
        residual_bucket = age_to_residual[age]

        # reset to global state before each replay
        _set_lora_state_inplace(model, global_state_cpu, lora_keys, device)

        # temporarily apply residual bucket
        _apply_lora_residual_inplace(model, residual_bucket, scale=scale)

        # rebuild dataloader to use the same shuffled order for fair comparison
        g = torch.Generator()
        g.manual_seed(int(seed))
        dataloader = torch.utils.data.DataLoader(
            client_dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=True,
            generator=g,
            collate_fn=collator,
        )

        replay_loss = _estimate_loss_on_batches(
            model, dataloader, device, max_batches=max_batches
        )

        if replay_loss is None:
            continue

        gain = baseline_loss - replay_loss
        factor_norm = math.sqrt(max(age_to_factor_norm_sq[age], 0.0))

        records.append({
            "global_round": int(rnd + 1),
            "cid": int(cid),
            "residual_age": int(age),
            "num_tensors": int(len(residual_bucket)),
            "baseline_loss": float(baseline_loss),
            "replay_loss": float(replay_loss),
            "replay_gain": float(gain),
            "positive_gain": bool(gain > 0),
            "gain_per_factor_norm": float(gain / (factor_norm + 1e-12)),
            "relative_gain_to_baseline": float(gain / (abs(baseline_loss) + 1e-12)),
            "residual_factor_norm": float(factor_norm),
            "replay_scale": float(scale),
            "max_batches": int(max_batches),
        })

    # important: restore global LoRA state after diagnosis
    _set_lora_state_inplace(model, global_state_cpu, lora_keys, device)

    return records




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

    from accelerate.utils import set_seed
    set_seed(fed_args.federated_seed, device_specific=False)  # 仅供 Trainer/Sampler 等内部用

    def _is_main():
        return getattr(training_args, "process_index", 0) == 0

    # Upload atomicity mode for lora_origin experiments.
    # tensor: original tensor-level upload
    # ab_pair: A/B pair atomic upload
    # qv_block: q/v adapter block atomic upload
    UPLOAD_ATOMIC_MODE = getattr(training_args, "upload_atomic_mode", "tensor").strip()

    if UPLOAD_ATOMIC_MODE not in ["tensor", "ab_pair", "qv_block"]:
        raise ValueError(
            f"Invalid upload_atomic_mode={UPLOAD_ATOMIC_MODE}. "
            "Choose from: tensor, ab_pair, qv_block."
        )

    if _is_main():
        logger.info(f"[UploadAtomicity] upload_atomic_mode={UPLOAD_ATOMIC_MODE}")

    UPLOAD_DIVERSITY_MODE = getattr(training_args, "upload_diversity_mode", "none").strip()
    if UPLOAD_DIVERSITY_MODE not in ["none", "group_mask", "coverage_penalty"]:
        raise ValueError(
            f"Invalid upload_diversity_mode={UPLOAD_DIVERSITY_MODE}. "
            "Choose from: none, group_mask, coverage_penalty."
        )

    DIVERSITY_NUM_GROUPS = max(1, int(getattr(training_args, "diversity_num_groups", 4)))
    COVERAGE_PENALTY_BETA = float(getattr(training_args, "coverage_penalty_beta", 1.0))

    if _is_main():
        logger.info(
            f"[UploadDiversity] mode={UPLOAD_DIVERSITY_MODE}, "
            f"num_groups={DIVERSITY_NUM_GROUPS}, "
            f"coverage_penalty_beta={COVERAGE_PENALTY_BETA}"
        )

    DIAGNOSE_RESIDUAL_ERRORS = bool(
        getattr(training_args, "diagnose_residual_errors", False)
    )

    if _is_main():
        logger.info(f"[ResidualDiagnostics] diagnose_residual_errors={DIAGNOSE_RESIDUAL_ERRORS}")


    DIAGNOSE_DRIFT_MAX_AGE = int(
        getattr(training_args, "diagnose_drift_max_age", 5)
    )

    if _is_main():
        logger.info(
            f"[ResidualDiagnostics] diagnose_drift_max_age={DIAGNOSE_DRIFT_MAX_AGE}"
        )

    LORA_RESIDUAL_ACCUMULATION = bool(
        getattr(training_args, "lora_residual_accumulation", False)
    )

    LORA_RESIDUAL_MAX_AGE = int(
        getattr(training_args, "lora_residual_max_age", -1)
    )

    if _is_main():
        logger.info(
            f"[LoRAResidual] accumulation={LORA_RESIDUAL_ACCUMULATION}, "
            f"max_age={LORA_RESIDUAL_MAX_AGE}"
        )

    DIAGNOSE_RESIDUAL_REPLAY_GAIN = bool(
        getattr(training_args, "diagnose_residual_replay_gain", False)
    )

    REPLAY_GAIN_MAX_CLIENTS_PER_ROUND = int(
        getattr(training_args, "replay_gain_max_clients_per_round", 2)
    )

    REPLAY_GAIN_MAX_BATCHES = int(
        getattr(training_args, "replay_gain_max_batches", 1)
    )

    REPLAY_GAIN_MIN_AGE = int(
        getattr(training_args, "replay_gain_min_age", 1)
    )

    REPLAY_GAIN_MAX_AGE = int(
        getattr(training_args, "replay_gain_max_age", 8)
    )

    REPLAY_GAIN_SCALE = float(
        getattr(training_args, "replay_gain_scale", 1.0)
    )

    if _is_main():
        logger.info(
            f"[ResidualReplayGain] enabled={DIAGNOSE_RESIDUAL_REPLAY_GAIN}, "
            f"max_clients_per_round={REPLAY_GAIN_MAX_CLIENTS_PER_ROUND}, "
            f"max_batches={REPLAY_GAIN_MAX_BATCHES}, "
            f"age_range=[{REPLAY_GAIN_MIN_AGE}, {REPLAY_GAIN_MAX_AGE}], "
            f"scale={REPLAY_GAIN_SCALE}"
        )

    UPLOAD_SCORE_MODE = str(
        getattr(training_args, "upload_score_mode", "factor_norm") or "factor_norm"
    ).lower()
    if UPLOAD_SCORE_MODE not in ("factor_norm", "effective_norm", "sn_p1p2"):
        raise ValueError(
            f"Unknown upload_score_mode={UPLOAD_SCORE_MODE}. "
            "Use factor_norm, effective_norm, or sn_p1p2."
        )

    if _is_main():
        logger.info(f"[UploadScore] mode={UPLOAD_SCORE_MODE}")

    SERVER_SN_UPLOAD = (UPLOAD_SCORE_MODE == "sn_p1p2")
    SN_GAP_ETA = float(getattr(training_args, "sn_gap_eta", 1.0))
    SN_FORCE_FULL_BUDGET = bool(getattr(training_args, "sn_force_full_budget", False))
    SN_MIN_SIGNAL_EPS = float(getattr(training_args, "sn_min_signal_eps", 1e-12))
    SN_SAVE_DIAGNOSTICS = bool(getattr(training_args, "sn_save_diagnostics", True))
    SN_P1_NORM_MODE = str(getattr(training_args, "sn_p1_norm_mode", "raw") or "raw")
    SN_DEPTH_GROUP_RATIOS = str(getattr(training_args, "sn_depth_group_ratios", "1,1,2") or "1,1,2")

    # Ours + encoding: run SN-P1P2 as a candidate scheduler, then encode
    # inside selected candidates under the original packet budget.
    SN_ENCODER_MODE = str(getattr(training_args, "sn_encoder_mode", "none") or "none").lower()
    SN_ENCODER_ALIASES = {
        "none": "none", "raw": "none",
        "compeft": "compeft", "topk_pq": "compeft", "pq": "compeft",
        "flasc": "flasc", "topk": "flasc",
    }
    SN_ENCODER_MODE = SN_ENCODER_ALIASES.get(SN_ENCODER_MODE, SN_ENCODER_MODE)
    if SN_ENCODER_MODE not in ("none", "compeft", "flasc"):
        raise ValueError("sn_encoder_mode must be one of: none/raw, compeft/topk_pq, flasc/topk")
    SN_CANDIDATE_BUDGET_MULTIPLIER = float(getattr(training_args, "sn_candidate_budget_multiplier", 1.0) or 1.0)
    SN_CANDIDATE_BUDGET_OVERRIDE = int(getattr(training_args, "sn_candidate_budget", 0) or 0)
    SN_ENCODER_PACKET_NUM = int(getattr(training_args, "sn_encoder_packet_num", 0) or 0)

    if SERVER_SN_UPLOAD and SN_ENCODER_MODE != "none":
        # effective_comm_budget is defined later after layer_costs/full_upload_cost are known.
        # At this configuration stage, use fed_args.comm_budget as the intended per-client
        # packet budget for the encoder. This fixes an early UnboundLocalError while keeping
        # the same budget semantics in the normal sparse-upload setting.
        _initial_comm_budget = getattr(fed_args, "comm_budget", None)
        if _initial_comm_budget is None or int(_initial_comm_budget) <= 0:
            raise ValueError("sn_encoder_mode requires a positive comm_budget.")
        if SN_ENCODER_PACKET_NUM <= 0:
            SN_ENCODER_PACKET_NUM = int(_initial_comm_budget)

    if SERVER_SN_UPLOAD:
        if UPLOAD_ATOMIC_MODE not in ("ab_pair", "qv_block"):
            raise ValueError("upload_score_mode=sn_p1p2 requires upload_atomic_mode=ab_pair or qv_block.")
        if LORA_RESIDUAL_ACCUMULATION:
            raise ValueError("upload_score_mode=sn_p1p2 currently does not support lora_residual_accumulation.")
        if UPLOAD_DIVERSITY_MODE != "none":
            raise ValueError("upload_score_mode=sn_p1p2 is a server-side scheduler; set upload_diversity_mode=none.")

    if _is_main():
        logger.info(
            f"[SignalNoiseUpload] enabled={SERVER_SN_UPLOAD}, "
            f"gap_eta={SN_GAP_ETA}, force_full_budget={SN_FORCE_FULL_BUDGET}, "
            f"min_eps={SN_MIN_SIGNAL_EPS}, save_diag={SN_SAVE_DIAGNOSTICS}, "
            f"encoder={SN_ENCODER_MODE}, candidate_budget_multiplier={SN_CANDIDATE_BUDGET_MULTIPLIER}, "
            f"candidate_budget_override={SN_CANDIDATE_BUDGET_OVERRIDE}, encoder_packet_num={SN_ENCODER_PACKET_NUM}"
        )

    # Pair-saliency diagnostics for A/B pair Top-K. This does not affect training.
    DIAGNOSE_PAIR_SALIENCY = bool(
        getattr(training_args, "diagnose_pair_saliency", False)
    )
    PAIR_SALIENCY_REPARAM_SCALES = _parse_float_list(
        getattr(training_args, "pair_saliency_reparam_scales", "0.25,0.5,2,4"),
        [0.25, 0.5, 2.0, 4.0],
    )
    PAIR_SALIENCY_SAVE_TOP_UNITS = bool(
        getattr(training_args, "pair_saliency_save_top_units", False)
    )
    PAIR_SALIENCY_TOP_N = int(
        getattr(training_args, "pair_saliency_top_n", 20)
    )

    if _is_main():
        logger.info(
            f"[PairSaliencyDiagnostics] enabled={DIAGNOSE_PAIR_SALIENCY}, "
            f"reparam_scales={PAIR_SALIENCY_REPARAM_SCALES}, "
            f"save_top_units={PAIR_SALIENCY_SAVE_TOP_UNITS}, "
            f"top_n={PAIR_SALIENCY_TOP_N}"
        )



    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        decoded_preds = skip_instructions(model, preds, tokenizer)
        references = [e["Instance"]["label"] for e in dataset]

        dataset_names = [str(x).lower() for x in dataset["Dataset"]]
        is_gsm8k = len(dataset_names) > 0 and all("gsm8k" in x for x in dataset_names)

        # ===== GSM8K 专用评测 =====
        if is_gsm8k:
            result = compute_gsm8k_metrics(decoded_preds, references)

            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}

            if save_prefix is not None:
                with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                    for example, pred, ref in zip(dataset, decoded_preds, references):
                        fout.write(json.dumps({
                            "Task": example["Task"],
                            "Dataset": example["Dataset"],
                            "Instance": example["Instance"],
                            "Prediction": pred,
                            "Prediction_final": extract_gsm8k_final_answer(pred),
                            "Reference_final": extract_gsm8k_final_answer(ref),
                        }, ensure_ascii=False) + "\n")

            return result

        # ===== 其他数据集仍走原逻辑 =====
        result = compute_metrics(predictions=decoded_preds, references=references)

        result_per_task = compute_grouped_metrics(
            predictions=decoded_preds,
            references=references,
            groups=dataset["Task"]
        )
        result.update(result_per_task)

        categories = dataset["Dataset"]
        result_per_category = compute_grouped_metrics(
            predictions=decoded_preds,
            references=references,
            groups=categories
        )
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


    # def compute_rouge_metrics(dataset, preds, save_prefix=None):
    #     # 对生成式模型的输出进行后处理
    #     print(type(preds), np.asarray(preds).dtype, np.asarray(preds).shape)
    #     decoded_preds = skip_instructions(model, preds, tokenizer)
    #     references = [e["Instance"]["label"] for e in dataset]
    #     result = compute_metrics(predictions=decoded_preds, references=references)
    #     # 按类别进行分类，考虑的是所有TC类的准确率
    #     result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references,
    #                                               groups=dataset["Task"])
    #     result.update(result_per_task)
    #     categories = dataset["Dataset"]
    #     result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references,
    #                                                   groups=categories)
    #     result.update(result_per_category)
    #     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    #     result["gen_len"] = np.mean(prediction_lens)
    #     result = {k: round(v, 4) for k, v in result.items()}
    #     if save_prefix is not None:
    #         with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
    #             for example, pred in zip(dataset, decoded_preds):
    #                 fout.write(json.dumps({
    #                     "Task": example["Task"],
    #                     "Dataset": example["Dataset"],
    #                     "Instance": example["Instance"],
    #                     "Prediction": pred
    #                 }) + "\n")
    #     return result

    def collator_for(model):
        # 确保数据整理器始终拿到未被分布式/Accelerate 包装的原始模型，
        # 以便访问 config 以及 prepare_decoder_input_ids 等方法。
        base_model = model
        if hasattr(base_model, "module"):
            base_model = base_model.module

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

    def get_lora_trainable_keys(model):
        return [name for name, param in model.named_parameters() if param.requires_grad and 'lora' in name]

    def get_param_bit_width(param):
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


    # --------- loading logging ---------
    logging.basicConfig(format="%(message)s", handlers=[logging.StreamHandler()])
    logger.info("Running federated learning mode")
    log_level = logging.INFO
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


    data_cache_dir = gen_cache_path(training_args.output_dir, data_args)

    print("data_args.data_dir =", data_args.data_dir)
    print("data_args.task_config_dir =", data_args.task_config_dir)
    print("data_args.instruction_file =", data_args.instruction_file)

    assert data_args.data_dir is not None and os.path.exists(data_args.data_dir), data_args.data_dir
    assert data_args.task_config_dir is not None and os.path.exists(
        data_args.task_config_dir), data_args.task_config_dir

    # instruction_file 对你这套脚本是可选的，不要强制 assert
    if data_args.instruction_file is not None:
        assert os.path.exists(data_args.instruction_file), data_args.instruction_file


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
            num_examples=data_args.num_examples,
            trust_remote_code=True,
        )
    raw_datasets.cleanup_cache_files()


    # --------- Detecting last checkpoint ---------
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


    # --------- Dataset ---------
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # unique sample
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


    # ------------ load client dataset --------------
    client_datasets = make_client_datasets(train_dataset, fed_args)

    if _is_main():
        partition_strategy = str(getattr(fed_args, "partition_strategy", "quantity"))
        partition_label_key = str(getattr(fed_args, "partition_label_key", "Dataset"))
        logger.info(
            f"[Partition] strategy={partition_strategy}, "
            f"label_key={partition_label_key}, "
            f"alpha={fed_args.dirichlet_alpha}, "
            f"num_clients={fed_args.num_clients}"
        )

        partition_summary = summarize_client_partitions(
            client_datasets,
            label_key=partition_label_key,
        )
        partition_summary_path = os.path.join(
            training_args.output_dir,
            "client_partition_summary.json"
        )
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(partition_summary_path, "w", encoding="utf-8") as fout:
            json.dump(partition_summary, fout, ensure_ascii=False, indent=2)
        logger.info(f"Saved client partition summary to {partition_summary_path}")


    # ------------ build model --------------
    model, tokenizer = build_model_and_tokenizer(model_args)

    # for n, p in model.named_parameters():
    #     if p.requires_grad and "lora" in n:
    #         print(n)


    # ------------ gradient checkpointing --------------
    # Qwen/Llama + LoRA + DDP may fail with the default re-entrant checkpointing:
    #   RuntimeError: Expected to mark a variable ready only once.
    # Use non-reentrant checkpointing when the Transformers version supports it.
    if training_args.gradient_checkpointing:
        logger.info("Gradient Checkpointing enabled.")

        # Keep Trainer-side arguments consistent as well. Some Transformers versions
        # read this field in Trainer._inner_training_loop.
        try:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
            logger.info("Set training_args.gradient_checkpointing_kwargs={'use_reentrant': False}.")
        except Exception as exc:
            logger.info(f"Could not set gradient_checkpointing_kwargs on training_args: {exc}")

        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                logger.info("Gradient Checkpointing enabled with use_reentrant=False.")
            except TypeError:
                # Older Transformers API: no kwargs support. Fall back to legacy mode.
                # If this branch is used and DDP still reports 'mark ready twice',
                # rerun with --gradient_checkpointing False.
                model.gradient_checkpointing_enable()
                logger.warning(
                    "Gradient Checkpointing enabled in legacy re-entrant mode because this "
                    "Transformers version does not accept gradient_checkpointing_kwargs. "
                    "If DDP fails with 'mark ready twice', disable gradient_checkpointing."
                )

        # Solve "does not have a grad_fn" error
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    else:
        logger.info("Gradient Checkpointing DISABLED (per arguments).")


    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    client_selection_tracker = {
        cid: {'count': 0, 'last_round': -1}
        for cid in range(fed_args.num_clients)
    }


    # -------- Begin Training ---------
    training_args.remove_unused_columns = False
    base_args = copy.deepcopy(training_args)
    base_args.do_train = True
    base_args.do_eval = False
    base_args.do_predict = False
    method = base_args.method
    BASELINE_METHOD_ALIASES = {
        "topk": "flasc",
        "flasc": "flasc",
        "topk_pq": "compeft",
        "compeft": "compeft",
        "fedcomp": "fedcomp",
        "block_opt": "flm_topk",
        "flm_topk": "flm_topk",
        "flm-topk": "flm_topk",
    }
    COMPRESSION_BASELINE_METHODS = set(BASELINE_METHOD_ALIASES.keys())
    baseline_method = BASELINE_METHOD_ALIASES.get(str(method).lower(), str(method).lower())
    if _is_main():
        logger.info("Use method: {}".format(method))
        if str(method).lower() in COMPRESSION_BASELINE_METHODS:
            logger.info(f"[MigratedBaseline] normalized_method={baseline_method}")
    global_model = model
    global_model.to("cpu")
    device = next(global_model.parameters()).device

    # 加载过去任务的fisher信息
    current_output_dir = training_args.output_dir
    # 解析当前任务序号（假设task_id已通过参数传入）
    current_task_id = data_args.task  # 例如：2

    # ================= PiLoRA: build fixed reference for this task =================
    pilora_ref_cpu = None
    if method == "pilora" and current_task_id > 1:
        # 1) Prefer explicit file from previous adapter dir: <adapter>/pilora_ref.pt
        pilora_ref_cpu = load_pilora_ref(model_args.model_name_or_path)

        # 2) Fallback: snapshot from the already-loaded adapter weights
        if pilora_ref_cpu is None:
            pilora_ref_cpu = extract_pilora_ref_from_model(global_model)
            if _is_main():
                logger.info(
                    "[PILoRA] pilora_ref.pt not found; using snapshot from loaded adapter weights as reference.")

        if _is_main():
            logger.info(f"[PILoRA] reference ready. #tensors={len(pilora_ref_cpu)}")
    # =====================================================================


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

    # ------------------ Lorm ---------------
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

    # ------------------ ewc ---------------
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

    # ------------------ replay-base method ---------------
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

    # ===== Group 3: Dense-Reduced-Participation baseline =====
    # 默认 False：正常跑 Group 1 / Group 2
    # 改成 True：自动把 clients_per_round 缩到与 sparse budget 匹配
    USE_DENSE_REDUCED_PARTICIPATION = False

    if USE_DENSE_REDUCED_PARTICIPATION and method != "lora_origin":
        raise ValueError("Dense-Reduced-Participation baseline is currently intended only for method='lora_origin'.")

    full_upload_cost = int(sum(layer_costs.values()))

    # 默认沿用原始值
    effective_clients_per_round = fed_args.clients_per_round
    effective_comm_budget = fed_args.comm_budget

    if USE_DENSE_REDUCED_PARTICIPATION:
        if fed_args.comm_budget is None or fed_args.comm_budget <= 0:
            raise ValueError("Dense-Reduced-Participation baseline needs a positive comm_budget, e.g. 1000.")

        # 匹配 Group 2 的系统级每轮总流量:
        #   Group 2: clients_per_round * comm_budget
        #   Group 3: effective_clients_per_round * full_upload_cost
        dense_clients_real = fed_args.clients_per_round * fed_args.comm_budget / full_upload_cost

        if dense_clients_real < 1:
            raise ValueError(
                f"Infeasible Dense-Reduced-Participation baseline: "
                f"one full-upload client costs {full_upload_cost}, "
                f"but sparse system budget per round is only "
                f"{fed_args.clients_per_round * fed_args.comm_budget}. "
                f"Please increase comm_budget or use a multi-round/window-matched dense baseline."
            )

        dense_clients = math.floor(dense_clients_real)
        effective_clients_per_round = min(fed_args.clients_per_round, dense_clients)

        # Group 3 必须 full upload，所以关闭按层裁剪
        effective_comm_budget = None

        if _is_main():
            logger.info(
                f"[Dense-Reduced-Participation] full_upload_cost={full_upload_cost}, "
                f"sparse_budget_per_client={fed_args.comm_budget}, "
                f"orig_clients_per_round={fed_args.clients_per_round}, "
                f"effective_clients_per_round={effective_clients_per_round}"
            )
    else:
        if _is_main():
            logger.info(
                f"[Normal FedLoRA] full_upload_cost={full_upload_cost}, "
                f"clients_per_round={effective_clients_per_round}, "
                f"comm_budget={effective_comm_budget}"
            )


    baseline_compressor = None
    client_compression_residuals = defaultdict(dict)
    baseline_compression_history = []
    if str(method).lower() in COMPRESSION_BASELINE_METHODS:
        baseline_packet_num = int(getattr(training_args, "baseline_packet_num", 0) or 0)
        if baseline_packet_num <= 0:
            if effective_comm_budget is None or effective_comm_budget <= 0:
                raise ValueError(
                    "Migrated compression baselines need a positive packet budget. "
                    "Set --comm_budget or --baseline_packet_num."
                )
            baseline_packet_num = int(effective_comm_budget)

        baseline_compressor = BaselineCompressor(
            packet_num=baseline_packet_num,
            packet_bytes=int(getattr(training_args, "packet_bytes", 1500)),
            blocks=int(getattr(training_args, "baseline_blocks", 1024)),
            bit=int(getattr(training_args, "baseline_bit", 18)),
            min_bit=int(getattr(training_args, "baseline_min_bit", 4)),
            topk_method=str(getattr(training_args, "baseline_topk_method", "gradient")),
            lora_rank=int(getattr(model_args, "lora_dim", 8)),
            flm_opt_max_iter=int(getattr(training_args, "baseline_flm_opt_max_iter", 40)),
            flm_max_blocks=int(getattr(training_args, "baseline_flm_max_blocks", 256)),
            flm_disable_optim=bool(getattr(training_args, "baseline_flm_disable_optim", False)),
        )
        if _is_main():
            logger.info(
                f"[MigratedBaseline] method={method}->{baseline_method}, "
                f"packet_num={baseline_packet_num}, "
                f"blocks={getattr(training_args, 'baseline_blocks', 1024)}, "
                f"bit={getattr(training_args, 'baseline_bit', 18)}"
            )


    sn_encoder_compressor = None
    sn_encoder_compression_history = []
    if SERVER_SN_UPLOAD and SN_ENCODER_MODE != "none":
        sn_encoder_compressor = BaselineCompressor(
            packet_num=int(SN_ENCODER_PACKET_NUM),
            packet_bytes=int(getattr(training_args, "packet_bytes", 1500)),
            blocks=int(getattr(training_args, "sn_encoder_blocks", 192)),
            bit=int(getattr(training_args, "sn_encoder_bit", 18)),
            min_bit=int(getattr(training_args, "sn_encoder_min_bit", 4)),
            topk_method=str(getattr(training_args, "baseline_topk_method", "gradient")),
            lora_rank=int(getattr(model_args, "lora_dim", 8)),
            flm_opt_max_iter=int(getattr(training_args, "baseline_flm_opt_max_iter", 40)),
            flm_max_blocks=int(getattr(training_args, "baseline_flm_max_blocks", 256)),
            flm_disable_optim=bool(getattr(training_args, "baseline_flm_disable_optim", False)),
        )
        if _is_main():
            logger.info(
                f"[SNEncoder] mode={SN_ENCODER_MODE}, packet_num={SN_ENCODER_PACKET_NUM}, "
                f"candidate_budget_multiplier={SN_CANDIDATE_BUDGET_MULTIPLIER}, "
                f"candidate_budget_override={SN_CANDIDATE_BUDGET_OVERRIDE}"
            )

    base_args.num_train_epochs = fed_args.local_epochs
    base_args.save_strategy = "no"
    base_args.logging_strategy = "no"
    base_args.evaluation_strategy = "no"

    if _is_main():
        logger.info("Initializing persistent DeepSpeed Trainer...")

    current_task_ewc_cache = {}
    current_task_replay_cache = {}
    adaptive_round_stats = defaultdict(list)
    selection_overlap_history = []
    residual_diag_history = []

    residual_multistep_history = []
    pending_residual_pre_records = []
    global_state_snapshots = {}

    client_lora_residuals = defaultdict(dict)
    client_lora_residual_ages = defaultdict(dict)
    residual_accumulation_history = []
    residual_replay_gain_history = []
    pair_saliency_history = []
    sn_schedule_history = []

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


    # ------------------ Training ------------------
    for rnd in range(fed_args.global_rounds):
        if _is_main():
            logger.info(f"Global round {rnd + 1}/{fed_args.global_rounds}")

        def pick_clients(num_clients, clients_per_round, round_id, base_seed):
            rng = np.random.RandomState(base_seed + 10007 * round_id)
            k = min(clients_per_round, num_clients)
            return rng.choice(np.arange(num_clients), size=k, replace=False).tolist()

        selected = pick_clients(
                        fed_args.num_clients,
                        effective_clients_per_round,
                        rnd,
                        fed_args.federated_seed,
        )

        if (
                DIAGNOSE_RESIDUAL_REPLAY_GAIN
                and LORA_RESIDUAL_ACCUMULATION
                and method == "lora_origin"
        ):
            replay_gain_clients = set(
                selected[:max(0, REPLAY_GAIN_MAX_CLIENTS_PER_ROUND)]
            )
        else:
            replay_gain_clients = set()


        if method == "lorm":
            target_matrix = "B" if (rnd % 2 == 0) else "A"
            freeze_target = "A" if target_matrix == "B" else "B"
            if _is_main():
                logger.info(f"Global Round {rnd + 1} [LoRM]: Training '{target_matrix}', Freezing '{freeze_target}'")
            lorm_client_updates = []

        if _is_main():
            logger.info(f"Selected client {selected}")

        # lora_keys = get_lora_trainable_keys(global_model)
        if method == "lorm":
            # 用 state_dict keys，确保包含 lora_A 与 lora_B（以及可能的 .default 后缀）
            lora_keys = [k for k in global_model.state_dict().keys() if "lora" in k]
        else:
            lora_keys = get_lora_trainable_keys(global_model)

        global_state_cpu = {
            k: v.detach().cpu()
            for k, v in global_model.state_dict().items()
            if k in lora_keys  # 只需要 LoRA 键
        }

        if DIAGNOSE_RESIDUAL_ERRORS and _is_main() and method == "lora_origin":
            # This is the anchor state (A^t, B^t) before local training and aggregation.
            global_state_snapshots[rnd + 1] = {
                k: v.detach().cpu().clone()
                for k, v in global_state_cpu.items()
            }

        # 2. 将其转换为 GPU 字典，用于快速加载
        global_state_gpu = {
            k: v.to(device) for k, v in global_state_cpu.items()
        }

        # ===== LoRM: 多卡同步（防止 rank0 聚合后其它 rank 仍是旧权重）=====
        if method == "lorm" and dist.is_available() and dist.is_initialized():
            # 注意：NCCL 要求 CUDA tensor；用 trainer.accelerator.device 最稳
            comm_device = trainer.accelerator.device if getattr(trainer, "accelerator", None) is not None else device

            # 确保广播用的 tensor 在正确 device 上
            for k in list(global_state_gpu.keys()):
                if global_state_gpu[k].device != comm_device:
                    global_state_gpu[k] = global_state_gpu[k].to(comm_device)

            # 从 rank0 广播到所有 rank
            for k in sorted(global_state_gpu.keys()):
                dist.broadcast(global_state_gpu[k], src=0)
            dist.barrier()

            # 写回 global_state_cpu（后面一些逻辑会继续用到）
            global_state_cpu = {k: v.detach().cpu() for k, v in global_state_gpu.items()}

        aggregated = {k: torch.zeros_like(global_state_cpu[k]) for k in lora_keys}
        total = 0
        round_selected_layers = {}
        # Atomic-unit coverage already selected by previous clients in this round.
        # Used only by upload_diversity_mode=coverage_penalty, but maintained for all modes.
        round_unit_coverage_counts = defaultdict(int)
        # Sorted unit list for deterministic group assignment in upload_diversity_mode=group_mask.
        round_atomic_unit_ids = sorted(build_atomic_units(lora_keys, UPLOAD_ATOMIC_MODE).keys())
        round_residual_pre_records = []
        round_server_sn_records = []

        for client_order, cid in enumerate(selected):
            if _is_main():
                logger.info(f"Client ID: {cid}")
                logger.info(f"Client {cid}: Resetting persistent trainer...")

            if (
                    baseline_method == "fedcomp"
                    and bool(getattr(training_args, "fedcomp_use_residual", True))
                    and len(client_compression_residuals.get(cid, {})) > 0
            ):
                # Old FedComp replays the previous unuploaded residual by initializing
                # the local client from global - residual before local training.
                client_init_state_cpu = apply_residual_to_lora_state(
                    global_state_cpu,
                    client_compression_residuals[cid],
                    lora_keys,
                )
                client_init_state_gpu = {k: v.to(device) for k, v in client_init_state_cpu.items()}
                trainer.model.load_state_dict(client_init_state_gpu, strict=False)
            else:
                trainer.model.load_state_dict(global_state_gpu, strict=False)

            trainer.train_dataset = client_datasets[cid]

            if method == "lorm":
                trainer.lorm_target_matrix = target_matrix
                trainer.lorm_grams = {}

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
                    logger.info(f"Client {cid}: Starting training (Method: {method}, Target: {target_matrix})...")

                # 2. 标准训练 (自动触发 Gram 计算)
                trainer.train(task_id=data_args.task, cid=cid)

                # 3. [后处理] 提取全量参数 & Grams
                model_to_save = _trainer_unwrap_model(trainer)

                # 只取本轮训练的目标矩阵 (A 或 B)
                # 类似于你 EWC 取 params
                client_params = {
                    k: v.detach().cpu().clone()
                    for k, v in model_to_save.state_dict().items()
                    if f"lora_{target_matrix}" in k and k.endswith("weight")
                }
                client_grams = getattr(trainer, "lorm_grams", {})

                # 4. [通信控制] 稀疏化筛选
                if effective_comm_budget is not None and effective_comm_budget  > 0:

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
                            candidate_keys, lorm_layer_costs, effective_comm_budget, seed
                        )
                        if _is_main():
                            logger.info(
                                f"Client {cid} [LoRM Random]: Selected {len(selected_layers)} layers, cost {selection_cost}/{effective_comm_budget}")

                    else:
                        # Top-K (Norm-based)
                        # 注意：select_layers_topk 通常接受 (delta, costs)
                        # 这里我们传 (client_params, costs)，因为它会算 value 的 norm
                        selected_layers, selection_cost = select_layers_topk(
                            client_params, lorm_layer_costs, effective_comm_budget
                        )
                        if _is_main():
                            logger.info(
                                f"Client {cid} [LoRM Top-K]: Selected {len(selected_layers)} layers, cost {selection_cost}/{effective_comm_budget}")

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

            if method in ["lora_origin", "ewc", "replay", "gem", "pilora"] or str(method).lower() in COMPRESSION_BASELINE_METHODS:
                if _is_main():
                    logger.info(f"Client {cid}: Starting training {method}...")

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

                if method == "pilora":
                    trainer.train(task_id=data_args.task, pilora_ref=pilora_ref_cpu)
                else:
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

                local_update = {
                    k: (-delta[k]).detach().cpu()
                    for k in delta
                }

                if (
                        LORA_RESIDUAL_ACCUMULATION
                        and method == "lora_origin"
                ):
                    upload_candidate_update = {}
                    for k in lora_keys:
                        prev_r = client_lora_residuals[cid].get(k, None)
                        if prev_r is None:
                            prev_r = torch.zeros_like(local_update[k])

                        prev_age = int(client_lora_residual_ages[cid].get(k, 0))

                        # Optional bounded-age support.
                        if LORA_RESIDUAL_MAX_AGE >= 0 and prev_age >= LORA_RESIDUAL_MAX_AGE:
                            prev_r = torch.zeros_like(local_update[k])
                            prev_age = 0

                        upload_candidate_update[k] = local_update[k] + prev_r
                else:
                    upload_candidate_update = local_update

                # The rest of the existing code expects delta = global - local.
                # For aggregation, selected upload should be -candidate_update.
                delta = {
                    k: (-upload_candidate_update[k]).detach().cpu()
                    for k in upload_candidate_update
                }

                baseline_compressed_this_client = False
                if str(method).lower() in COMPRESSION_BASELINE_METHODS:
                    if baseline_compressor is None:
                        raise RuntimeError("baseline_compressor is not initialized")
                    param_for_compress = {
                        k: name_to_param[k].detach().cpu()
                        for k in lora_keys
                        if k in name_to_param
                    }
                    compressed_delta, residual_delta, comp_stats = baseline_compressor.compress(
                        baseline_method,
                        delta,
                        param_for_compress,
                        lora_keys,
                    )
                    delta = compressed_delta
                    baseline_compressed_this_client = True

                    if baseline_method == "fedcomp" and residual_delta is not None:
                        client_compression_residuals[cid] = residual_delta

                    # For logs/overlap diagnostics, record tensors that still contain non-zero entries.
                    round_selected_layers[cid] = sorted([
                        k for k, v in delta.items()
                        if torch.count_nonzero(v).item() > 0
                    ])

                    if _is_main():
                        residual_nonzero = None
                        residual_norm = None
                        residual_tensor_count = None
                        if residual_delta is not None:
                            residual_nonzero = int(sum(torch.count_nonzero(v).item() for v in residual_delta.values()))
                            residual_norm = float(math.sqrt(sum(torch.norm(v.float()).item() ** 2 for v in residual_delta.values())))
                            residual_tensor_count = int(sum(1 for v in residual_delta.values() if torch.count_nonzero(v).item() > 0))

                        compression_entry = {
                            "global_round": int(rnd + 1),
                            "client_order": int(client_order),
                            "cid": int(cid),
                            "method_arg": str(method),
                            "normalized_method": str(baseline_method),
                            "packet_num": int(comp_stats.packet_num),
                            "packet_bytes": int(getattr(training_args, "packet_bytes", 1500)),
                            "comm_budget": int(effective_comm_budget) if effective_comm_budget is not None else None,
                            "full_upload_cost": int(full_upload_cost),
                            "nonzero": int(comp_stats.nonzero),
                            "total_numel": int(comp_stats.total_numel),
                            "density": float(comp_stats.nonzero / comp_stats.total_numel) if comp_stats.total_numel > 0 else 0.0,
                            "nonzero_tensor_count": int(len(round_selected_layers[cid])),
                            "total_lora_tensor_count": int(len(lora_keys)),
                            "selected_tensor_names": list(round_selected_layers[cid]),
                            "residual_nonzero": residual_nonzero,
                            "residual_norm": residual_norm,
                            "residual_tensor_count": residual_tensor_count,
                            "fedcomp_use_residual": bool(getattr(training_args, "fedcomp_use_residual", True)),
                            "baseline_blocks": int(getattr(training_args, "baseline_blocks", 1024)),
                            "baseline_bit": int(getattr(training_args, "baseline_bit", 18)),
                            "baseline_min_bit": int(getattr(training_args, "baseline_min_bit", 4)),
                            "baseline_topk_method": str(getattr(training_args, "baseline_topk_method", "gradient")),
                            "extra": dict(comp_stats.extra),
                        }
                        baseline_compression_history.append(compression_entry)

                        logger.info(
                            f"Client {cid} [MigratedBaseline/{baseline_method}]: "
                            f"nnz={comp_stats.nonzero}/{comp_stats.total_numel}, "
                            f"packet_num={comp_stats.packet_num}, extra={comp_stats.extra}"
                        )

                # ===== Server-side signal-noise scheduling path =====
                # For upload_score_mode=sn_p1p2, we first collect every selected client's
                # full A/B-pair candidate update, and only after all local training is done
                # let the server solve P1+gap-aware P2-L for the whole round.
                if (not baseline_compressed_this_client) and SERVER_SN_UPLOAD and method == "lora_origin":
                    round_server_sn_records.append({
                        "cid": int(cid),
                        "weight": int(len(client_datasets[cid])),
                        "delta": {k: delta[k].detach().cpu() for k in delta},
                        "upload_candidate_update": {
                            k: upload_candidate_update[k].detach().cpu()
                            for k in upload_candidate_update
                        },
                    })
                    try:
                        del name_to_param, trained_model
                    except Exception:
                        pass
                    continue

                # ===== Residual replay-gain diagnosis =====
                # This is only a diagnostic. It does not affect selection, residual update, or aggregation.
                if (
                        DIAGNOSE_RESIDUAL_REPLAY_GAIN
                        and _is_main()
                        and method == "lora_origin"
                        and LORA_RESIDUAL_ACCUMULATION
                        and cid in replay_gain_clients
                ):
                    try:
                        replay_records = _diagnose_residual_replay_gain(
                            model=trained_model,
                            global_state_cpu=global_state_cpu,
                            lora_keys=lora_keys,
                            client_dataset=client_datasets[cid],
                            collator=collator_for(trained_model),
                            device=device,
                            cid=cid,
                            rnd=rnd,
                            client_residuals=client_lora_residuals[cid],
                            client_residual_ages=client_lora_residual_ages[cid],
                            batch_size=training_args.per_device_eval_batch_size,
                            max_batches=REPLAY_GAIN_MAX_BATCHES,
                            min_age=REPLAY_GAIN_MIN_AGE,
                            max_age=REPLAY_GAIN_MAX_AGE,
                            scale=REPLAY_GAIN_SCALE,
                            seed=fed_args.federated_seed + 7919 * (rnd + 1) + cid,
                        )

                        residual_replay_gain_history.extend(replay_records)

                        if len(replay_records) > 0:
                            logger.info(
                                f"[ReplayGain] round={rnd + 1}, cid={cid}, "
                                f"records={len(replay_records)}"
                            )

                    except Exception as e:
                        logger.warning(
                            f"[ReplayGain] failed at round={rnd + 1}, cid={cid}: {e}"
                        )



                # 2. Apply Selection Strategy (Compressed Upload)
                if (not baseline_compressed_this_client) and effective_comm_budget is not None and effective_comm_budget > 0:
                    selected_layers = set()
                    selection_cost = 0

                    atomic_mode = UPLOAD_ATOMIC_MODE if method == "lora_origin" else "tensor"

                    if training_args.random_layer_selection:
                        seed = fed_args.federated_seed + rnd + cid

                        if atomic_mode == "tensor":
                            selected_layers, selection_cost = select_layers_random(
                                lora_keys, layer_costs, effective_comm_budget, seed
                            )
                            selected_units = set(selected_layers)
                        else:
                            selected_layers, selection_cost, selected_units = select_units_random(
                                lora_keys, layer_costs, effective_comm_budget, seed, atomic_mode
                            )

                        if _is_main():
                            logger.info(
                                f"Client {cid} [Random/{atomic_mode}]: "
                                f"Selected {len(selected_units)} units, "
                                f"{len(selected_layers)} tensors, "
                                f"cost {selection_cost}/{effective_comm_budget}"
                            )

                    else:
                        diversity_tag = "independent"
                        if atomic_mode == "tensor":
                            selected_layers, selection_cost = select_layers_topk(
                                delta, layer_costs, effective_comm_budget
                            )
                            selected_units = set(selected_layers)
                        else:
                            allowed_units = None
                            coverage_counts = None
                            coverage_beta = 0.0
                            diversity_tag = "independent"

                            if (
                                    method == "lora_origin"
                                    and atomic_mode != "tensor"
                                    and UPLOAD_DIVERSITY_MODE == "group_mask"
                            ):
                                group_id = client_order % DIVERSITY_NUM_GROUPS
                                allowed_units = {
                                    uid for idx, uid in enumerate(round_atomic_unit_ids)
                                    if idx % DIVERSITY_NUM_GROUPS == group_id
                                }
                                diversity_tag = f"group_mask:g{group_id}/{DIVERSITY_NUM_GROUPS}"

                            elif (
                                    method == "lora_origin"
                                    and atomic_mode != "tensor"
                                    and UPLOAD_DIVERSITY_MODE == "coverage_penalty"
                            ):
                                coverage_counts = round_unit_coverage_counts
                                coverage_beta = COVERAGE_PENALTY_BETA
                                diversity_tag = f"coverage_penalty:beta={COVERAGE_PENALTY_BETA}"

                            if UPLOAD_SCORE_MODE == "effective_norm":
                                if atomic_mode != "ab_pair":
                                    raise ValueError(
                                        "upload_score_mode=effective_norm is currently supported only "
                                        "with upload_atomic_mode=ab_pair."
                                    )
                                selected_layers, selection_cost, selected_units = select_ab_pair_effective_topk(
                                    global_state_cpu=global_state_cpu,
                                    upload_candidate_update=upload_candidate_update,
                                    lora_keys=lora_keys,
                                    layer_costs=layer_costs,
                                    budget=effective_comm_budget,
                                    allowed_units=allowed_units,
                                    coverage_counts=coverage_counts,
                                    coverage_penalty_beta=coverage_beta,
                                )
                            else:
                                selected_layers, selection_cost, selected_units = select_units_topk(
                                    delta,
                                    layer_costs,
                                    effective_comm_budget,
                                    atomic_mode,
                                    allowed_units=allowed_units,
                                    coverage_counts=coverage_counts,
                                    coverage_penalty_beta=coverage_beta,
                                )

                        if _is_main():
                            logger.info(
                                f"Client {cid} [Top-K/{atomic_mode}/{UPLOAD_SCORE_MODE}/{diversity_tag}]: "
                                f"Selected {len(selected_units)} units, "
                                f"{len(selected_layers)} tensors, "
                                f"cost {selection_cost}/{effective_comm_budget}"
                            )

                    # ===== A/B-pair saliency diagnostics =====
                    # Diagnostic only: keep the original selected_layers/selected_units unchanged.
                    if (
                            DIAGNOSE_PAIR_SALIENCY
                            and _is_main()
                            and method == "lora_origin"
                            and atomic_mode == "ab_pair"
                            and effective_comm_budget is not None
                            and effective_comm_budget > 0
                    ):
                        try:
                            pair_diag = _diagnose_ab_pair_saliency(
                                cid=cid,
                                rnd=rnd,
                                selected_units=set(selected_units),
                                selection_cost=selection_cost,
                                budget=effective_comm_budget,
                                global_state_cpu=global_state_cpu,
                                upload_candidate_update=upload_candidate_update,
                                lora_keys=lora_keys,
                                layer_costs=layer_costs,
                                reparam_scales=PAIR_SALIENCY_REPARAM_SCALES,
                                seed=fed_args.federated_seed + 104729 * (rnd + 1) + cid,
                                save_top_units=PAIR_SALIENCY_SAVE_TOP_UNITS,
                                top_n=PAIR_SALIENCY_TOP_N,
                            )
                            if pair_diag is not None:
                                pair_saliency_history.append(pair_diag)
                                logger.info(
                                    f"[PairSaliency][Round {rnd + 1}][Client {cid}] "
                                    f"rho={pair_diag.get('spearman_factor_vs_effective')}, "
                                    f"factor/eff_jaccard="
                                    f"{pair_diag['factor_vs_effective_selection_overlap']['jaccard']:.4f}, "
                                    f"mass_ratio="
                                    f"{pair_diag['effective_mass_ratio_factor_to_effective']:.4f}"
                                )
                        except Exception as e:
                            logger.warning(
                                f"[PairSaliency] failed at round={rnd + 1}, cid={cid}: {e}"
                            )

                    # if training_args.random_layer_selection:
                    #     # Random Selection
                    #     seed = fed_args.federated_seed + rnd + cid
                    #     selected_layers, selection_cost = select_layers_random(
                    #         lora_keys, layer_costs, effective_comm_budget, seed
                    #     )
                    #     if _is_main():
                    #         logger.info(
                    #             f"Client {cid} [Random]: Selected {len(selected_layers)} layers, cost {selection_cost}/{effective_comm_budget}")
                    # else:
                    #     # Top-K (Norm-based) Selection
                    #     selected_layers, selection_cost = select_layers_topk(
                    #         delta, layer_costs, effective_comm_budget
                    #     )
                    #     if _is_main():
                    #         logger.info(
                    #             f"Client {cid} [Top-K]: Selected {len(selected_layers)} layers, cost {selection_cost}/{effective_comm_budget}")

                    if (
                            DIAGNOSE_RESIDUAL_ERRORS
                            and _is_main()
                            and method == "lora_origin"
                    ):
                        # Code delta is global - local, while the actual local update is local - global.
                        full_update_for_diag = {
                            k: upload_candidate_update[k].detach().cpu()
                            for k in upload_candidate_update
                        }

                        pre_diag = _compute_residual_pre_diagnostics(
                            cid=cid,
                            rnd=rnd,
                            atomic_mode=atomic_mode,
                            selected_layers=set(selected_layers),
                            full_update=full_update_for_diag,
                            global_state_cpu=global_state_cpu,
                            lora_keys=lora_keys,
                        )

                        pre_diag["origin_round"] = rnd + 1

                        round_residual_pre_records.append(pre_diag)
                        pending_residual_pre_records.append(pre_diag)

                    # ===== Naive residual accumulation update =====
                    if (
                            LORA_RESIDUAL_ACCUMULATION
                            and method == "lora_origin"
                    ):
                        before_residual_sq = 0.0
                        after_residual_sq = 0.0
                        uploaded_sq = 0.0

                        new_residual = {}
                        new_age = {}

                        for k in lora_keys:
                            prev_r = client_lora_residuals[cid].get(k, None)
                            if prev_r is not None:
                                before_residual_sq += torch.norm(prev_r.float()).item() ** 2

                            if k in selected_layers:
                                # Candidate update is uploaded, so no residual remains for this tensor.
                                new_residual[k] = torch.zeros_like(upload_candidate_update[k])
                                new_age[k] = 0
                                uploaded_sq += torch.norm(upload_candidate_update[k].float()).item() ** 2
                            else:
                                # Candidate update is not uploaded, keep it as residual.
                                new_residual[k] = upload_candidate_update[k].detach().cpu()
                                prev_age = int(client_lora_residual_ages[cid].get(k, 0))
                                new_age[k] = prev_age + 1
                                after_residual_sq += torch.norm(new_residual[k].float()).item() ** 2

                        client_lora_residuals[cid] = new_residual
                        client_lora_residual_ages[cid] = new_age

                        if _is_main():
                            residual_accumulation_history.append({
                                "global_round": int(rnd + 1),
                                "cid": int(cid),
                                "atomic_mode": atomic_mode,
                                "comm_budget": int(
                                    effective_comm_budget) if effective_comm_budget is not None else None,
                                "before_residual_norm": float(math.sqrt(before_residual_sq)),
                                "after_residual_norm": float(math.sqrt(after_residual_sq)),
                                "uploaded_candidate_norm": float(math.sqrt(uploaded_sq)),
                                "max_residual_age": int(max(new_age.values())) if len(new_age) > 0 else 0,
                                "mean_residual_age": float(np.mean(list(new_age.values()))) if len(
                                    new_age) > 0 else 0.0,
                            })


                    # 3. Mask unselected layers (set delta to 0)
                    for k in delta:
                        if k not in selected_layers:
                            delta[k] = torch.zeros_like(delta[k])



                    if method == "lora_origin":
                        round_selected_layers[cid] = sorted(selected_layers)
                        for uid in selected_units:
                            round_unit_coverage_counts[uid] += 1

                elif (not baseline_compressed_this_client) and method == "lora_origin":
                    round_selected_layers[cid] = sorted(lora_keys)
                    # Dense upload: every atomic unit is selected by this client.
                    for uid in build_atomic_units(lora_keys, UPLOAD_ATOMIC_MODE).keys():
                        round_unit_coverage_counts[uid] += 1


                # ------------ Aggregate --------------
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

                    if _is_main():
                        logger.info(f"Client {cid} [Adaptive Task 1]: Computing Fisher (Distributed)...")

                    ADAPTIVE_SAMPLE_LIMIT = 500
                    fisher_ds = client_datasets[cid]
                    if len(fisher_ds) > ADAPTIVE_SAMPLE_LIMIT:
                        rng = np.random.RandomState(fed_args.federated_seed + cid + 1)
                        indices = rng.choice(len(fisher_ds), ADAPTIVE_SAMPLE_LIMIT, replace=False)
                        fisher_ds = fisher_ds.select(indices)

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


                    local_fisher_sum, local_count = compute_fisher_diag(trained_model, fisher_loader)
                    local_count_tensor = torch.tensor(local_count, device=trainer.accelerator.device)
                    total_count_tensor = trainer.accelerator.reduce(local_count_tensor, reduction="sum")
                    total_samples = total_count_tensor.item()

                    sorted_keys = sorted(local_fisher_sum.keys())
                    final_fisher = {}
                    for name in sorted_keys:
                        local_tensor = local_fisher_sum[name].to(trainer.accelerator.device)

                        global_sum_tensor = trainer.accelerator.reduce(local_tensor, reduction="sum")


                        if total_samples > 0:
                            avg_fisher = (global_sum_tensor / total_samples).cpu()
                        else:
                            avg_fisher = global_sum_tensor.cpu()

                        final_fisher[name] = avg_fisher

                    # 5. [仅主进程] 归一化 -> 更新状态 -> 存入 Cache
                    # 这里才需要缩进
                    if _is_main():

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

                    if _is_main() and getattr(trainer, "adaptive_task_stats", None) is not None:
                        stat = copy.deepcopy(trainer.adaptive_task_stats)
                        stat["global_round"] = int(rnd + 1)
                        adaptive_round_stats[cid].append(stat)

                    if _is_main():

                        client_state.update(F_client, theta_last)  # 此处 client_state 包含 T-1 历史，被更新为 T 的状态
                        per_task_cache[cid] = client_state  # 将包含 T 状态的 *完整对象* 存入 cache

                    try:
                        del delta, F_client, theta_last
                    except Exception:
                        pass


        if SERVER_SN_UPLOAD and len(round_server_sn_records) > 0:
            if SN_CANDIDATE_BUDGET_OVERRIDE > 0:
                sn_schedule_budget = int(SN_CANDIDATE_BUDGET_OVERRIDE)
            else:
                sn_schedule_budget = int(round(float(effective_comm_budget) * float(SN_CANDIDATE_BUDGET_MULTIPLIER)))
            sn_schedule_budget = max(1, sn_schedule_budget)

            selected_by_client, sn_diag = _run_signal_noise_p1_p2_schedule(
                client_records=round_server_sn_records,
                global_state_cpu=global_state_cpu,
                lora_keys=lora_keys,
                layer_costs=layer_costs,
                budget=sn_schedule_budget,
                atomic_mode=UPLOAD_ATOMIC_MODE,
                gap_eta=SN_GAP_ETA,
                force_full_budget=SN_FORCE_FULL_BUDGET,
                min_eps=SN_MIN_SIGNAL_EPS,
                p1_norm_mode=SN_P1_NORM_MODE,
                depth_group_ratios=SN_DEPTH_GROUP_RATIOS,
            )
            sn_diag["global_round"] = int(rnd + 1)
            sn_diag["actual_comm_budget"] = int(effective_comm_budget) if effective_comm_budget is not None else None
            sn_diag["candidate_schedule_budget"] = int(sn_schedule_budget)
            sn_diag["sn_encoder_mode"] = str(SN_ENCODER_MODE)
            sn_diag["sn_candidate_budget_multiplier"] = float(SN_CANDIDATE_BUDGET_MULTIPLIER)
            sn_diag["sn_candidate_budget_override"] = int(SN_CANDIDATE_BUDGET_OVERRIDE)
            if SN_SAVE_DIAGNOSTICS and _is_main():
                sn_schedule_history.append(sn_diag)
                logger.info(
                    f"[SignalNoiseUpload][Round {rnd + 1}] "
                    f"active_units={sn_diag.get('active_units', 0)}, "
                    f"scheduled={sn_diag.get('scheduled_units_total', 0)}/"
                    f"{sn_diag.get('total_slot_budget', 0)}, "
                    f"unit_cost={sn_diag.get('unit_cost', None)}, "
                    f"flow_score={sn_diag.get('flow_score', 0.0):.4e}"
                )

            for rec in round_server_sn_records:
                cid = rec["cid"]
                selected_layers = set(selected_by_client[cid]["selected_layers"])
                selected_units = set(selected_by_client[cid]["selected_units"])
                selection_cost = int(selected_by_client[cid]["selection_cost"])
                delta = rec["delta"]
                raw_candidate_layers = set(selected_layers)
                encoded = False
                encoder_stats = None

                if SN_ENCODER_MODE != "none":
                    if sn_encoder_compressor is None:
                        raise RuntimeError("sn_encoder_compressor is not initialized")
                    # Gate the compressor to SN-selected candidate tensors only.
                    # The compressor uses the actual packet budget, not the enlarged candidate budget.
                    encode_keys = [k for k in lora_keys if k in raw_candidate_layers]
                    compressed_delta, _, encoder_stats = sn_encoder_compressor.compress(
                        SN_ENCODER_MODE,
                        delta,
                        rec.get("params", None),
                        encode_keys,
                    )
                    new_delta = {k: torch.zeros_like(delta[k]) for k in lora_keys}
                    for k, v in compressed_delta.items():
                        new_delta[k] = v
                    delta = new_delta
                    encoded = True
                    # After encoding, selected_layers should reflect actually nonzero tensors.
                    selected_layers = set(k for k in lora_keys if torch.count_nonzero(delta[k]).item() > 0)
                else:
                    for k in delta:
                        if k not in selected_layers:
                            delta[k] = torch.zeros_like(delta[k])

                round_selected_layers[cid] = sorted(selected_layers)
                for uid in selected_units:
                    round_unit_coverage_counts[uid] += 1

                w = int(rec["weight"])
                for k in lora_keys:
                    aggregated[k] += delta[k] * w
                total += w

                if _is_main() and encoded and encoder_stats is not None:
                    nonzero_tensor_names = sorted([k for k in lora_keys if torch.count_nonzero(delta[k]).item() > 0])
                    candidate_tensor_count = int(len(raw_candidate_layers))
                    compression_entry = {
                        "global_round": int(rnd + 1),
                        "cid": int(cid),
                        "method_arg": "lora_origin",
                        "upload_score_mode": str(UPLOAD_SCORE_MODE),
                        "upload_atomic_mode": str(UPLOAD_ATOMIC_MODE),
                        "sn_encoder_mode": str(SN_ENCODER_MODE),
                        "normalized_encoder_method": str(encoder_stats.method),
                        "packet_num": int(encoder_stats.packet_num),
                        "comm_budget": int(effective_comm_budget) if effective_comm_budget is not None else None,
                        "candidate_schedule_budget": int(sn_schedule_budget),
                        "candidate_budget_multiplier": float(SN_CANDIDATE_BUDGET_MULTIPLIER),
                        "candidate_selection_cost": int(selection_cost),
                        "candidate_unit_count": int(len(selected_units)),
                        "candidate_tensor_count": candidate_tensor_count,
                        "nonzero": int(encoder_stats.nonzero),
                        "total_numel": int(encoder_stats.total_numel),
                        "density": float(encoder_stats.nonzero / encoder_stats.total_numel) if encoder_stats.total_numel else 0.0,
                        "nonzero_tensor_count": int(len(nonzero_tensor_names)),
                        "selected_tensor_names": nonzero_tensor_names,
                        "candidate_tensor_names": sorted(raw_candidate_layers),
                        "extra": dict(encoder_stats.extra),
                    }
                    sn_encoder_compression_history.append(compression_entry)

                if _is_main():
                    enc_msg = f", encoder={SN_ENCODER_MODE}, nnz={encoder_stats.nonzero if encoder_stats is not None else 'raw'}"
                    logger.info(
                        f"Client {cid} [SN-P1P2/{UPLOAD_ATOMIC_MODE}]: "
                        f"Candidate {len(selected_units)} units, "
                        f"{len(raw_candidate_layers)} tensors, candidate_cost {selection_cost}/{sn_schedule_budget}, "
                        f"actual_budget={effective_comm_budget}{enc_msg}"
                    )

        for cid in selected:
            client_selection_tracker[cid]['count'] += 1
            client_selection_tracker[cid]['last_round'] = rnd
            current_task_selected_clients.add(cid)

        if _is_main() and method == "lora_origin" and len(round_selected_layers) > 0:
            overlap_stats = compute_selection_overlap_stats(round_selected_layers, lora_keys)
            if overlap_stats is not None:
                overlap_stats["global_round"] = int(rnd + 1)
                overlap_stats["upload_diversity_mode"] = UPLOAD_DIVERSITY_MODE
                overlap_stats["diversity_num_groups"] = int(DIVERSITY_NUM_GROUPS)
                overlap_stats["coverage_penalty_beta"] = float(COVERAGE_PENALTY_BETA)
                overlap_stats["selected_client_ids"] = list(sorted(round_selected_layers.keys()))
                overlap_stats["per_client_num_layers"] = {
                    str(cid): len(layers) for cid, layers in round_selected_layers.items()
                }
                selection_overlap_history.append(overlap_stats)
                logger.info(
                    f"[SelectionOverlap][Round {rnd + 1}] "
                    f"clients={overlap_stats['num_clients']}, "
                    f"mean_selected={overlap_stats['mean_selected_layers']:.2f}, "
                    f"jaccard_mean={overlap_stats['pairwise_jaccard_mean']:.4f}, "
                    f"coverage_mean={overlap_stats['mean_layer_coverage_ratio']:.4f}, "
                    f"fully_shared={overlap_stats['fully_shared_layers']}, "
                    f"singleton_layers={overlap_stats['singleton_layers']}"
                )




        if method == "lorm" and len(lorm_client_updates) > 0:
            if _is_main():
                logger.info(f"[LoRM] Aggregating {target_matrix} Matrix...")
                new_state = lorm_aggregate(
                    lorm_client_updates,
                    global_model,
                    target_matrix=target_matrix,
                    device=device
                )
                # 只更新本轮目标矩阵（A 或 B）
                global_model.load_state_dict(new_state, strict=False)

            # 先 barrier，保证 rank0 已经完成聚合
            _trainer_wait_for_everyone(trainer)

            # 统一同步：广播“完整 LoRA(A+B)”到所有 rank，然后所有 rank 都 load
            lora_all_keys = [k for k in global_model.state_dict().keys() if "lora" in k]

            # 每个 rank 都先准备同 shape 的 buffer（值会被 rank0 覆盖）
            global_state_cpu = {k: global_model.state_dict()[k].detach().cpu() for k in lora_all_keys}

            if dist.is_available() and dist.is_initialized():
                comm_device = trainer.accelerator.device if getattr(trainer, "accelerator",
                                                                    None) is not None else device
                global_state_gpu_sync = {k: v.to(comm_device) for k, v in global_state_cpu.items()}

                for k in sorted(lora_all_keys):
                    dist.broadcast(global_state_gpu_sync[k], src=0)
                dist.barrier()

                global_state_cpu = {k: v.detach().cpu() for k, v in global_state_gpu_sync.items()}

            # 所有 rank 都把同步后的 LoRA 写回 global_model（避免下一轮用到旧权重）
            global_model.load_state_dict(global_state_cpu, strict=False)

            del lorm_client_updates
            torch.cuda.empty_cache()

        global_update_cpu = {}
        if method != "lorm":
            for k in lora_keys:
                mu = aggregated[k] / max(total, 1)
                global_update_cpu[k] = (-mu).detach().cpu()
                global_state_cpu[k] = global_state_cpu[k] + global_update_cpu[k]
                # global_state_cpu[k] = global_state_cpu[k] - mu

            # ===== Multi-round drift diagnostics =====
            if (
                    DIAGNOSE_RESIDUAL_ERRORS
                    and _is_main()
                    and method == "lora_origin"
                    and len(pending_residual_pre_records) > 0
            ):
                current_after_round = rnd + 1
                multistep_finished = []

                for rec in list(pending_residual_pre_records):
                    origin_round = int(rec.get("origin_round", rec["global_round"]))
                    residual_age = current_after_round - origin_round + 1

                    if residual_age <= 0:
                        continue

                    if residual_age > DIAGNOSE_DRIFT_MAX_AGE:
                        continue

                    if origin_round not in global_state_snapshots:
                        continue

                    anchor_state_cpu = global_state_snapshots[origin_round]

                    cumulative_update_cpu = _make_state_delta_cpu(
                        anchor_state_cpu,
                        global_state_cpu,
                        lora_keys,
                    )

                    out = _finish_residual_post_diagnostics(rec, cumulative_update_cpu)
                    out["origin_round"] = origin_round
                    out["eval_round"] = current_after_round
                    out["residual_age"] = residual_age

                    residual_multistep_history.append(out)
                    multistep_finished.append(out)

                # Log mean drift by age.
                if len(multistep_finished) > 0:
                    age_to_vals = defaultdict(list)
                    for x in multistep_finished:
                        age_to_vals[int(x["residual_age"])].append(x["drift_ratio_to_residual"])

                    msg_parts = []
                    for age in sorted(age_to_vals.keys()):
                        msg_parts.append(
                            f"age{age}={np.mean(age_to_vals[age]):.4f}"
                        )

                    logger.info(
                        f"[ResidualMultiStepDrift][After Round {rnd + 1}] "
                        + ", ".join(msg_parts)
                    )

                # Remove very old records to control memory.
                pending_residual_pre_records = [
                    rec for rec in pending_residual_pre_records
                    if current_after_round - int(rec.get("origin_round", rec["global_round"])) + 1
                       < DIAGNOSE_DRIFT_MAX_AGE
                ]

                # Remove old snapshots no longer needed.
                min_needed_round = current_after_round - DIAGNOSE_DRIFT_MAX_AGE + 2
                old_rounds = [r for r in global_state_snapshots.keys() if r < min_needed_round]
                for r in old_rounds:
                    del global_state_snapshots[r]


            # -----
            if (
                    DIAGNOSE_RESIDUAL_ERRORS
                    and _is_main()
                    and method == "lora_origin"
                    and len(round_residual_pre_records) > 0
            ):
                round_finished = []
                for rec in round_residual_pre_records:
                    out = _finish_residual_post_diagnostics(rec, global_update_cpu)
                    residual_diag_history.append(out)
                    round_finished.append(out)

                if len(round_finished) > 0:
                    logger.info(
                        f"[ResidualDiagnostics][Round {rnd + 1}] "
                        f"split/full={np.mean([x['split_ratio_to_full'] for x in round_finished]):.4f}, "
                        f"split/missing={np.mean([x['split_ratio_to_missing'] for x in round_finished]):.4f}, "
                        f"drift/residual={np.mean([x['drift_ratio_to_residual'] for x in round_finished]):.4f}, "
                        f"comp/missing={np.mean([x['comp_error_ratio_to_missing'] for x in round_finished]):.4f}"
                    )
            global_model.load_state_dict(global_state_cpu, strict=False)


            # global_model.load_state_dict(global_state_cpu, strict=False)


        wait_for_everyone()
        try:
            del aggregated, global_state_cpu, global_state_gpu
        except Exception:
            pass
        gc.collect()

    if method == "adaptive" and data_args.task > 1 and _is_main():
        adaptive_dir = os.path.join(current_output_dir, "adaptive_stats")
        os.makedirs(adaptive_dir, exist_ok=True)

        # 1) 先保存最原始的 round-level stats
        round_stats_path = os.path.join(
            adaptive_dir,
            f"task_{data_args.task}_client_round_stats.json"
        )
        with open(round_stats_path, "w", encoding="utf-8") as f:
            json.dump(adaptive_round_stats, f, indent=2, ensure_ascii=False)

        # 2) 再做一个按 client 聚合后的 task-level summary
        client_task_summary = {}

        for cid, stats_list in adaptive_round_stats.items():
            if len(stats_list) == 0:
                continue

            total_updates = sum(x.get("total_update_events", 0) for x in stats_list)
            total_risky = sum(x.get("risky_event_count", 0) for x in stats_list)
            total_capped = sum(x.get("capped_event_count", 0) for x in stats_list)

            eta_count = sum(x.get("eta_summary", {}).get("count", 0) or 0 for x in stats_list)
            eta_safe_count = sum(x.get("eta_safe_summary", {}).get("count", 0) or 0 for x in stats_list)

            eta_sum = sum(
                (x.get("eta_summary", {}).get("mean", 0.0) or 0.0) *
                (x.get("eta_summary", {}).get("count", 0) or 0)
                for x in stats_list
            )
            eta_safe_sum = sum(
                (x.get("eta_safe_summary", {}).get("mean", 0.0) or 0.0) *
                (x.get("eta_safe_summary", {}).get("count", 0) or 0)
                for x in stats_list
            )

            avg_risky_layer_ratio = float(np.mean([x.get("risky_layer_ratio", 0.0) for x in stats_list]))
            avg_capped_layer_ratio = float(np.mean([x.get("capped_layer_ratio", 0.0) for x in stats_list]))
            avg_selected_layer_ratio = float(np.mean([x.get("selected_layer_ratio", 0.0) for x in stats_list]))

            client_task_summary[str(cid)] = {
                "num_round_records": len(stats_list),
                "total_update_events": int(total_updates),
                "risky_event_count": int(total_risky),
                "risky_event_ratio": (total_risky / total_updates) if total_updates > 0 else 0.0,
                "capped_event_count": int(total_capped),
                "capped_event_ratio": (total_capped / total_updates) if total_updates > 0 else 0.0,
                "eta_mean": (eta_sum / eta_count) if eta_count > 0 else None,
                "eta_safe_mean": (eta_safe_sum / eta_safe_count) if eta_safe_count > 0 else None,
                "avg_risky_layer_ratio": avg_risky_layer_ratio,
                "avg_capped_layer_ratio": avg_capped_layer_ratio,
                "avg_selected_layer_ratio": avg_selected_layer_ratio,
            }

        task_summary_path = os.path.join(
            adaptive_dir,
            f"task_{data_args.task}_client_task_summary.json"
        )
        with open(task_summary_path, "w", encoding="utf-8") as f:
            json.dump(client_task_summary, f, indent=2, ensure_ascii=False)



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

    if _is_main() and len(sn_schedule_history) > 0:
        sn_path = os.path.join(training_args.output_dir, "signal_noise_schedule_history.json")
        with open(sn_path, "w", encoding="utf-8") as fout:
            json.dump(sn_schedule_history, fout, ensure_ascii=False, indent=2)
        logger.info(f"Saved signal-noise schedule history to {sn_path}")

    if _is_main() and len(selection_overlap_history) > 0:
        overlap_path = os.path.join(training_args.output_dir, "selection_overlap_history.json")
        with open(overlap_path, "w", encoding="utf-8") as fout:
            json.dump(selection_overlap_history, fout, ensure_ascii=False, indent=2)
        logger.info(f"Saved selection-overlap history to {overlap_path}")

    if _is_main() and len(residual_accumulation_history) > 0:
        residual_acc_path = os.path.join(
            training_args.output_dir,
            "residual_accumulation_history.json"
        )
        with open(residual_acc_path, "w", encoding="utf-8") as fout:
            json.dump(residual_accumulation_history, fout, ensure_ascii=False, indent=2)
        logger.info(f"Saved residual accumulation history to {residual_acc_path}")

    if _is_main() and len(residual_replay_gain_history) > 0:
        replay_gain_path = os.path.join(
            training_args.output_dir,
            "residual_replay_gain_history.json"
        )
        with open(replay_gain_path, "w", encoding="utf-8") as fout:
            json.dump(residual_replay_gain_history, fout, ensure_ascii=False, indent=2)
        logger.info(f"Saved residual replay-gain history to {replay_gain_path}")

        # Aggregate by residual age.
        age_groups = defaultdict(list)
        for r in residual_replay_gain_history:
            age_groups[int(r["residual_age"])].append(r)

        replay_gain_by_age = []
        for age in sorted(age_groups.keys()):
            rows = age_groups[age]
            gains = [float(x["replay_gain"]) for x in rows]
            gains_per_norm = [float(x["gain_per_factor_norm"]) for x in rows]
            rel_gains = [float(x["relative_gain_to_baseline"]) for x in rows]
            pos = [1.0 if bool(x["positive_gain"]) else 0.0 for x in rows]
            norms = [float(x["residual_factor_norm"]) for x in rows]

            replay_gain_by_age.append({
                "residual_age": int(age),
                "count": int(len(rows)),
                "mean_replay_gain": float(np.mean(gains)),
                "median_replay_gain": float(np.median(gains)),
                "positive_gain_ratio": float(np.mean(pos)),
                "mean_gain_per_factor_norm": float(np.mean(gains_per_norm)),
                "median_gain_per_factor_norm": float(np.median(gains_per_norm)),
                "mean_relative_gain_to_baseline": float(np.mean(rel_gains)),
                "mean_residual_factor_norm": float(np.mean(norms)),
            })

        replay_gain_by_age_path = os.path.join(
            training_args.output_dir,
            "residual_replay_gain_by_age.json"
        )
        with open(replay_gain_by_age_path, "w", encoding="utf-8") as fout:
            json.dump(replay_gain_by_age, fout, ensure_ascii=False, indent=2)
        logger.info(f"Saved residual replay-gain by age to {replay_gain_by_age_path}")


    if _is_main() and len(residual_diag_history) > 0:
        residual_diag_path = os.path.join(training_args.output_dir, "residual_diagnostics_history.json")
        with open(residual_diag_path, "w", encoding="utf-8") as fout:
            json.dump(residual_diag_history, fout, ensure_ascii=False, indent=2)
        logger.info(f"Saved residual diagnostics history to {residual_diag_path}")

    if _is_main() and len(residual_multistep_history) > 0:
        residual_multistep_path = os.path.join(
            training_args.output_dir,
            "residual_multistep_drift_history.json"
        )
        with open(residual_multistep_path, "w", encoding="utf-8") as fout:
            json.dump(residual_multistep_history, fout, ensure_ascii=False, indent=2)
        logger.info(
            f"Saved residual multi-step drift history to {residual_multistep_path}"
        )

    if _is_main() and len(pair_saliency_history) > 0:
        pair_diag_path = os.path.join(training_args.output_dir, "pair_saliency_diagnostics_history.json")
        with open(pair_diag_path, "w", encoding="utf-8") as fout:
            json.dump(pair_saliency_history, fout, ensure_ascii=False, indent=2)
        logger.info(f"Saved pair-saliency diagnostics history to {pair_diag_path}")

        pair_summary = {
            "count": int(len(pair_saliency_history)),
            "mean_spearman_factor_vs_effective": float(np.mean([
                x["spearman_factor_vs_effective"] for x in pair_saliency_history
                if x.get("spearman_factor_vs_effective") is not None
            ])) if any(x.get("spearman_factor_vs_effective") is not None for x in pair_saliency_history) else None,
            "mean_factor_effective_selection_jaccard": float(np.mean([
                x["factor_vs_effective_selection_overlap"]["jaccard"] for x in pair_saliency_history
            ])),
            "mean_effective_mass_ratio_factor_to_effective": float(np.mean([
                x["effective_mass_ratio_factor_to_effective"] for x in pair_saliency_history
            ])),
            "mean_effective_mass_ratio_actual_to_effective": float(np.mean([
                x["effective_mass_ratio_actual_to_effective"] for x in pair_saliency_history
            ])),
        }

        # Aggregate reparameterization diagnostics by scheme/scale.
        reparam_groups = defaultdict(list)
        for x in pair_saliency_history:
            for r in x.get("reparameterization_diagnostics", []):
                if r.get("scheme") == "global_scale":
                    key = f"global_scale={r.get('scale')}"
                else:
                    key = str(r.get("scheme"))
                reparam_groups[key].append(r)

        pair_summary["reparameterization_summary"] = {}
        for key, rows in sorted(reparam_groups.items()):
            pair_summary["reparameterization_summary"][key] = {
                "count": int(len(rows)),
                "mean_selection_jaccard_with_original_factor": float(np.mean([
                    r["overlap_with_original_factor_selection"]["jaccard"] for r in rows
                ])),
                "mean_selection_jaccard_with_effective": float(np.mean([
                    r["overlap_with_effective_selection"]["jaccard"] for r in rows
                ])),
                "mean_effective_mass_ratio_reparam_to_effective": float(np.mean([
                    r["effective_mass_ratio_reparam_to_effective"] for r in rows
                ])),
                "mean_spearman_factor_vs_reparam_factor": float(np.mean([
                    r["spearman_factor_vs_reparam_factor"] for r in rows
                    if r.get("spearman_factor_vs_reparam_factor") is not None
                ])) if any(r.get("spearman_factor_vs_reparam_factor") is not None for r in rows) else None,
            }
            if any("max_effective_score_relative_error_after_reparam" in r for r in rows):
                pair_summary["reparameterization_summary"][key]["max_effective_score_relative_error_after_reparam"] = float(max([
                    r.get("max_effective_score_relative_error_after_reparam", 0.0) for r in rows
                ]))

        pair_summary_path = os.path.join(training_args.output_dir, "pair_saliency_diagnostics_summary.json")
        with open(pair_summary_path, "w", encoding="utf-8") as fout:
            json.dump(pair_summary, fout, ensure_ascii=False, indent=2)
        logger.info(f"Saved pair-saliency diagnostics summary to {pair_summary_path}")


    if _is_main() and len(sn_encoder_compression_history) > 0:
        sn_enc_history_path = os.path.join(training_args.output_dir, "sn_encoder_compression_history.json")
        with open(sn_enc_history_path, "w", encoding="utf-8") as fout:
            json.dump(sn_encoder_compression_history, fout, ensure_ascii=False, indent=2)
        logger.info(f"Saved SN-encoder compression history to {sn_enc_history_path}")

        enc_groups = defaultdict(list)
        for row in sn_encoder_compression_history:
            enc_groups[row.get("sn_encoder_mode", "unknown")].append(row)
        sn_enc_summary = {
            "num_records": int(len(sn_encoder_compression_history)),
            "encoders": {},
        }
        for enc_name, rows in sorted(enc_groups.items()):
            sn_enc_summary["encoders"][enc_name] = {
                "num_records": int(len(rows)),
                "mean_density": float(np.mean([float(r.get("density", 0.0)) for r in rows])),
                "mean_nonzero": float(np.mean([int(r.get("nonzero", 0)) for r in rows])),
                "mean_nonzero_tensor_count": float(np.mean([int(r.get("nonzero_tensor_count", 0)) for r in rows])),
                "mean_candidate_tensor_count": float(np.mean([int(r.get("candidate_tensor_count", 0)) for r in rows])),
                "mean_candidate_unit_count": float(np.mean([int(r.get("candidate_unit_count", 0)) for r in rows])),
                "mean_candidate_selection_cost": float(np.mean([int(r.get("candidate_selection_cost", 0)) for r in rows])),
                "mean_packet_num": float(np.mean([int(r.get("packet_num", 0)) for r in rows])),
            }
        sn_enc_summary_path = os.path.join(training_args.output_dir, "sn_encoder_compression_summary.json")
        with open(sn_enc_summary_path, "w", encoding="utf-8") as fout:
            json.dump(sn_enc_summary, fout, ensure_ascii=False, indent=2)
        logger.info(f"Saved SN-encoder compression summary to {sn_enc_summary_path}")

    if _is_main() and len(baseline_compression_history) > 0:
        baseline_history_path = os.path.join(training_args.output_dir, "baseline_compression_history.json")
        with open(baseline_history_path, "w", encoding="utf-8") as fout:
            json.dump(baseline_compression_history, fout, ensure_ascii=False, indent=2)
        logger.info(f"Saved migrated-baseline compression history to {baseline_history_path}")

        method_groups = defaultdict(list)
        for row in baseline_compression_history:
            method_groups[row.get("normalized_method", "unknown")].append(row)

        baseline_summary = {
            "num_records": int(len(baseline_compression_history)),
            "methods": {},
        }
        for m_name, rows in sorted(method_groups.items()):
            densities = [float(r.get("density", 0.0)) for r in rows]
            nnzs = [int(r.get("nonzero", 0)) for r in rows]
            tensor_counts = [int(r.get("nonzero_tensor_count", 0)) for r in rows]
            packets = [int(r.get("packet_num", 0)) for r in rows]
            residual_norms = [float(r["residual_norm"]) for r in rows if r.get("residual_norm") is not None]
            baseline_summary["methods"][m_name] = {
                "num_records": int(len(rows)),
                "mean_density": float(np.mean(densities)) if densities else 0.0,
                "mean_nonzero": float(np.mean(nnzs)) if nnzs else 0.0,
                "mean_nonzero_tensor_count": float(np.mean(tensor_counts)) if tensor_counts else 0.0,
                "mean_packet_num": float(np.mean(packets)) if packets else 0.0,
                "mean_residual_norm": float(np.mean(residual_norms)) if residual_norms else None,
            }

        baseline_summary_path = os.path.join(training_args.output_dir, "baseline_compression_summary.json")
        with open(baseline_summary_path, "w", encoding="utf-8") as fout:
            json.dump(baseline_summary, fout, ensure_ascii=False, indent=2)
        logger.info(f"Saved migrated-baseline compression summary to {baseline_summary_path}")


    # ========== 保存 Adapter ==========
    peft_model_id = os.path.join(training_args.output_dir, "adapter")

    if _is_main():
        global_model.save_pretrained(peft_model_id)
        tokenizer.save_pretrained(peft_model_id)
        logger.info(f"Saved LoRA adapter/tokenizer to {peft_model_id}")

        if method == "pilora":
            save_pilora_ref(peft_model_id, global_model)
            logger.info(f"[PILoRA] Saved pilora_ref.pt to {peft_model_id}")

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

