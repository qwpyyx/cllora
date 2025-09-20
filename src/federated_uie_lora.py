#!/usr/bin/env python
# coding=utf-8
"""Federated learning training loop for UIE LoRA models."""

import copy
import logging
import os
import random
from collections import defaultdict
from typing import List
import json
import datasets
import numpy as np
import torch
import math
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import matplotlib.pyplot as plt
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from uie_collator import DataCollatorForUIE
from uie_trainer_lora import UIETrainer, DenserEvalCallback, skip_instructions, compute_fisher
from compute_metrics import compute_metrics, compute_grouped_metrics
from model.llama import LlamaForCausalLM_with_lossmask
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from uie_dataset_lora import gen_cache_path
#from plot.data_distribution import compare
from run_uie_lora import ModelArguments, DataTrainingArguments, UIETrainingArguments, FederatedArguments
from torch.utils.data import DataLoader
from continual_fisher_client import ContinualFisherClient, ClientState
from fed_continual_state import ContinualState

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


def run_federated_training(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: UIETrainingArguments, fed_args: FederatedArguments):

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
        # TODO 会不会是skip_instructions的问题
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

    lora_params = {k: p for k, p in global_model.named_parameters() if "lora" in k}

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

    # 预计算所有LoRA层的成本（只需要计算一次，全局共享）
    layer_costs = {
        k: calculate_layer_packet_cost(p)
        for k, p in lora_params.items()
    }
    logger.info(f"预计算的LoRA层通信成本: {layer_costs['base_model.model.encoder.block.0.layer.0.SelfAttention.q.lora_A.default.weight']}")

    # 新增：跟踪客户端被选中的次数和最后选中轮次
    client_selection_tracker = {
        cid: {'count': 0, 'last_round': -1}
        for cid in range(fed_args.num_clients)
    }

    # 新增：记录当前任务中被选中的客户端
    current_task_selected_clients = set()

    # TODO 可能有问题
    client_state_dir = os.path.join(current_output_dir, "client_states")
    os.makedirs(client_state_dir, exist_ok=True)

    # —— 任务级缓存：只记录“本任务”的 Fisher 与该任务最后一次的 θ*
    reduce_mode = getattr(fed_args, "fisher_reduce_mode", "last")  # "last" 或 "mean"
    per_task_cache = {
        cid: {"F_sum": None, "count": 0, "F_last": None, "theta_last": None}
        for cid in range(fed_args.num_clients)
    }






    for rnd in range(fed_args.global_rounds):
        logger.info(f"Global round {rnd + 1}/{fed_args.global_rounds}")

        selected = client_rng.sample(
            range(fed_args.num_clients),
            min(fed_args.clients_per_round, fed_args.num_clients),
        )

        logger.info(f"Selected client {selected}")

        lora_keys = get_lora_trainable_keys(global_model)
        global_state_cpu = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        aggregated = {k: torch.zeros_like(global_state_cpu[k]) for k in lora_keys}
        total = 0

        for cid in selected:
            logger.info(f"Client ID: {cid}")
            # 更新客户端选择跟踪
            client_selection_tracker[cid]['count'] += 1
            client_selection_tracker[cid]['last_round'] = rnd
            current_task_selected_clients.add(cid)

            local_model = copy.deepcopy(global_model)
            local_args = copy.deepcopy(base_args)
            local_args.resume_from_checkpoint = None
            local_args.num_train_epochs = fed_args.local_epochs
            local_args.save_strategy = "no"
            local_args.logging_strategy = "no"
            local_args.evaluation_strategy = "no"

            # TODO 考虑如何把deepspeed兼容进来
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
                    F_full = compute_fisher(
                        local_model,
                        trainer.get_train_dataloader(),
                        engine=deepspeed_engine  # 关键：传入DeepSpeed引擎
                    )
                    F_client = {
                        k: F_full.get(k, torch.zeros_like(global_state_cpu[k]))
                        for k in lora_keys
                    }
                    theta_last = {k: state_dict[k].detach().cpu() for k in F_client.keys()}
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
                acc = per_task_cache[cid]

                # 记录该客户端在本任务的“最后一次” θ*
                acc["theta_last"] = theta_last  # ← 直接用 trainer 返回的

                if reduce_mode == "last":
                    # 只保留最新一轮
                    acc["F_last"] = {k: v.clone() for k, v in F_client.items()}
                else:  # "mean"
                    if acc["F_sum"] is None:
                        acc["F_sum"] = {k: v.clone() for k, v in F_client.items()}
                        acc["count"] = 1
                    else:
                        for k in F_client:
                            acc["F_sum"][k] += F_client[k]
                        acc["count"] += 1



            # Server update
            weight = len(client_datasets[cid])
            for k in lora_keys:
                aggregated[k] += delta[k] * weight
            total += weight

        for k in lora_keys:
            mu = aggregated[k] / max(total, 1)
            global_state_cpu[k] = global_state_cpu[k] - mu

        update_dict = {k: global_state_cpu[k].to(device) for k in lora_keys}
        global_model.load_state_dict(update_dict, strict=False)

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
            client_state.update(F_task, theta_star, gamma=gamma)  # 只在任务末更新历史
            client_state.save(client_state_path)

    # ========== 保存 Adapter ==========
    peft_model_id = os.path.join(training_args.output_dir, "adapter")
    global_model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    all_metrics.update({"adapter_saved": peft_model_id})
    logger.info(f"Saved LoRA adapter/tokenizer to {peft_model_id}")

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

    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()

    return all_metrics

