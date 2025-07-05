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
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import matplotlib.pyplot as plt
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from uie_collator import DataCollatorForUIE
from uie_trainer_lora import UIETrainer, DenserEvalCallback, skip_instructions
from compute_metrics import compute_metrics, compute_grouped_metrics
from model.llama import LlamaForCausalLM_with_lossmask
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from uie_dataset_lora import gen_cache_path
#from plot.data_distribution import compare
from run_uie_lora import ModelArguments, DataTrainingArguments, UIETrainingArguments, FederatedArguments

logger = logging.getLogger(__name__)
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

    model.print_trainable_parameters()
    model.resize_token_embeddings(len(tokenizer))

    if 'llama' in model_args.model_name_or_path.lower():
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        model.generation_config.pad_token_id = 1

    # LoRA
    # for name, param in model.named_parameters():
    #     if 'lora_' in name:
    #         param.requires_grad = True
    #     elif 'shared' in name:
    #         param.requires_grad = False

    # O-LoRA
    for name, param in model.named_parameters():
        if name.find("loranew_") != -1:
            param.requires_grad = True
        elif name.find("lora_") != -1:
            param.requires_grad = False
        # this module should always be frozen because we change the vocabulary
        elif name.find("shared") != -1:
            param.requires_grad = False

    return model, tokenizer

def run_federated_training(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: UIETrainingArguments, fed_args: FederatedArguments):

    # loading logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler()])
    logger.info("Running federated learning mode")
    log_level = training_args.get_process_log_level()
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
    logger.info(f"Training/evaluation parameters {training_args}")


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

    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        # 对生成式模型的输出进行后处理
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

    global_model = model

    device = next(global_model.parameters()).device

    for rnd in range(fed_args.global_rounds):
        logger.info(f"Global round {rnd + 1}/{fed_args.global_rounds}")
        selected = random.sample(range(fed_args.num_clients), min(fed_args.clients_per_round, fed_args.num_clients))

        # 获取要聚合的 lora 参数名
        lora_keys = get_lora_trainable_keys(global_model)

        global_state_cpu = {
            k: v.detach().cpu()
            for k, v in global_model.state_dict().items()
        }

        # 初始化 LoRA 聚合容器
        aggregated = {
            k: torch.zeros_like(global_state_cpu[k])
            for k in lora_keys
        }

        total = 0

        for cid in selected:
            local_model = copy.deepcopy(global_model)
            local_args = copy.deepcopy(base_args)
            # local_args.output_dir = os.path.join(training_args.output_dir, f"client_{cid}")  # 或者一个临时目录
            local_args.resume_from_checkpoint = None
            local_args.num_train_epochs = fed_args.local_epochs
            local_args.save_strategy = "no"
            local_args.logging_strategy = "no"
            local_args.evaluation_strategy = "no"

            trainer = UIETrainer(
                model=local_model,
                args=local_args,
                train_dataset=client_datasets[cid],
                tokenizer=tokenizer,
                data_collator=collator_for(local_model),
                compute_metrics=None,
                callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
            )
            # optimizer = trainer.optimizer
            # print("Param groups:", optimizer.param_groups)

            trainer.train()

            weight = len(client_datasets[cid])
            state_dict = local_model.state_dict()
            for k in lora_keys:
                aggregated[k] += state_dict[k].detach().cpu() * weight
            total += weight

        for k in lora_keys:
            aggregated[k] /= max(total, 1)

        # 更新 global model 的 LoRA 权重
        update_dict = {
            k: aggregated[k].to(device)
            for k in lora_keys
        }
        # strict=False 会忽略掉 state_dict 里没给到的其他 key
        global_model.load_state_dict(update_dict, strict=False)

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

