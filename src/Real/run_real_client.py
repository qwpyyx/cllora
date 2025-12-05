# run_real_client.py
import os
import sys
import torch
import socket
import logging
import math
import copy
import transformers
from transformers import HfArgumentParser
from fed_socket import send_data, recv_data
from src.fed_continual_state import ContinualState

# === 导入原有代码库 ===
# 确保当前目录在 path 中以便导入
sys.path.append(os.getcwd())

# 从原有代码导入配置类和工具函数
from src.run_uie_lora import ModelArguments, DataTrainingArguments, UIETrainingArguments, FederatedArguments
from src.federated_uie_lora import (
    build_model_and_tokenizer,
    partition_dataset,
    _trainer_unwrap_model,
    compute_fisher_diag,
    get_lora_trainable_keys,
    load_dataset,
    CURRENT_DIR,
    gen_cache_path,
    collator_for,
    _trainer_wait_for_everyone
)
from src.uie_trainer_lora import UIETrainer



logger = logging.getLogger("FedClient")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UIETrainingArguments, FederatedArguments))
    # 增加真实分布式的参数
    parser.add_argument("--server_ip", type=str, required=True, help="Server IP address")
    parser.add_argument("--server_port", type=int, default=9999, help="Server port")
    parser.add_argument("--client_id", type=int, required=True, help="My client ID (0-3)")

    # 解析参数
    model_args, data_args, training_args, fed_args, extra_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)

    # 解析 extra_args 获取 server_ip 等 (因为 HfArgumentParser 可能没包含上面的自定义参数)
    # 简单起见，我们假设通过 --server_ip 传进来的参数已经被包含在 fed_args 或者我们手动处理
    # 更好的方式是扩展 FederatedArguments，但为了不改原代码，我们在这里手动解析剩下的 sys.argv
    import argparse
    manual_parser = argparse.ArgumentParser()
    manual_parser.add_argument("--server_ip", type=str, default="127.0.0.1")
    manual_parser.add_argument("--server_port", type=int, default=9999)
    manual_parser.add_argument("--client_id", type=int, default=0)
    manual_args, _ = manual_parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(
        f"Starting Client {manual_args.client_id} connecting to {manual_args.server_ip}:{manual_args.server_port}")

    # ================= 1. 数据准备 (复用原有逻辑) =================
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

    train_dataset = raw_datasets["train"]
    # 关键：使用相同的种子进行切分，确保每个节点切分一致
    # 然后只取属于自己的那一份
    all_client_datasets = partition_dataset(
        train_dataset,
        fed_args.num_clients,  # 比如 4
        fed_args.dirichlet_alpha,
        base_seed=fed_args.federated_seed
    )
    my_dataset = all_client_datasets[manual_args.client_id]
    logger.info(f"Data partition loaded. My dataset size: {len(my_dataset)}")

    # ================= 2. 模型与 Trainer 初始化 =================
    model, tokenizer = build_model_and_tokenizer(model_args)

    # 初始化 Trainer
    trainer = UIETrainer(
        model=model,
        args=training_args,
        train_dataset=my_dataset,
        tokenizer=tokenizer,
        data_collator=collator_for(model),
        state=None  # 初始状态为空，后面加载
    )

    # 状态持久化路径
    state_path = os.path.join(training_args.output_dir, f"client_{manual_args.client_id}_state.pt")

    # 仅主进程负责网络通信
    is_main_process = training_args.local_rank in [-1, 0]
    sock = None

    if is_main_process:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((manual_args.server_ip, manual_args.server_port))
        logger.info("Connected to server.")

    # ================= 3. 联邦训练循环 =================
    while True:
        # --- 阶段 A: 接收全局模型 ---
        global_weights = None
        stop_signal = False

        if is_main_process:
            logger.info("Waiting for global model...")
            packet = recv_data(sock)
            if packet and 'stop' in packet:
                stop_signal = True
            elif packet:
                global_weights = packet.get('weights')
                current_round = packet.get('round')
                logger.info(f"Received Round {current_round}")

        # 广播停止信号
        # 注意：这里需要完善的多卡同步逻辑，简单起见假设单卡或 reliable accelerate
        if stop_signal:
            break

        # 如果收到全局权重 (Round > 0)，加载它
        if global_weights is not None:
            # 将 CPU 权重转为 GPU
            device = trainer.args.device
            global_weights_gpu = {k: v.to(device) for k, v in global_weights.items()}

            # 确保 Unwrap
            model_to_load = _trainer_unwrap_model(trainer)
            model_to_load.load_state_dict(global_weights_gpu, strict=False)
            logger.info("Global weights loaded.")

        _trainer_wait_for_everyone(trainer)

        # --- 阶段 B: 加载 Adaptive 状态 (如果存在) ---
        client_state = ContinualState()
        if os.path.exists(state_path) and training_args.method == "adaptive":
            client_state = ContinualState.load(state_path)
            trainer.continual_state = client_state
            logger.info(f"Loaded ContinualState from {state_path}")

        # --- 阶段 C: 训练 ---
        # 重新初始化 Optimizer (这对 Adaptive 方法很重要)
        trainer.create_optimizer_and_scheduler(num_training_steps=100)  # 这里的 steps 会在 train 内部根据 dataloader 长度重算

        # 调用原有的 train 逻辑
        # 如果是 Adaptive 方法，trainer.train 会返回 (delta, F_client, theta_last)
        # 如果是 lora_origin，它只进行训练

        logger.info("Start local training...")
        if training_args.method == "adaptive" and data_args.task > 1:
            # 提取 global_state 用于 adaptive 计算
            lora_keys = get_lora_trainable_keys(_trainer_unwrap_model(trainer))
            global_state_cpu = {k: v.detach().cpu() for k, v in _trainer_unwrap_model(trainer).named_parameters() if
                                k in lora_keys}

            delta, F_client, theta_last = trainer.train(
                task_id=data_args.task,
                base_params=global_state_cpu,
                cid=manual_args.client_id
            )
            # 更新状态
            client_state.update(F_client, theta_last)
            # 保存状态到磁盘
            if is_main_process:
                client_state.save(state_path)

        else:
            # Task 1 或 lora_origin
            trainer.train(task_id=data_args.task)

            # 如果是 task 1 的 adaptive，需要计算 Fisher
            if training_args.method == "adaptive" and data_args.task == 1:
                trained_model = _trainer_unwrap_model(trainer)
                # 计算 Fisher (使用原有函数)
                train_dl = trainer.get_train_dataloader()
                F_client = compute_fisher_diag(trained_model, train_dl)

                # 获取参数
                lora_keys = get_lora_trainable_keys(trained_model)
                theta_last = {k: v.detach().cpu() for k, v in trained_model.named_parameters() if k in lora_keys}

                client_state.update(F_client, theta_last)
                if is_main_process:
                    client_state.save(state_path)

        _trainer_wait_for_everyone(trainer)

        # --- 阶段 D: 发送结果 ---
        if is_main_process:
            logger.info("Extracting and sending weights...")
            trained_model = _trainer_unwrap_model(trainer)
            lora_keys = get_lora_trainable_keys(trained_model)

            # 提取权重
            local_weights = {k: v.detach().cpu() for k, v in trained_model.named_parameters() if k in lora_keys}

            payload = {
                'weights': local_weights,
                'num_samples': len(my_dataset)  # <--- 新增这行，my_dataset 在前文已定义
            }
            send_data(sock, payload)
            logger.info(f"Weights sent (samples={len(my_dataset)}).")

    if is_main_process:
        sock.close()


if __name__ == "__main__":
    main()