#!/bin/bash
set -x

export PYTHONWARNINGS="ignore"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/qiuwenqi/.cache/huggingface
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

method=lora_origin
lora_rank=8
lamda_2=0
lamda_1=0
lr=1e-04
radius=0
com_budget=2200
is_random_select=False
di_alpha=0.5
seed=42

data_dir=/home/qiuwenqi/LLM/Third_work/data_dolly/dolly_uie/data
task_config_dir=/home/qiuwenqi/LLM/Third_work/data_dolly/dolly_uie/task_config
model_path=/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct

if [ "$is_random_select" = "True" ]; then
    random_layer_param="--random_layer_selection True"
else
    random_layer_param=""
fi


############################################
# Exp 1: Dolly tensor-level Top-K
############################################

port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file script_accelerate/accelerate_config.yaml \
   --main_process_port $port \
   src/run_uie_lora.py \
   --report_to none \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path $model_path \
   --data_dir $data_dir \
   --task_config_dir $task_config_dir \
   --output_dir results/Qwen2_dolly/lora_exp_dolly_alpha05/2200_seed42_tensor_topk_5client \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha $di_alpha \
   --partition_strategy label \
   --partition_label_key Dataset \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name dolly_tensor_topk_seed42_5client \
   --max_source_length 512 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 150 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed $seed \
   --method $method \
   --task 1 \
   --radius $radius \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False \
   --upload_atomic_mode tensor \
   --upload_diversity_mode none \
   --diagnose_residual_errors False \
   --lora_residual_accumulation False \
   $random_layer_param

sleep 5


############################################
# Exp 2: Dolly A/B pair independent Top-K
############################################

port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file script_accelerate/accelerate_config.yaml \
   --main_process_port $port \
   src/run_uie_lora.py \
   --report_to none \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path $model_path \
   --data_dir $data_dir \
   --task_config_dir $task_config_dir \
   --output_dir results/Qwen2_dolly/lora_exp_dolly_alpha05/2200_seed42_abpair_topk_5client \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha $di_alpha \
   --partition_strategy label \
   --partition_label_key Dataset \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name dolly_abpair_topk_seed42_5client \
   --max_source_length 512 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 150 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed $seed \
   --method $method \
   --task 1 \
   --radius $radius \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False \
   --upload_atomic_mode ab_pair \
   --upload_diversity_mode none \
   --diagnose_residual_errors False \
   --lora_residual_accumulation False \
   $random_layer_param

sleep 5


############################################
# Exp 3: Dolly A/B pair + group mask G=4
############################################

port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file script_accelerate/accelerate_config.yaml \
   --main_process_port $port \
   src/run_uie_lora.py \
   --report_to none \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path $model_path \
   --data_dir $data_dir \
   --task_config_dir $task_config_dir \
   --output_dir results/Qwen2_dolly/lora_exp_dolly_alpha05/2200_seed42_abpair_groupmask_g4_5client \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha $di_alpha \
   --partition_strategy label \
   --partition_label_key Dataset \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name dolly_abpair_groupmask_g4_seed42_5client \
   --max_source_length 512 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 150 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed $seed \
   --method $method \
   --task 1 \
   --radius $radius \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False \
   --upload_atomic_mode ab_pair \
   --upload_diversity_mode group_mask \
   --diversity_num_groups 4 \
   --diagnose_residual_errors False \
   --lora_residual_accumulation False \
   $random_layer_param

sleep 5


############################################
# Exp 4: Dolly A/B pair + coverage penalty beta=0.05
############################################

port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file script_accelerate/accelerate_config.yaml \
   --main_process_port $port \
   src/run_uie_lora.py \
   --report_to none \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path $model_path \
   --data_dir $data_dir \
   --task_config_dir $task_config_dir \
   --output_dir results/Qwen2_dolly/lora_exp_dolly_alpha05/2200_seed42_abpair_covpenalty_beta005_5client \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha $di_alpha \
   --partition_strategy label \
   --partition_label_key Dataset \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name dolly_abpair_covpenalty_beta005_seed42_5client \
   --max_source_length 512 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 150 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed $seed \
   --method $method \
   --task 1 \
   --radius $radius \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False \
   --upload_atomic_mode ab_pair \
   --upload_diversity_mode coverage_penalty \
   --coverage_penalty_beta 0.05 \
   --diagnose_residual_errors False \
   --lora_residual_accumulation False \
   $random_layer_param

sleep 5


############################################
# Exp 5: Dolly A/B pair + coverage penalty beta=0.1
############################################

port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file script_accelerate/accelerate_config.yaml \
   --main_process_port $port \
   src/run_uie_lora.py \
   --report_to none \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path $model_path \
   --data_dir $data_dir \
   --task_config_dir $task_config_dir \
   --output_dir results/Qwen2_dolly/lora_exp_dolly_alpha05/2200_seed42_abpair_covpenalty_beta01_5client \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha $di_alpha \
   --partition_strategy label \
   --partition_label_key Dataset \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name dolly_abpair_covpenalty_beta01_seed42_5client \
   --max_source_length 512 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 150 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed $seed \
   --method $method \
   --task 1 \
   --radius $radius \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False \
   --upload_atomic_mode ab_pair \
   --upload_diversity_mode coverage_penalty \
   --coverage_penalty_beta 0.1 \
   --diagnose_residual_errors False \
   --lora_residual_accumulation False \
   $random_layer_param

echo "All Dolly 5-client experiments finished."