#!/bin/bash
set -x
export PYTHONWARNINGS="ignore"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/qiuwenqi/.cache/huggingface
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN          # 或 INFO
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

port=$(shuf -i25000-30000 -n1)
method=adaptive
lora_rank=8
lamda_2=0
lamda_1=0
lr=1e-04
radius=1.0
com_budget=0
is_random_select=False
di_alpha=0.1
# 根据is_random_select生成对应的参数（核心修改）
if [ "$is_random_select" = "True" ]; then
    random_layer_param="--random_layer_selection True"
else
    random_layer_param=""
fi

# bash scripts/order_1_adaptive.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 | tee run_order3_llama_adaptive_5e-04.log

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --report_to none \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path /home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_llama_configs/task748_glucose_reverse_cause_event_detection \
   --output_dir results/SuperNI/order_2_qwen/$method/qwen_test/alpha/$di_alpha/outputs/$lr/1-task748_glucose_reverse_cause_event_detection \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 2 \
   --global_rounds 1 \
   --local_epochs 5 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha $di_alpha \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name order2_round1 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
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
   --federated_seed 42 \
   --method $method \
   --task 1 \
   --radius $radius \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False \
   $random_layer_param

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --report_to none \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2_qwen/$method/qwen_test/alpha/$di_alpha/outputs/$lr/1-task748_glucose_reverse_cause_event_detection/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_llama_configs/task073_commonsenseqa_answer_generation \
   --output_dir results/SuperNI/order_2_qwen/$method/qwen_test/alpha/$di_alpha/outputs/$lr/2-task073_commonsenseqa_answer_generation \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 2 \
   --global_rounds 1 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha $di_alpha \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name order2_round2 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
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
   --federated_seed 42 \
   --method $method \
   --task 2 \
   --radius $radius \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False \
   $random_layer_param