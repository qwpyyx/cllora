#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/qiuwenqi/.cache/huggingface
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN          # 或 INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1



port=$(shuf -i25000-30000 -n1)
method=adaptive
lora_rank=16
lamda_2=0
lamda_1=0
# bash scripts/order_1.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path /home/qiuwenqi/LLM/models/t5-base \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_1/$method/outputs/1-dbpedia \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 5e-01 \
   --num_train_epochs 1 \
   --run_name order1_round1 \
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
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method
sleep 5

#CUDA_VISIBLE_DEVICES=0,1,2,3,4 deepspeed --master_port $port src/run_uie_lora.py \
#   --do_train \
#   --do_predict \
#   --predict_with_generate \
#   --lora_dim $lora_rank \
#   --model_name_or_path logs_and_outputs/order_1/$method/outputs/1-dbpedia/adapter \
#   --data_dir CL_Benchmark \
#   --task_config_dir configs/order1_configs/amazon \
#   --instruction_file configs/instruction_config.json \
#   --instruction_strategy single \
#   --output_dir logs_and_outputs/order_1/$method/outputs/2-amazon \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 128 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate 1e-03 \
#   --num_train_epochs 1 \
#   --deepspeed configs/ds_configs/stage2.config \
#   --run_name order1_round2 \
#   --max_source_length 512 \
#   --max_target_length 50 \
#   --generation_max_length 50 \
#   --add_task_name True \
#   --add_dataset_name True \
#   --overwrite_output_dir \
#   --overwrite_cache \
#   --lr_scheduler_type constant \
#   --warmup_steps 0 \
#   --logging_strategy steps \
#   --logging_steps 10 \
#   --evaluation_strategy no \
#   --save_strategy no \
#   --save_steps 1500 \
#   --lamda_1 $lamda_1 \
#   --lamda_2 $lamda_2
#
#sleep 5
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4 deepspeed --master_port $port src/run_uie_lora.py \
#   --do_train \
#   --do_predict \
#   --predict_with_generate \
#   --lora_dim $lora_rank \
#   --model_name_or_path logs_and_outputs/order_1/$method/outputs/2-amazon/adapter \
#   --data_dir CL_Benchmark \
#   --task_config_dir configs/order1_configs/yahoo \
#   --instruction_file configs/instruction_config.json \
#   --instruction_strategy single \
#   --output_dir logs_and_outputs/order_1/$method/outputs/3-yahoo \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 128 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate 1e-03 \
#   --num_train_epochs 1 \
#   --deepspeed configs/ds_configs/stage2.config \
#   --run_name order1_round3 \
#   --max_source_length 512 \
#   --max_target_length 50 \
#   --generation_max_length 50 \
#   --add_task_name True \
#   --add_dataset_name True \
#   --overwrite_output_dir \
#   --overwrite_cache \
#   --lr_scheduler_type constant \
#   --warmup_steps 0 \
#   --logging_strategy steps \
#   --logging_steps 10 \
#   --evaluation_strategy no \
#   --save_strategy no \
#   --save_steps 1500 \
#   --lamda_1 $lamda_1 \
#   --lamda_2 $lamda_2
#
#sleep 5
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4 deepspeed --master_port $port src/run_uie_lora.py \
#   --do_train \
#   --do_predict \
#   --predict_with_generate \
#   --lora_dim $lora_rank \
#   --model_name_or_path logs_and_outputs/order_1/$method/outputs/3-yahoo/adapter \
#   --data_dir CL_Benchmark \
#   --task_config_dir configs/order1_configs/agnews \
#   --instruction_file configs/instruction_config.json \
#   --instruction_strategy single \
#   --output_dir logs_and_outputs/order_1/$method/outputs/4-agnews \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 128 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate 1e-03 \
#   --num_train_epochs 1 \
#   --deepspeed configs/ds_configs/stage2.config \
#   --run_name order1_round4 \
#   --max_source_length 512 \
#   --max_target_length 50 \
#   --generation_max_length 50 \
#   --add_task_name True \
#   --add_dataset_name True \
#   --overwrite_output_dir \
#   --overwrite_cache \
#   --lr_scheduler_type constant \
#   --warmup_steps 0 \
#   --logging_strategy steps \
#   --logging_steps 10 \
#   --evaluation_strategy no \
#   --save_strategy no \
#   --save_steps 1500 \
#   --lamda_1 $lamda_1 \
#   --lamda_2 $lamda_2
