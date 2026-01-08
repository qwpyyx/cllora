#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/qiuwenqi/.cache/huggingface
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN          # 或 INFO
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
#export NCCL_P2P_DISABLE=1


port=$(shuf -i25000-30000 -n1)
method=lorm
lora_rank=8
lamda_2=0
lamda_1=0
lr=1e-05
radius=1.0
com_budget=0
# bash scripts/order_1_adaptive.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &

#CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
#   --main_process_port $port \
#   src/run_uie_lora.py \
#   --do_train \
#   --do_predict \
#   --predict_with_generate \
#   --lora_dim $lora_rank \
#   --model_name_or_path /home/qiuwenqi/LLM/models/llama-2-7b-hf \
#   --data_dir CL_Benchmark \
#   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/mnli \
#   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/1-mnli\
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --gradient_accumulation_steps 2 \
#   --global_rounds 5 \
#   --local_epochs 10 \
#   --num_clients 50 \
#   --clients_per_round 5 \
#   --dirichlet_alpha 10 \
#   --comm_budget $com_budget \
#   --learning_rate $lr \
#   --run_name order2_round1 \
#   --max_source_length 512 \
#   --max_target_length 10 \
#   --generation_max_length 10 \
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
#   --save_steps 150 \
#   --lamda_1 $lamda_1 \
#   --lamda_2 $lamda_2 \
#   --federated_seed 42 \
#   --method $method \
#   --task 1 \
#   --radius $radius \
#   --gradient_checkpointing False \
#   --bf16 True \
#   --ddp_find_unused_parameters True
#
#sleep 5
#
#CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
#   --main_process_port $port \
#   src/run_uie_lora.py \
#   --do_train \
#   --do_predict \
#   --predict_with_generate \
#   --lora_dim $lora_rank \
#   --model_name_or_path results/Longseq/order_4_llama/$method/outputs/$lr/1-mnli/adapter \
#   --data_dir CL_Benchmark \
#   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/cb \
#   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/2-cb \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --gradient_accumulation_steps 2 \
#   --global_rounds 5 \
#   --local_epochs 10 \
#   --num_clients 50 \
#   --clients_per_round 5 \
#   --dirichlet_alpha 10 \
#   --comm_budget $com_budget \
#   --learning_rate $lr \
#   --run_name order2_round2 \
#   --max_source_length 512 \
#   --max_target_length 10 \
#   --generation_max_length 10 \
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
#   --lamda_2 $lamda_2 \
#   --federated_seed 42 \
#   --method $method \
#   --task 2 \
#   --radius $radius \
#   --gradient_checkpointing False \
#   --bf16 True \
#   --ddp_find_unused_parameters True
#
#sleep 5
#
## Task 3: task1590
#CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
#   --main_process_port $port \
#   src/run_uie_lora.py \
#   --do_train \
#   --do_predict \
#   --predict_with_generate \
#   --lora_dim $lora_rank \
#   --model_name_or_path results/Longseq/order_4_llama/$method/outputs/$lr/2-cb/adapter \
#   --data_dir CL_Benchmark \
#   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/wic \
#   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/3-wic \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --gradient_accumulation_steps 2 \
#   --global_rounds 5 \
#   --local_epochs 10 \
#   --num_clients 50 \
#   --clients_per_round 5 \
#   --dirichlet_alpha 10 \
#   --comm_budget $com_budget \
#   --learning_rate $lr \
#   --run_name order2_round3 \
#   --max_source_length 512 \
#   --max_target_length 10 \
#   --generation_max_length 10 \
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
#   --lamda_2 $lamda_2 \
#   --federated_seed 42 \
#   --method $method \
#   --task 3 \
#   --radius $radius \
#   --gradient_checkpointing False \
#   --bf16 True \
#   --ddp_find_unused_parameters True
#
#sleep 5
#
## Task 4: task639
#CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
#   --main_process_port $port \
#   src/run_uie_lora.py \
#   --do_train \
#   --do_predict \
#   --predict_with_generate \
#   --lora_dim $lora_rank \
#   --model_name_or_path results/Longseq/order_4_llama/$method/outputs/$lr/3-wic/adapter \
#   --data_dir CL_Benchmark \
#   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/copa \
#   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/4-copa \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --gradient_accumulation_steps 2 \
#   --global_rounds 5 \
#   --local_epochs 10 \
#   --num_clients 50 \
#   --clients_per_round 5 \
#   --dirichlet_alpha 10 \
#   --comm_budget $com_budget \
#   --learning_rate $lr \
#   --run_name order2_round4 \
#   --max_source_length 512 \
#   --max_target_length 10 \
#   --generation_max_length 10 \
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
#   --lamda_2 $lamda_2 \
#   --federated_seed 42 \
#   --method $method \
#   --task 4 \
#   --radius $radius \
#   --gradient_checkpointing False \
#   --bf16 True \
#   --ddp_find_unused_parameters True
#
#sleep 5
#
## Task 5: task1572
#CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
#   --main_process_port $port \
#   src/run_uie_lora.py \
#   --do_train \
#   --do_predict \
#   --predict_with_generate \
#   --lora_dim $lora_rank \
#   --model_name_or_path results/Longseq/order_4_llama/$method/outputs/$lr/4-copa/adapter \
#   --data_dir CL_Benchmark \
#   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/qqp \
#   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/5-qqp \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --gradient_accumulation_steps 2 \
#   --global_rounds 5 \
#   --local_epochs 10 \
#   --num_clients 50 \
#   --clients_per_round 5 \
#   --dirichlet_alpha 10 \
#   --comm_budget $com_budget \
#   --learning_rate $lr \
#   --run_name order2_round5 \
#   --max_source_length 512 \
#   --max_target_length 10 \
#   --generation_max_length 10 \
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
#   --lamda_2 $lamda_2 \
#   --federated_seed 42 \
#   --method $method \
#   --task 5 \
#   --radius $radius \
#   --gradient_checkpointing False \
#   --bf16 True \
#   --ddp_find_unused_parameters True
#
#sleep 5

# Task 6: task1687
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_4_llama/$method/outputs/$lr/5-qqp/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/boolq \
   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/6-boolq \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name order2_round6 \
   --max_source_length 512 \
   --max_target_length 10 \
   --generation_max_length 10 \
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
   --method $method \
   --task 6 \
   --radius $radius \
   --gradient_checkpointing False \
   --bf16 True \
   --ddp_find_unused_parameters True

sleep 5

# Task 7: task591
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_4_llama/$method/outputs/$lr/6-boolq/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/rte \
   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/7-rte \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name order2_round7 \
   --max_source_length 512 \
   --max_target_length 10 \
   --generation_max_length 10 \
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
   --method $method \
   --task 7 \
   --radius $radius \
   --gradient_checkpointing False \
   --bf16 True \
   --ddp_find_unused_parameters True
sleep 5

# Task 8: task363
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_4_llama/$method/outputs/$lr/7-rte/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/imdb \
   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/8-imdb \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name order2_round8 \
   --max_source_length 512 \
   --max_target_length 10 \
   --generation_max_length 10 \
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
   --method $method \
   --task 8 \
   --radius $radius \
   --gradient_checkpointing False \
   --bf16 True \
   --ddp_find_unused_parameters True

sleep 5

# Task 9: task1510
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_4_llama/$method/outputs/$lr/8-imdb/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/yelp \
   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/9-yelp \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name order2_round9 \
   --max_source_length 512 \
   --max_target_length 10 \
   --generation_max_length 10 \
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
   --method $method \
   --task 9 \
   --radius $radius \
   --gradient_checkpointing False \
   --bf16 True \
   --ddp_find_unused_parameters True

sleep 5

# Task 10: task1729
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_4_llama/$method/outputs/$lr/9-yelp/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/amazon \
   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/10-amazon \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name order2_round10 \
   --max_source_length 512 \
   --max_target_length 10 \
   --generation_max_length 10 \
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
   --method $method \
   --task 10 \
   --radius $radius \
   --gradient_checkpointing False \
   --bf16 True \
   --ddp_find_unused_parameters True

sleep 5

# Task 11: task181
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_4_llama/$method/outputs/$lr/10-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/sst2 \
   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/11-sst2 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name order2_round11 \
   --max_source_length 512 \
   --max_target_length 10 \
   --generation_max_length 10 \
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
   --method $method \
   --task 11 \
   --radius $radius \
   --gradient_checkpointing False \
   --bf16 True \
   --ddp_find_unused_parameters True

sleep 5

# Task 12: task511
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_4_llama/$method/outputs/$lr/11-sst2/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/dbpedia \
   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/12-dbpedia \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name order2_round12 \
   --max_source_length 512 \
   --max_target_length 10 \
   --generation_max_length 10 \
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
   --method $method \
   --task 12 \
   --radius $radius \
   --gradient_checkpointing False \
   --bf16 True \
   --ddp_find_unused_parameters True

sleep 5

# Task 13: task002
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_4_llama/$method/outputs/$lr/12-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/agnews \
   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/13-agnews \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name order2_round13 \
   --max_source_length 512 \
   --max_target_length 10 \
   --generation_max_length 10 \
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
   --method $method \
   --task 13 \
   --radius $radius \
   --gradient_checkpointing False \
   --bf16 True \
   --ddp_find_unused_parameters True

sleep 5

# Task 14: task1290
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_4_llama/$method/outputs/$lr/13-agnews/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/multirc \
   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/14-multirc \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name order2_round14 \
   --max_source_length 512 \
   --max_target_length 10 \
   --generation_max_length 10 \
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
   --method $method \
   --task 14 \
   --radius $radius \
   --gradient_checkpointing False \
   --bf16 True \
   --ddp_find_unused_parameters True

sleep 5

# Task 15: task875
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_4_llama/$method/outputs/$lr/14-multirc/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order4_t5_configs/yahoo \
   --output_dir results/Longseq/order_4_llama/$method/outputs/$lr/15-yahoo \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget $com_budget \
   --learning_rate $lr \
   --run_name order2_round15 \
   --max_source_length 512 \
   --max_target_length 10 \
   --generation_max_length 10 \
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
   --method $method \
   --task 15 \
   --radius $radius \
   --gradient_checkpointing False \
   --bf16 True \
   --ddp_find_unused_parameters True