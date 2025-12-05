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
# bash scripts/order_1_adaptive.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config_file script_accelerate/accelerate_config_llama.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --report_to none \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path /home/qiuwenqi/LLM/models/llama-2-7b-hf \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/algorithm/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/yelp \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/1-yelp \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate $lr \
   --run_name order2_round1 \
   --max_source_length 512 \
   --max_target_length 10 \
   --generation_max_length 10 \
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
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False


sleep 5

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config_file script_accelerate/accelerate_config_llama.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --report_to none \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_3_llama/$method/llama/outputs/$lr/1-yelp/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/algorithm/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/amazon \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/2-amazon \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate $lr \
   --run_name order2_round2 \
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
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 150 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 2 \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False

sleep 5

# Task 3: task1590
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --report_to none \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_3_llama/$method/llama/outputs/$lr/2-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/algorithm/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/mnli \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/3-mnli \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate $lr \
   --run_name order2_round3 \
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
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 3 \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False

sleep 5

# Task 4: task639
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --report_to none \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_3_llama/$method/llama/outputs/$lr/3-mnli/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/algorithm/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/cb \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/4-cb \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate $lr \
   --run_name order2_round4 \
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
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 4 \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False

sleep 5

# Task 5: task1572
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_3_llama/$method/llama/outputs/$lr/4-cb/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/algorithm/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/copa \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/5-copa \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate $lr \
   --run_name order2_round5 \
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
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 5 \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False

sleep 5

# Task 6: task1687
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_3_llama/$method/llama/outputs/$lr/5-copa/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/algorithm/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/qqp \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/6-qqp \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
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
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 6 \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False

sleep 5

# Task 7: task591
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_3_llama/$method/llama/outputs/$lr/6-qqp/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/algorithm/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/rte \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/7-rte \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
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
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 7 \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False

sleep 5
#
# Task 8: task363
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_3_llama/$method/llama/outputs/$lr/7-rte/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/imdb \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/8-imdb \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
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
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 8 \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False

sleep 5
#
# Task 9: task1510
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_3_llama/$method/llama/outputs/$lr/8-imdb/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/sst2 \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/9-sst2 \
   --per_device_train_batch_size 16\
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
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
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 9 \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False

sleep 5

# Task 10: task1729
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_3_llama/$method/llama/outputs/$lr/9-sst2/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/dbpedia \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/10-dbpedia \
   --per_device_train_batch_size 16\
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
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
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 10 \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False

sleep 5

 Task 11: task181
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_3_llama/$method/llama/outputs/$lr/10-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/agnews \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/11-agnews \
   --per_device_train_batch_size 16\
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
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
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 11 \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False

sleep 5

# Task 12: task511
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_3_llama/$method/llama/outputs/$lr/11-agnews/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/yahoo \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/12-yahoo \
   --per_device_train_batch_size 16\
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
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
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 12 \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False

sleep 5

# Task 13: task002
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_3_llama/$method/llama/outputs/$lr/12-yahoo/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/multirc \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/13-multirc \
   --per_device_train_batch_size 16\
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
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
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 13 \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False

sleep 5

# Task 14: task1290
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_3_llama/$method/llama/outputs/$lr/13-multirc/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/boolq \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/14-boolq \
   --per_device_train_batch_size 16\
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
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
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 14 \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False

sleep 5

# Task 15: task875
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/Longseq/order_3_llama/$method/llama/outputs/$lr/14-boolq/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_long_order3_t5_configs/wic \
   --output_dir results/Longseq/order_3_llama/$method/llama/outputs/$lr/15-wic \
   --per_device_train_batch_size 16\
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
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
   --logging_steps 2 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 15 \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False