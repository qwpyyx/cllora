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
method=lora_origin
lora_rank=8
lamda_2=0
lamda_1=0
# bash scripts/order_1_adaptive.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path /home/qiuwenqi/LLM/models/t5-large \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task748_glucose_reverse_cause_event_detection \
   --output_dir results/SuperNI/order_2/$method/outputs/1-task748_glucose_reverse_cause_event_detection \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
   --run_name order2_round1 \
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
   --save_steps 150 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 1
sleep 5

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2/$method/outputs/1-task748_glucose_reverse_cause_event_detection/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task073_commonsenseqa_answer_generation \
   --output_dir results/SuperNI/order_2/$method/outputs/2-task073_commonsenseqa_answer_generation \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
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
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 $lamda_1 \
   --lamda_2 $lamda_2 \
   --federated_seed 42 \
   --method $method \
   --task 2

sleep 5

# Task 3: task1590
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2/$method/outputs/2-task073_commonsenseqa_answer_generation/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task1590_diplomacy_text_generation \
   --output_dir results/SuperNI/order_2/$method/outputs/3-task1590_diplomacy_text_generation \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
   --run_name order2_round3 \
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
   --method $method \
   --task 3
sleep 5

# Task 4: task639
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2/$method/outputs/3-task1590_diplomacy_text_generation/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task639_multi_woz_user_utterance_generation \
   --output_dir results/SuperNI/order_2/$method/outputs/4-task639_multi_woz_user_utterance_generation \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
   --run_name order2_round4 \
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
   --method $method \
   --task 4
sleep 5

# Task 5: task1572
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2/$method/outputs/4-task639_multi_woz_user_utterance_generation/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task1572_samsum_summary \
   --output_dir results/SuperNI/order_2/$method/outputs/5-task1572_samsum_summary \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
   --run_name order2_round5 \
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
   --method $method \
   --task 5
sleep 5

# Task 6: task1687
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2/$method/outputs/5-task1572_samsum_summary/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task1687_sentiment140_classification \
   --output_dir results/SuperNI/order_2/$method/outputs/6-task1687_sentiment140_classification \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
   --run_name order2_round6 \
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
   --method $method \
   --task 6
sleep 5

 Task 7: task591
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2/$method/outputs/6-task1687_sentiment140_classification/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task591_sciq_answer_generation \
   --output_dir results/SuperNI/order_2/$method/outputs/7-task591_sciq_answer_generation \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
   --run_name order2_round7 \
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
   --method $method \
   --task 7
sleep 5

# Task 8: task363
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2/$method/outputs/7-task591_sciq_answer_generation/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task363_sst2_polarity_classification \
   --output_dir results/SuperNI/order_2/$method/outputs/8-task363_sst2_polarity_classification \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
   --run_name order2_round8 \
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
   --method $method \
   --task 8
sleep 5

# Task 9: task1510
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2/$method/outputs/8-task363_sst2_polarity_classification/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task1510_evalution_relation_extraction \
   --output_dir results/SuperNI/order_2/$method/outputs/9-task1510_evalution_relation_extraction \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
   --run_name order2_round9 \
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
   --method $method \
   --task 9
sleep 5

# Task 10: task1729
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2/$method/outputs/9-task1510_evalution_relation_extraction/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task1729_personachat_generate_next \
   --output_dir results/SuperNI/order_2/$method/outputs/10-task1729_personachat_generate_next \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
   --run_name order2_round10 \
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
   --method $method \
   --task 10
sleep 5

# Task 11: task181
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2/$method/outputs/10-task1729_personachat_generate_next/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task181_outcome_extraction \
   --output_dir results/SuperNI/order_2/$method/outputs/11-task181_outcome_extraction \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
   --run_name order2_round11 \
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
   --method $method \
   --task 11
sleep 5

# Task 12: task511
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2/$method/outputs/11-task181_outcome_extraction/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task511_reddit_tifu_long_text_summarization \
   --output_dir results/SuperNI/order_2/$method/outputs/12-task511_reddit_tifu_long_text_summarization \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
   --run_name order2_round12 \
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
   --method $method \
   --task 12
sleep 5

# Task 13: task002
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2/$method/outputs/12-task511_reddit_tifu_long_text_summarization/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task002_quoref_answer_generation \
   --output_dir results/SuperNI/order_2/$method/outputs/13-task002_quoref_answer_generation \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
   --run_name order2_round13 \
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
   --method $method \
   --task 13
sleep 5

# Task 14: task1290
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2/$method/outputs/13-task002_quoref_answer_generation/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task1290_xsum_summarization \
   --output_dir results/SuperNI/order_2/$method/outputs/14-task1290_xsum_summarization \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
   --run_name order2_round14 \
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
   --method $method \
   --task 14
sleep 5

# Task 15: task875
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file script_accelerate/accelerate_config.yaml\
   --main_process_port $port \
   src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path results/SuperNI/order_2/$method/outputs/14-task1290_xsum_summarization/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir /home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/configs/SuperniAndLongseq/gen_script_superni_order2_t5_configs/task875_emotion_classification \
   --output_dir results/SuperNI/order_2/$method/outputs/15-task875_emotion_classification \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --global_rounds 5 \
   --local_epochs 20 \
   --dirichlet_alpha 10 \
   --comm_budget 300 \
   --learning_rate 3e-04 \
   --run_name order2_round15 \
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
   --method $method \
   --task 15