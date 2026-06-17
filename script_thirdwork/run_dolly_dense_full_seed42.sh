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

# ===== Basic config =====
method=lora_origin
lora_rank=8
lamda_2=0
lamda_1=0
lr=1e-04
radius=0
di_alpha=0.5
seed=42

# Dense full upload: set comm_budget <= 0 so ahe code bypasses sparse Top-K selection.
# This keeps clients_per_round=5, so it is the full-upload upper-bound baseline,
# not a communication-matched dense-reduced-participation baseline.
com_budget=-1

GPU_DEVICES=0,1,2,3

# ===== Paths =====
data_dir=/home/qiuwenqi/LLM/Third_work/data_dolly/dolly_uie/data
task_config_dir=/home/qiuwenqi/LLM/Third_work/data_dolly/dolly_uie/task_config
model_path=/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct
ACC_CONFIG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py

OUT_ROOT=results/Qwen2_dolly/lora_exp_dolly_alpha05/dense_full_seed42
out_dir=${OUT_ROOT}/dense_full_upload_5client
run_name=dolly_dense_full_upload_seed42_5client
mkdir -p ${OUT_ROOT}/logs
log_file=${OUT_ROOT}/logs/${run_name}.log

port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} accelerate launch --config_file ${ACC_CONFIG} \
   --main_process_port ${port} \
   ${PY_SCRIPT} \
   --report_to none \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim ${lora_rank} \
   --model_name_or_path ${model_path} \
   --data_dir ${data_dir} \
   --task_config_dir ${task_config_dir} \
   --output_dir ${out_dir} \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --global_rounds 5 \
   --local_epochs 10 \
   --num_clients 50 \
   --clients_per_round 5 \
   --dirichlet_alpha ${di_alpha} \
   --partition_strategy label \
   --partition_label_key Dataset \
   --comm_budget ${com_budget} \
   --learning_rate ${lr} \
   --run_name ${run_name} \
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
   --lamda_1 ${lamda_1} \
   --lamda_2 ${lamda_2} \
   --federated_seed ${seed} \
   --method ${method} \
   --task 1 \
   --radius ${radius} \
   --gradient_checkpointing True \
   --bf16 True \
   --ddp_find_unused_parameters False \
   --upload_atomic_mode ab_pair \
   --upload_diversity_mode none \
   --diagnose_pair_saliency False \
   --diagnose_residual_errors False \
   --lora_residual_accumulation False \
   2>&1 | tee ${log_file}

echo "Dense full-upload Dolly baseline finished. Output: ${out_dir}"
