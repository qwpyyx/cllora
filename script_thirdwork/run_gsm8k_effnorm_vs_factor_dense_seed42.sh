#!/bin/bash
set -euo pipefail
set -x

export PYTHONWARNINGS="ignore"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/qiuwenqi/.cache/huggingface
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

# ============================================================
# GSM8K: factor-norm vs effective-norm vs dense full-upload
# Purpose:
#   Verify whether the simplest representation-consistent baseline
#   (A/B-pair effective-update-norm selection) actually improves task EM.
#
# Required code version:
#   src/run_uie_lora.py and src/federated_uie_lora.py must support:
#     --upload_score_mode factor_norm/effective_norm
#   If not, replace them with the effective versions first:
#     cp run_uie_lora_effective.py src/run_uie_lora.py
#     cp federated_uie_lora_effective.py src/federated_uie_lora.py
# ============================================================

if ! grep -q "upload_score_mode" src/run_uie_lora.py; then
    echo "[ERROR] src/run_uie_lora.py does not contain upload_score_mode." >&2
    echo "Please replace it with run_uie_lora_effective.py first." >&2
    exit 1
fi

if ! grep -q "effective_norm" src/federated_uie_lora.py; then
    echo "[ERROR] src/federated_uie_lora.py does not contain effective_norm selection." >&2
    echo "Please replace it with federated_uie_lora_effective.py first." >&2
    exit 1
fi

# ===== Basic config =====
method=lora_origin
lora_rank=8
lamda_2=0
lamda_1=0
lr=1e-04
radius=0
seed=42

# GSM8K previous setting. Keep this consistent with your old GSM8K baselines.
di_alpha=10
num_clients=50
clients_per_round=20
global_rounds=5
local_epochs=10

# Sparse budget. Change here only if your previous GSM8K main budget is not 2200.
com_budget=2200

# ===== Paths =====
data_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data
task_config_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config
model_path=/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct
ACC_CONFIG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py

OUT_ROOT=results/Qwen2_gsm8k/effnorm_vs_factor_dense_seed42
mkdir -p ${OUT_ROOT}/logs

run_one() {
    local gpus=$1
    local mode=$2
    local budget=$3

    local port
    port=$(shuf -i25000-30000 -n1)

    local out_dir=${OUT_ROOT}/${mode}_budget${budget}_20client_seed42
    local run_name=gsm8k_${mode}_budget${budget}_20client_seed42
    local log_file=${OUT_ROOT}/logs/${run_name}.log

    local score_param=""
    local atomic_param="--upload_atomic_mode ab_pair"

    if [ "${mode}" = "factor_norm" ]; then
        score_param="--upload_score_mode factor_norm"
    elif [ "${mode}" = "effective_norm" ]; then
        score_param="--upload_score_mode effective_norm"
    elif [ "${mode}" = "dense_full" ]; then
        # dense full upload: budget <= 0 bypasses sparse selection.
        score_param="--upload_score_mode factor_norm"
    else
        echo "Unknown mode: ${mode}" >&2
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=${gpus} accelerate launch --config_file ${ACC_CONFIG} \
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
       --global_rounds ${global_rounds} \
       --local_epochs ${local_epochs} \
       --num_clients ${num_clients} \
       --clients_per_round ${clients_per_round} \
       --dirichlet_alpha ${di_alpha} \
       --partition_strategy quantity \
       --comm_budget ${budget} \
       --learning_rate ${lr} \
       --run_name ${run_name} \
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
       --lamda_1 ${lamda_1} \
       --lamda_2 ${lamda_2} \
       --federated_seed ${seed} \
       --method ${method} \
       --task 1 \
       --radius ${radius} \
       --gradient_checkpointing True \
       --bf16 True \
       --ddp_find_unused_parameters False \
       ${atomic_param} \
       ${score_param} \
       --upload_diversity_mode none \
       --diagnose_pair_saliency False \
       --diagnose_residual_errors False \
       --lora_residual_accumulation False \
       > ${log_file} 2>&1
}

# Run two sparse methods in parallel on 8 GPUs.
run_one 0,1,2,3 factor_norm ${com_budget} &
PID_FACTOR=$!

run_one 4,5,6,7 effective_norm ${com_budget} &
PID_EFFECTIVE=$!

wait ${PID_FACTOR}
wait ${PID_EFFECTIVE}

# Run dense full-upload after the two sparse jobs finish.
# Use GPU 0,1,2,3 by default; change to 4,5,6,7 if needed.
run_one 0,1,2,3 dense_full -1

echo "All GSM8K factor/effective/dense experiments finished. Results in: ${OUT_ROOT}"
