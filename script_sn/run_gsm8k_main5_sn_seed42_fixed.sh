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
# GSM8K main results: Dense / Tensor-TopK / AB-Factor /
#                     AB-Effective / Ours(SN-P1P2)
#
# Before running, copy the modified Python files into src/:
#   cp run_uie_lora_sn.py src/run_uie_lora.py
#   cp federated_uie_lora_sn.py src/federated_uie_lora.py
# ============================================================

if ! grep -q "sn_p1p2" src/run_uie_lora.py; then
    echo "[ERROR] src/run_uie_lora.py does not contain sn_p1p2 arguments." >&2
    echo "Please copy run_uie_lora_sn.py to src/run_uie_lora.py first." >&2
    exit 1
fi

if ! grep -q "_run_signal_noise_p1_p2_schedule" src/federated_uie_lora.py; then
    echo "[ERROR] src/federated_uie_lora.py does not contain SN-P1P2 scheduler." >&2
    echo "Please copy federated_uie_lora_sn.py to src/federated_uie_lora.py first." >&2
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

# GSM8K setting, kept consistent with previous runs.
di_alpha=10
num_clients=50
clients_per_round=20
global_rounds=5
local_epochs=10

# Sparse per-client packet budget.
com_budget=2200

# ===== Paths =====
data_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data
task_config_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config
model_path=/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct
ACC_CONFIG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py

OUT_ROOT=results/Qwen2_gsm8k/main5_sn_seed42
mkdir -p ${OUT_ROOT}/logs

run_one() {
    local gpus=$1
    local mode=$2
    local budget=$3

    local port
    port=$(shuf -i25000-30000 -n1)

    local budget_label=${budget}
    if [ "${budget}" = "-1" ] || [ "${budget}" = "0" ]; then
        budget_label=full
    fi

    local out_dir=${OUT_ROOT}/${mode}_budget${budget_label}_K${clients_per_round}_seed${seed}
    local run_name=gsm8k_${mode}_budget${budget_label}_K${clients_per_round}_seed${seed}
    local log_file=${OUT_ROOT}/logs/${run_name}.log

    local score_param=""
    local atomic_param=""
    local sn_params=""

    if [ "${mode}" = "dense_full" ]; then
        # Dense full upload: comm_budget <= 0 bypasses sparse selection.
        atomic_param="--upload_atomic_mode ab_pair"
        score_param="--upload_score_mode factor_norm"
    elif [ "${mode}" = "tensor_topk" ]; then
        # Original tensor-level Top-K baseline.
        atomic_param="--upload_atomic_mode tensor"
        score_param="--upload_score_mode factor_norm"
    elif [ "${mode}" = "ab_factor" ]; then
        # A/B-pair atomic upload ranked by factor-space norm.
        atomic_param="--upload_atomic_mode ab_pair"
        score_param="--upload_score_mode factor_norm"
    elif [ "${mode}" = "ab_effective" ]; then
        # A/B-pair atomic upload ranked by effective-update norm.
        atomic_param="--upload_atomic_mode ab_pair"
        score_param="--upload_score_mode effective_norm"
    elif [ "${mode}" = "ours_sn_p1p2" ]; then
        # Ours: server-side signal-noise P1 allocation + gap-aware P2 assignment.
        # Current code uses exact low-rank effective-update statistics for the main run;
        # sketch-vs-exact can be evaluated later as an overhead ablation.
        atomic_param="--upload_atomic_mode ab_pair"
        score_param="--upload_score_mode sn_p1p2"
        sn_params="--sn_gap_eta 1.0 --sn_force_full_budget False --sn_save_diagnostics True"
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
       ${sn_params} \
       --upload_diversity_mode none \
       --diagnose_pair_saliency False \
       --diagnose_residual_errors False \
       --lora_residual_accumulation False \
       > ${log_file} 2>&1
}

# Run two jobs at a time on 8 GPUs, each job using 4 GPUs.
run_one 0,1,2,3 tensor_topk ${com_budget} &
PID_TENSOR=$!
run_one 4,5,6,7 ab_factor ${com_budget} &
PID_FACTOR=$!
wait ${PID_TENSOR}
wait ${PID_FACTOR}

run_one 0,1,2,3 ab_effective ${com_budget} &
PID_EFFECTIVE=$!
run_one 4,5,6,7 ours_sn_p1p2 ${com_budget} &
PID_OURS=$!
wait ${PID_EFFECTIVE}
wait ${PID_OURS}

# Dense full-upload reference. Run after sparse jobs finish.
run_one 0,1,2,3 dense_full -1

echo "All GSM8K main-five experiments finished. Results in: ${OUT_ROOT}"
