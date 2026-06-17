#!/bin/bash
set -euo pipefail
set -x

export PYTHONWARNINGS="ignore"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/qiuwenqi/.cache/huggingface
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

# =============================================================
# GSM8K Ours-only diagnostic sweep for SN-P1P2.
# Purpose:
#   Diagnose why Ours may underperform AB-Factor / AB-Effective.
#   This script runs only Ours under several gap-aware strengths,
#   budgets, and client participation settings.
#
# Default sweep:
#   K_LIST="10"
#   BUDGETS="880 1320 1760"
#   GAP_ETAS="0.0 0.1 1.0"
#
# Recommended first run:
#   bash run_gsm8k_ours_sn_sweep_diagnosis_seed42.sh
#
# Optional broader run:
#   K_LIST="5 10" BUDGETS="880 1320" GAP_ETAS="0.0 0.1 1.0" \
#     bash run_gsm8k_ours_sn_sweep_diagnosis_seed42.sh
#
# Optional plain P1+P2-L only:
#   GAP_ETAS="0.0" BUDGETS="880 1320" CLIENTS_PER_ROUND_LIST="10" \
#     bash run_gsm8k_ours_sn_sweep_diagnosis_seed42.sh
#
# Before running, copy the fixed Python files into src/:
#   cp run_uie_lora_sn_fixed.py src/run_uie_lora.py
#   cp federated_uie_lora_sn_fixed.py src/federated_uie_lora.py
# ============================================================

if ! grep -q "sn_p1p2" src/run_uie_lora.py; then
    echo "[ERROR] src/run_uie_lora.py does not contain sn_p1p2 arguments." >&2
    echo "Please copy run_uie_lora_sn_fixed.py to src/run_uie_lora.py first." >&2
    exit 1
fi

if ! grep -q "greedy_generalized_assignment_heterogeneous_cost" src/federated_uie_lora.py; then
    echo "[ERROR] src/federated_uie_lora.py does not contain the heterogeneous-cost SN-P1P2 fix." >&2
    echo "Please copy federated_uie_lora_sn_fixed.py to src/federated_uie_lora.py first." >&2
    exit 1
fi

# ===== Basic config =====
method=lora_origin
lora_rank=8
lamda_2=0
lamda_1=0
lr=${LR:-1e-04}
radius=0
seed=${SEED:-42}

di_alpha=${DIRICHLET_ALPHA:-10}
num_clients=${NUM_CLIENTS:-50}
global_rounds=${GLOBAL_ROUNDS:-5}
local_epochs=${LOCAL_EPOCHS:-10}

# Diagnostic sweep knobs.
# K_LIST and CLIENTS_PER_ROUND_LIST are aliases; K_LIST has priority.
K_LIST_STR=${K_LIST:-${CLIENTS_PER_ROUND_LIST:-"10"}}
BUDGETS_STR=${BUDGETS:-"880 1320 1760"}
GAP_ETAS_STR=${GAP_ETAS:-"0.0 0.1 1.0"}
SN_FORCE_FULL_BUDGET=${SN_FORCE_FULL_BUDGET:-True}

# Run two jobs in parallel if 8 GPUs are available.
PARALLEL_JOBS=${PARALLEL_JOBS:-2}
GPU_GROUP_0=${GPU_GROUP_0:-0,1,2,3}
GPU_GROUP_1=${GPU_GROUP_1:-4,5,6,7}

# ===== Paths =====
data_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data
task_config_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config
model_path=/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct
ACC_CONFIG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py

OUT_ROOT=results/Qwen2_gsm8k/ours_sn_sweep_seed${seed}
mkdir -p ${OUT_ROOT}/logs

sanitize_eta() {
    echo "$1" | sed 's/\./p/g'
}

run_one() {
    local gpus=$1
    local clients_per_round=$2
    local budget=$3
    local gap_eta=$4

    local eta_tag
    eta_tag=$(sanitize_eta ${gap_eta})

    local port
    port=$(shuf -i25000-30000 -n1)

    local mode=ours_sn_p1p2_gap${eta_tag}
    local out_dir=${OUT_ROOT}/${mode}_budget${budget}_K${clients_per_round}_seed${seed}
    local run_name=gsm8k_${mode}_budget${budget}_K${clients_per_round}_seed${seed}
    local log_file=${OUT_ROOT}/logs/${run_name}.log

    echo "[RUN] K=${clients_per_round}, budget=${budget}, gap_eta=${gap_eta}, gpus=${gpus}"

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
       --upload_atomic_mode ab_pair \
       --upload_score_mode sn_p1p2 \
       --sn_gap_eta ${gap_eta} \
       --sn_force_full_budget ${SN_FORCE_FULL_BUDGET} \
       --sn_save_diagnostics True \
       --upload_diversity_mode none \
       --diagnose_pair_saliency False \
       --diagnose_residual_errors False \
       --lora_residual_accumulation False \
       > ${log_file} 2>&1

    echo "[DONE] ${run_name}"
    echo "       out: ${out_dir}"
    echo "       log: ${log_file}"
}

wait_for_batch() {
    local -n _pids=$1
    for pid in "${_pids[@]:-}"; do
        wait ${pid}
    done
    _pids=()
}

pids=()
slot=0
for clients_per_round in ${K_LIST_STR}; do
    for budget in ${BUDGETS_STR}; do
        for gap_eta in ${GAP_ETAS_STR}; do
            if [ "${PARALLEL_JOBS}" -ge 2 ]; then
                if [ $((slot % 2)) -eq 0 ]; then
                    run_one ${GPU_GROUP_0} ${clients_per_round} ${budget} ${gap_eta} &
                else
                    run_one ${GPU_GROUP_1} ${clients_per_round} ${budget} ${gap_eta} &
                fi
                pids+=("$!")
                slot=$((slot + 1))
                if [ "${#pids[@]}" -ge 2 ]; then
                    wait_for_batch pids
                fi
            else
                run_one ${GPU_GROUP_0} ${clients_per_round} ${budget} ${gap_eta}
            fi
        done
    done
done

if [ "${#pids[@]}" -gt 0 ]; then
    wait_for_batch pids
fi

echo "All Ours SN-P1P2 diagnostic runs finished."
echo "Results root: ${OUT_ROOT}"
echo "Logs: ${OUT_ROOT}/logs"
echo "Check each run's signal_noise_schedule_history.json for budget usage, unfilled quotas, and selected projection types."
