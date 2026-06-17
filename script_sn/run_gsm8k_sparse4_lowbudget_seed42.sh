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

# ===============================================================
# GSM8K low-budget stress run: Tensor-TopK / AB-Factor /
#                              AB-Effective / Ours(SN-P1P2)
# Why this script:
#   budget=2200 is often too loose: AB-Factor / AB-Effective / Ours
#   can all land around ~29 EM. To expose the difference, rerun all
#   sparse methods under tighter budgets.
#
# Usage:
#   cp run_uie_lora_sn_fixed.py src/run_uie_lora.py
#   cp federated_uie_lora_sn_fixed.py src/federated_uie_lora.py
#   bash run_gsm8k_sparse4_lowbudget_seed42.sh
#
# Optional overrides:
#   BUDGETS="440 880" bash run_gsm8k_sparse4_lowbudget_seed42.sh
#   MODES="ours_sn_p1p2" BUDGETS="880" bash run_gsm8k_sparse4_lowbudget_seed42.sh
#   CLIENTS_PER_ROUND=5 BUDGETS="440 880" bash run_gsm8k_sparse4_lowbudget_seed42.sh
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
lr=1e-04
radius=0
seed=${SEED:-42}

# Keep the training protocol consistent; only stress the upload budget by default.
di_alpha=${DIRICHLET_ALPHA:-10}
num_clients=${NUM_CLIENTS:-50}
clients_per_round=${CLIENTS_PER_ROUND:-10}
global_rounds=${GLOBAL_ROUNDS:-5}
local_epochs=${LOCAL_EPOCHS:-10}

# Tighter sparse per-client packet budgets. 2200 was too loose for separation.
BUDGETS_STR=${BUDGETS:-"440 880 1320"}
MODES_STR=${MODES:-"tensor_topk ab_factor ab_effective ours_sn_p1p2"}

# For main performance, force Ours to use the available budget for fair matched-budget comparison.
SN_FORCE_FULL_BUDGET=${SN_FORCE_FULL_BUDGET:-True}
SN_GAP_ETA=${SN_GAP_ETA:-1.0}

# ===== Paths =====
data_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data
task_config_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config
model_path=/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct
ACC_CONFIG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py

OUT_ROOT=results/Qwen2_gsm8k/sparse4_lowbudget_seed${seed}_K${clients_per_round}
mkdir -p ${OUT_ROOT}/logs

run_one() {
    local gpus=$1
    local mode=$2
    local budget=$3

    local port
    port=$(shuf -i25000-30000 -n1)

    local out_dir=${OUT_ROOT}/${mode}_budget${budget}_K${clients_per_round}_seed${seed}
    local run_name=gsm8k_${mode}_budget${budget}_K${clients_per_round}_seed${seed}
    local log_file=${OUT_ROOT}/logs/${run_name}.log

    local score_param=""
    local atomic_param=""
    local sn_params=""

    if [ "${mode}" = "tensor_topk" ]; then
        atomic_param="--upload_atomic_mode tensor"
        score_param="--upload_score_mode factor_norm"
    elif [ "${mode}" = "ab_factor" ]; then
        atomic_param="--upload_atomic_mode ab_pair"
        score_param="--upload_score_mode factor_norm"
    elif [ "${mode}" = "ab_effective" ]; then
        atomic_param="--upload_atomic_mode ab_pair"
        score_param="--upload_score_mode effective_norm"
    elif [ "${mode}" = "ours_sn_p1p2" ]; then
        atomic_param="--upload_atomic_mode ab_pair"
        score_param="--upload_score_mode sn_p1p2"
        sn_params="--sn_gap_eta ${SN_GAP_ETA} --sn_force_full_budget ${SN_FORCE_FULL_BUDGET} --sn_save_diagnostics True"
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

contains_mode() {
    local target=$1
    for m in ${MODES_STR}; do
        if [ "$m" = "$target" ]; then
            return 0
        fi
    done
    return 1
}

for budget in ${BUDGETS_STR}; do
    echo "========== GSM8K low-budget run: budget=${budget}, K=${clients_per_round} =========="

    pids=()
    if contains_mode tensor_topk; then
        run_one 0,1,2,3 tensor_topk ${budget} &
        pids+=("$!")
    fi
    if contains_mode ab_factor; then
        run_one 4,5,6,7 ab_factor ${budget} &
        pids+=("$!")
    fi
    for pid in "${pids[@]:-}"; do wait ${pid}; done

    pids=()
    if contains_mode ab_effective; then
        run_one 0,1,2,3 ab_effective ${budget} &
        pids+=("$!")
    fi
    if contains_mode ours_sn_p1p2; then
        run_one 4,5,6,7 ours_sn_p1p2 ${budget} &
        pids+=("$!")
    fi
    for pid in "${pids[@]:-}"; do wait ${pid}; done

done

echo "All GSM8K low-budget sparse runs finished. Results in: ${OUT_ROOT}"
