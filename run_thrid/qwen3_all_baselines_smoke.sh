#!/bin/bash
set -euo pipefail
set -x

# ============================================================
# Qwen3-14B All-Baselines Smoke Test
# Purpose: verify every baseline code path works with Qwen3
# Lightweight: 1 round, 1 epoch, 10 clients, 2 per round
#
# 7 baselines + our method:
#   Method-based:  flasc, compeft, flm_topk, fedcomp
#   Upload-mode:   tensor_topk, ab_factor, ours_sn_p1p2
#
# Run in project root via DevContainer
# ============================================================

export PYTHONWARNINGS="ignore"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/qiuwenqi/.cache/huggingface
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

# ===== Paths =====
PY=/home/qiuwenqi/.conda/envs/qwen3/bin/python
MODEL_PATH=/home/qiuwenqi/LLM/models/Qwen3-14B-Instruct
DATA_DIR=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data
TASK_CONFIG_DIR=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config
ACC_CONFIG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py
OUT_ROOT=results/Qwen3_all_baselines_smoke
GPUS=0,1,2,3

# ===== Super lightweight =====
SEED=42
NUM_CLIENTS=10
CLIENTS_PER_ROUND=2
GLOBAL_ROUNDS=1
LOCAL_EPOCHS=1
LR=1e-04
LORA_DIM=8
BUDGET=440

mkdir -p ${OUT_ROOT}/logs

echo "============================================"
echo "  Qwen3 All-Baselines Smoke Test"
echo "  Python: $(${PY} --version)"
echo "  Model: ${MODEL_PATH}"
echo "  Settings: ${GLOBAL_ROUNDS}r x ${LOCAL_EPOCHS}e, ${NUM_CLIENTS} clients, ${CLIENTS_PER_ROUND}/round"
echo "============================================"

# ============================================================
# 1) Method-based baselines: flasc, compeft, flm_topk, fedcomp
# ============================================================
run_method_baseline() {
    local method=$1
    local port
    port=$(shuf -i25000-30000 -n1)

    local out_dir=${OUT_ROOT}/${method}_seed${SEED}
    local log_file=${OUT_ROOT}/logs/${method}.log

    echo "=== [${method}] Starting on GPUs ${GPUS} ==="
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} -m accelerate.commands.launch \
       --config_file ${ACC_CONFIG} \
       --main_process_port ${port} \
       ${PY_SCRIPT} \
       --report_to none \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --lora_dim ${LORA_DIM} \
       --model_name_or_path ${MODEL_PATH} \
       --data_dir ${DATA_DIR} \
       --task_config_dir ${TASK_CONFIG_DIR} \
       --output_dir ${out_dir} \
       --per_device_train_batch_size 2 \
       --per_device_eval_batch_size 8 \
       --gradient_accumulation_steps 2 \
       --global_rounds ${GLOBAL_ROUNDS} \
       --local_epochs ${LOCAL_EPOCHS} \
       --num_clients ${NUM_CLIENTS} \
       --clients_per_round ${CLIENTS_PER_ROUND} \
       --dirichlet_alpha 10 \
       --partition_strategy quantity \
       --comm_budget ${BUDGET} \
       --learning_rate ${LR} \
       --run_name qwen3_smoke_${method} \
       --max_source_length 512 \
       --max_target_length 16 \
       --generation_max_length 16 \
       --add_task_name False \
       --add_dataset_name False \
       --overwrite_output_dir \
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 1 \
       --evaluation_strategy no \
       --save_strategy no \
       --save_steps 150 \
       --lamda_1 0 \
       --lamda_2 0 \
       --federated_seed ${SEED} \
       --method ${method} \
       --task 1 \
       --radius 0 \
       --gradient_checkpointing True \
       --bf16 True \
       --ddp_find_unused_parameters False \
       --baseline_packet_num 0 \
       --baseline_blocks 192 \
       --baseline_bit 18 \
       --baseline_min_bit 4 \
       --baseline_topk_method gradient \
       --fedcomp_use_residual True \
       --upload_diversity_mode none \
       --diagnose_pair_saliency False \
       --diagnose_residual_errors False \
       --lora_residual_accumulation False \
       > ${log_file} 2>&1

    local ec=$?
    if [ ${ec} -eq 0 ]; then
        echo "=== [${method}] PASSED ==="
    else
        echo "=== [${method}] FAILED (exit ${ec}) - check ${log_file} ==="
    fi
    return ${ec}
}

# ============================================================
# 2) Upload-mode baselines: tensor_topk, ab_factor, ours_sn_p1p2
# ============================================================
run_upload_mode() {
    local mode=$1
    local port
    port=$(shuf -i25000-30000 -n1)

    local out_dir=${OUT_ROOT}/${mode}_seed${SEED}
    local log_file=${OUT_ROOT}/logs/${mode}.log

    local atomic_param=""
    local score_param=""
    local sn_params=""

    if [ "${mode}" = "tensor_topk" ]; then
        atomic_param="--upload_atomic_mode tensor"
        score_param="--upload_score_mode factor_norm"
    elif [ "${mode}" = "ab_factor" ]; then
        atomic_param="--upload_atomic_mode ab_pair"
        score_param="--upload_score_mode factor_norm"
    elif [ "${mode}" = "ours_sn_p1p2" ]; then
        atomic_param="--upload_atomic_mode ab_pair"
        score_param="--upload_score_mode sn_p1p2"
        sn_params="--sn_gap_eta 1.0 --sn_force_full_budget False --sn_save_diagnostics True"
    else
        echo "Unknown mode: ${mode}" >&2
        return 1
    fi

    echo "=== [${mode}] Starting on GPUs ${GPUS} ==="
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} -m accelerate.commands.launch \
       --config_file ${ACC_CONFIG} \
       --main_process_port ${port} \
       ${PY_SCRIPT} \
       --report_to none \
       --do_train \
       --do_predict \
       --predict_with_generate \
       --lora_dim ${LORA_DIM} \
       --model_name_or_path ${MODEL_PATH} \
       --data_dir ${DATA_DIR} \
       --task_config_dir ${TASK_CONFIG_DIR} \
       --output_dir ${out_dir} \
       --per_device_train_batch_size 2 \
       --per_device_eval_batch_size 8 \
       --gradient_accumulation_steps 2 \
       --global_rounds ${GLOBAL_ROUNDS} \
       --local_epochs ${LOCAL_EPOCHS} \
       --num_clients ${NUM_CLIENTS} \
       --clients_per_round ${CLIENTS_PER_ROUND} \
       --dirichlet_alpha 10 \
       --partition_strategy quantity \
       --comm_budget ${BUDGET} \
       --learning_rate ${LR} \
       --run_name qwen3_smoke_${mode} \
       --max_source_length 512 \
       --max_target_length 16 \
       --generation_max_length 16 \
       --add_task_name False \
       --add_dataset_name False \
       --overwrite_output_dir \
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps 1 \
       --evaluation_strategy no \
       --save_strategy no \
       --save_steps 150 \
       --lamda_1 0 \
       --lamda_2 0 \
       --federated_seed ${SEED} \
       --method lora_origin \
       --task 1 \
       --radius 0 \
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

    local ec=$?
    if [ ${ec} -eq 0 ]; then
        echo "=== [${mode}] PASSED ==="
    else
        echo "=== [${mode}] FAILED (exit ${ec}) - check ${log_file} ==="
    fi
    return ${ec}
}

# ============================================================
# Run all baselines sequentially
# ============================================================

PASSED=()
FAILED=()

run_and_record() {
    local label=$1
    shift
    if "$@"; then
        PASSED+=("${label}")
    else
        FAILED+=("${label}")
    fi
    echo ""
}

echo ""
echo "=== PHASE 1: Method-based baselines ==="
run_and_record "flasc"     run_method_baseline flasc
run_and_record "compeft"   run_method_baseline compeft
run_and_record "flm_topk"  run_method_baseline flm_topk
run_and_record "fedcomp"   run_method_baseline fedcomp

echo ""
echo "=== PHASE 2: Upload-mode baselines ==="
run_and_record "tensor_topk"    run_upload_mode tensor_topk
run_and_record "ab_factor"      run_upload_mode ab_factor
run_and_record "ours_sn_p1p2"   run_upload_mode ours_sn_p1p2

echo ""
echo "============================================"
echo "  Qwen3 All-Baselines Smoke Test Summary"
echo "============================================"
echo "PASSED (${#PASSED[@]}):"
for p in "${PASSED[@]}"; do echo "  ✅ ${p}"; done
echo ""
echo "FAILED (${#FAILED[@]}):"
for f in "${FAILED[@]}"; do echo "  ❌ ${f}"; done
echo "============================================"
echo "Logs: ${OUT_ROOT}/logs/"
echo "============================================"

if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
