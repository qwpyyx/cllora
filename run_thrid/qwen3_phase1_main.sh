#!/bin/bash
set -euo pipefail
set -x

# ============================================================
# Phase 1: Qwen3-14B + GSM8K Main Comparison Table
#   Dense | ComPEFT | FLM-TopK | FLASC | Raw Ours
#   12.5% budget (1760 packets), 3 seeds (28, 42, 45)
#
# Run via DevContainer: bash script_sn/qwen3_phase1_main.sh
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

# Use qwen3 conda env Python
PY=/home/qiuwenqi/.conda/envs/qwen3/bin/python
ACCELERATE="${PY} -m accelerate.commands.launch"

# ===== Paths =====
MODEL_PATH=/home/qiuwenqi/LLM/models/Qwen3-14B-Instruct
DATA_DIR=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data
TASK_CONFIG_DIR=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config
ACC_CONFIG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py
OUT_ROOT=results/Qwen3_phase1_main
SEEDS="28 42 45"

# ===== FL settings =====
NUM_CLIENTS=50
CLIENTS_PER_ROUND=10
GLOBAL_ROUNDS=5
LOCAL_EPOCHS=10
LR=1e-04
LORA_DIM=8
DIRICHLET_ALPHA=10
PARTITION_STRATEGY=quantity

# Qwen3-14B: full_upload_cost=14080, 12.5%=1760
FULL_COST=14080
MAIN_BUDGET=1760  # 12.5%

# Training
TRAIN_BS=16
EVAL_BS=16
GRAD_ACC=2
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=50
GENERATION_MAX_LENGTH=50
LOGGING_STEPS=2

# GPU parallelism: two 4-GPU jobs at a time
GPU_GROUPS="0,1,2,3 4,5,6,7"
PARALLEL_JOBS=2

mkdir -p ${OUT_ROOT}/logs

# ============================================================
# 1) Method-based baselines: flasc, compeft, flm_topk, fedcomp
# ============================================================
run_method_baseline() {
    local method=$1
    local seed=$2
    local budget=$3
    local gpus=$4
    local port
    port=$(shuf -i25000-30000 -n1)

    local out_dir=${OUT_ROOT}/${method}_budget${budget}_K${CLIENTS_PER_ROUND}_seed${seed}
    local run_name=qwen3_gsm8k_${method}_seed${seed}
    local log_file=${OUT_ROOT}/logs/${run_name}.log

    if [ -f "${out_dir}/all_results.json" ]; then
        local em=$(grep -o '"predict_gsm8k_em": [0-9.]*' "${out_dir}/all_results.json" | head -1)
        echo "[SKIP] ${method} seed${seed} already done: ${em}"
        return 0
    fi

    echo "[RUN] ${method} seed=${seed} budget=${budget} gpus=${gpus} out=${out_dir}"

    CUDA_VISIBLE_DEVICES=${gpus} ${ACCELERATE} \
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
       --per_device_train_batch_size ${TRAIN_BS} \
       --per_device_eval_batch_size ${EVAL_BS} \
       --gradient_accumulation_steps ${GRAD_ACC} \
       --global_rounds ${GLOBAL_ROUNDS} \
       --local_epochs ${LOCAL_EPOCHS} \
       --num_clients ${NUM_CLIENTS} \
       --clients_per_round ${CLIENTS_PER_ROUND} \
       --dirichlet_alpha ${DIRICHLET_ALPHA} \
       --partition_strategy ${PARTITION_STRATEGY} \
       --comm_budget ${budget} \
       --learning_rate ${LR} \
       --run_name ${run_name} \
       --max_source_length ${MAX_SOURCE_LENGTH} \
       --max_target_length ${MAX_TARGET_LENGTH} \
       --generation_max_length ${GENERATION_MAX_LENGTH} \
       --add_task_name False \
       --add_dataset_name False \
       --overwrite_output_dir \
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps ${LOGGING_STEPS} \
       --evaluation_strategy no \
       --save_strategy no \
       --save_steps 150 \
       --lamda_1 0 \
       --lamda_2 0 \
       --federated_seed ${seed} \
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
    local em="N/A"
    if [ -f "${out_dir}/all_results.json" ]; then
        em=$(grep -o '"predict_gsm8k_em": [0-9.]*' "${out_dir}/all_results.json" | head -1)
    fi
    echo "[DONE] ${method} seed${seed} exit=${ec} ${em}"
}

# ============================================================
# 2) Dense full upload
# ============================================================
run_dense() {
    local seed=$1
    local gpus=$2
    local port
    port=$(shuf -i25000-30000 -n1)

    local out_dir=${OUT_ROOT}/dense_full_K${CLIENTS_PER_ROUND}_seed${seed}
    local run_name=qwen3_gsm8k_dense_seed${seed}
    local log_file=${OUT_ROOT}/logs/${run_name}.log

    if [ -f "${out_dir}/all_results.json" ]; then
        local em=$(grep -o '"predict_gsm8k_em": [0-9.]*' "${out_dir}/all_results.json" | head -1)
        echo "[SKIP] dense seed${seed} already done: ${em}"
        return 0
    fi

    echo "[RUN] dense seed=${seed} gpus=${gpus} out=${out_dir}"

    CUDA_VISIBLE_DEVICES=${gpus} ${ACCELERATE} \
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
       --per_device_train_batch_size ${TRAIN_BS} \
       --per_device_eval_batch_size ${EVAL_BS} \
       --gradient_accumulation_steps ${GRAD_ACC} \
       --global_rounds ${GLOBAL_ROUNDS} \
       --local_epochs ${LOCAL_EPOCHS} \
       --num_clients ${NUM_CLIENTS} \
       --clients_per_round ${CLIENTS_PER_ROUND} \
       --dirichlet_alpha ${DIRICHLET_ALPHA} \
       --partition_strategy ${PARTITION_STRATEGY} \
       --comm_budget 0 \
       --learning_rate ${LR} \
       --run_name ${run_name} \
       --max_source_length ${MAX_SOURCE_LENGTH} \
       --max_target_length ${MAX_TARGET_LENGTH} \
       --generation_max_length ${GENERATION_MAX_LENGTH} \
       --add_task_name False \
       --add_dataset_name False \
       --overwrite_output_dir \
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps ${LOGGING_STEPS} \
       --evaluation_strategy no \
       --save_strategy no \
       --save_steps 150 \
       --lamda_1 0 \
       --lamda_2 0 \
       --federated_seed ${seed} \
       --method lora_origin \
       --task 1 \
       --radius 0 \
       --gradient_checkpointing True \
       --bf16 True \
       --ddp_find_unused_parameters False \
       --upload_diversity_mode none \
       --diagnose_pair_saliency False \
       --diagnose_residual_errors False \
       --lora_residual_accumulation False \
       > ${log_file} 2>&1

    local ec=$?
    local em="N/A"
    if [ -f "${out_dir}/all_results.json" ]; then
        em=$(grep -o '"predict_gsm8k_em": [0-9.]*' "${out_dir}/all_results.json" | head -1)
    fi
    echo "[DONE] dense seed${seed} exit=${ec} ${em}"
}

# ============================================================
# 3) Raw Ours: SN-P1/P2 + qv-block
# ============================================================
run_ours() {
    local seed=$1
    local budget=$2
    local gpus=$3
    local port
    port=$(shuf -i25000-30000 -n1)

    local out_dir=${OUT_ROOT}/ours_sn_p1p2_budget${budget}_K${CLIENTS_PER_ROUND}_seed${seed}
    local run_name=qwen3_gsm8k_ours_seed${seed}
    local log_file=${OUT_ROOT}/logs/${run_name}.log

    if [ -f "${out_dir}/all_results.json" ]; then
        local em=$(grep -o '"predict_gsm8k_em": [0-9.]*' "${out_dir}/all_results.json" | head -1)
        echo "[SKIP] ours seed${seed} already done: ${em}"
        return 0
    fi

    echo "[RUN] ours seed=${seed} budget=${budget} gpus=${gpus} out=${out_dir}"

    CUDA_VISIBLE_DEVICES=${gpus} ${ACCELERATE} \
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
       --per_device_train_batch_size ${TRAIN_BS} \
       --per_device_eval_batch_size ${EVAL_BS} \
       --gradient_accumulation_steps ${GRAD_ACC} \
       --global_rounds ${GLOBAL_ROUNDS} \
       --local_epochs ${LOCAL_EPOCHS} \
       --num_clients ${NUM_CLIENTS} \
       --clients_per_round ${CLIENTS_PER_ROUND} \
       --dirichlet_alpha ${DIRICHLET_ALPHA} \
       --partition_strategy ${PARTITION_STRATEGY} \
       --comm_budget ${budget} \
       --learning_rate ${LR} \
       --run_name ${run_name} \
       --max_source_length ${MAX_SOURCE_LENGTH} \
       --max_target_length ${MAX_TARGET_LENGTH} \
       --generation_max_length ${GENERATION_MAX_LENGTH} \
       --add_task_name False \
       --add_dataset_name False \
       --overwrite_output_dir \
       --overwrite_cache \
       --lr_scheduler_type constant \
       --warmup_steps 0 \
       --logging_strategy steps \
       --logging_steps ${LOGGING_STEPS} \
       --evaluation_strategy no \
       --save_strategy no \
       --save_steps 150 \
       --lamda_1 0 \
       --lamda_2 0 \
       --federated_seed ${seed} \
       --method lora_origin \
       --task 1 \
       --radius 0 \
       --gradient_checkpointing True \
       --bf16 True \
       --ddp_find_unused_parameters False \
       --upload_atomic_mode qv_block \
       --upload_score_mode sn_p1p2 \
       --sn_p1_norm_mode depth_balanced \
       --sn_depth_group_ratios 1,1,2 \
       --sn_gap_eta 1.0 \
       --sn_force_full_budget False \
       --sn_save_diagnostics True \
       --upload_diversity_mode none \
       --diagnose_pair_saliency False \
       --diagnose_residual_errors False \
       --lora_residual_accumulation False \
       > ${log_file} 2>&1

    local ec=$?
    local em="N/A"
    if [ -f "${out_dir}/all_results.json" ]; then
        em=$(grep -o '"predict_gsm8k_em": [0-9.]*' "${out_dir}/all_results.json" | head -1)
    fi
    echo "[DONE] ours seed${seed} exit=${ec} ${em}"
}

# ============================================================
# Execution Plan
# ============================================================

echo "============================================"
echo " Phase 1: Qwen3-14B + GSM8K Main Table"
echo " Model: ${MODEL_PATH}"
echo " Budget: ${MAIN_BUDGET} (12.5% of ${FULL_COST})"
echo " Seeds: ${SEEDS}"
echo " Methods: Dense + ComPEFT + FLM-TopK + FLASC + Ours"
echo "============================================"

# Run in pairs: each job uses 4 GPUs, two jobs in parallel
# Pair 1: flasc + compeft, all 3 seeds each
for seed in ${SEEDS}; do
    run_method_baseline flasc ${seed} ${MAIN_BUDGET} "0,1,2,3" &
    PID1=$!
    run_method_baseline compeft ${seed} ${MAIN_BUDGET} "4,5,6,7" &
    PID2=$!
    wait ${PID1} ${PID2}
done

# Pair 2: flm_topk + fedcomp, all 3 seeds each
for seed in ${SEEDS}; do
    run_method_baseline flm_topk ${seed} ${MAIN_BUDGET} "0,1,2,3" &
    PID1=$!
    run_method_baseline fedcomp ${seed} ${MAIN_BUDGET} "4,5,6,7" &
    PID2=$!
    wait ${PID1} ${PID2}
done

# Pair 3: Dense + Ours (ours heavier, gives it 4 GPUs, dense 4 GPUs)
for seed in ${SEEDS}; do
    run_dense ${seed} "0,1,2,3" &
    PID1=$!
    run_ours ${seed} ${MAIN_BUDGET} "4,5,6,7" &
    PID2=$!
    wait ${PID1} ${PID2}
done

# ===== Summary =====
echo ""
echo "============================================"
echo " Phase 1 Done! Results: ${OUT_ROOT}"
echo "============================================"
for method in dense flasc compeft flm_topk fedcomp ours; do
    for seed in ${SEEDS}; do
        if [ "${method}" = "dense" ]; then
            d=${OUT_ROOT}/dense_full_K${CLIENTS_PER_ROUND}_seed${seed}
        elif [ "${method}" = "ours" ]; then
            d=${OUT_ROOT}/ours_sn_p1p2_budget${MAIN_BUDGET}_K${CLIENTS_PER_ROUND}_seed${seed}
        else
            d=${OUT_ROOT}/${method}_budget${MAIN_BUDGET}_K${CLIENTS_PER_ROUND}_seed${seed}
        fi
        if [ -f "${d}/all_results.json" ]; then
            em=$(grep -o '"predict_gsm8k_em": [0-9.]*' "${d}/all_results.json")
            echo "  ${method} seed${seed}: ${em}"
        else
            echo "  ${method} seed${seed}: MISSING"
        fi
    done
done
