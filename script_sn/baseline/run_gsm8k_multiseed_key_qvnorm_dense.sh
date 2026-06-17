#!/bin/bash
set -euo pipefail
set -x

export PYTHONWARNINGS="ignore"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/qiuwenqi/.cache/huggingface
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

# ==============================================================
# GSM8K multi-seed key experiment + Dense reference
# Default purpose:
#   1) Multi-seed matched comparison at the strongest budget point.
#   2) Add Dense full-upload reference under the same K/seeds.
#
# Recommended code version:
#   cp run_uie_lora_sn_qvblock_norm.py src/run_uie_lora.py
#   cp federated_uie_lora_sn_qvblock_norm.py src/federated_uie_lora.py
#   bash run_gsm8k_multiseed_key_qvnorm_dense.sh
#
# Default runs:
#   SEEDS="42 45 28"
#   BUDGETS="2112"
#   SPARSE_MODES="ab_factor ab_effective ours_depth_balanced"
#   RUN_DENSE=true
#
# Optional examples:
#   BUDGETS="1760 2112" bash run_gsm8k_multiseed_key_qvnorm_dense.sh
#   SEEDS="45 28" BUDGETS="2112" RUN_DENSE=false bash run_gsm8k_multiseed_key_qvnorm_dense.sh
#   SPARSE_MODES="ours_depth_balanced" RUN_DENSE=false bash run_gsm8k_multiseed_key_qvnorm_dense.sh
# ============================================================

if ! grep -q "upload_score_mode" src/run_uie_lora.py; then
    echo "[ERROR] src/run_uie_lora.py does not support --upload_score_mode." >&2
    exit 1
fi
if ! grep -q "sn_p1_norm_mode" src/run_uie_lora.py; then
    echo "[ERROR] src/run_uie_lora.py does not support --sn_p1_norm_mode. Please use qvblock_norm version." >&2
    exit 1
fi
if ! grep -q "depth_balanced" src/federated_uie_lora.py; then
    echo "[ERROR] src/federated_uie_lora.py does not support depth_balanced. Please use qvblock_norm version." >&2
    exit 1
fi

# ===== Basic config =====
method=lora_origin
lora_rank=8
lamda_2=0
lamda_1=0
lr=${LR:-1e-04}
radius=0

di_alpha=${DIRICHLET_ALPHA:-10}
num_clients=${NUM_CLIENTS:-50}
clients_per_round=${CLIENTS_PER_ROUND:-10}
global_rounds=${GLOBAL_ROUNDS:-5}
local_epochs=${LOCAL_EPOCHS:-10}

SEEDS=${SEEDS:-"42 45 28"}
BUDGETS=${BUDGETS:-"2112"}
SPARSE_MODES=${SPARSE_MODES:-"ab_factor ab_effective ours_depth_balanced"}
RUN_DENSE=${RUN_DENSE:-true}
DENSE_BUDGET=${DENSE_BUDGET:-0}

# Ours default settings: final current version.
SN_GAP_ETA=${SN_GAP_ETA:-0.0}
SN_P1_NORM_MODE=${SN_P1_NORM_MODE:-depth_balanced}
SN_DEPTH_GROUP_RATIOS=${SN_DEPTH_GROUP_RATIOS:-1,1,2}

# ===== Paths =====
data_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data
task_config_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config
model_path=/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct
ACC_CONFIG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py

OUT_ROOT=${OUT_ROOT:-results/Qwen2_gsm8k/multiseed_key_qvnorm_dense}
mkdir -p ${OUT_ROOT}/logs

GPU_GROUP_0=${GPU_GROUP_0:-0,1,2,3}
GPU_GROUP_1=${GPU_GROUP_1:-4,5,6,7}
PARALLEL_JOBS=${PARALLEL_JOBS:-2}
SKIP_FINISHED=${SKIP_FINISHED:-true}

run_one() {
    local gpus=$1
    local mode=$2
    local budget=$3
    local seed=$4

    local port
    port=$(shuf -i25000-30000 -n1)

    local out_dir
    local run_name
    if [ "${mode}" = "dense_full" ]; then
        out_dir=${OUT_ROOT}/${mode}_K${clients_per_round}_seed${seed}
        run_name=gsm8k_${mode}_K${clients_per_round}_seed${seed}
    else
        out_dir=${OUT_ROOT}/${mode}_budget${budget}_K${clients_per_round}_seed${seed}
        run_name=gsm8k_${mode}_budget${budget}_K${clients_per_round}_seed${seed}
    fi
    local log_file=${OUT_ROOT}/logs/${run_name}.log

    if [ "${SKIP_FINISHED}" = "true" ] && [ -f "${out_dir}/all_results.json" ]; then
        echo "[SKIP] ${run_name} already has all_results.json"
        return 0
    fi

    local atomic_param=""
    local score_param=""
    local sn_params=""
    local budget_to_use="${budget}"

    if [ "${mode}" = "tensor_topk" ]; then
        atomic_param="--upload_atomic_mode tensor"
        score_param="--upload_score_mode factor_norm"
    elif [ "${mode}" = "ab_factor" ]; then
        atomic_param="--upload_atomic_mode ab_pair"
        score_param="--upload_score_mode factor_norm"
    elif [ "${mode}" = "ab_effective" ]; then
        atomic_param="--upload_atomic_mode ab_pair"
        score_param="--upload_score_mode effective_norm"
    elif [ "${mode}" = "ours_depth_balanced" ]; then
        atomic_param="--upload_atomic_mode qv_block"
        score_param="--upload_score_mode sn_p1p2"
        sn_params="--sn_gap_eta ${SN_GAP_ETA} --sn_force_full_budget True --sn_save_diagnostics True --sn_p1_norm_mode ${SN_P1_NORM_MODE} --sn_depth_group_ratios ${SN_DEPTH_GROUP_RATIOS}"
    elif [ "${mode}" = "dense_full" ]; then
        # Dense full upload: comm_budget <= 0 bypasses sparse selection.
        atomic_param="--upload_atomic_mode ab_pair"
        score_param="--upload_score_mode factor_norm"
        budget_to_use="${DENSE_BUDGET}"
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
       --comm_budget ${budget_to_use} \
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

job_idx=0
pids=()
launch_job() {
    local mode=$1
    local budget=$2
    local seed=$3
    local gpus
    if (( job_idx % 2 == 0 )); then
      gpus=${GPU_GROUP_0}
    else
      gpus=${GPU_GROUP_1}
    fi
    run_one ${gpus} ${mode} ${budget} ${seed} &
    pids+=("$!")
    job_idx=$((job_idx + 1))

    if (( ${#pids[@]} >= PARALLEL_JOBS )); then
      for pid in "${pids[@]}"; do
        wait ${pid}
      done
      pids=()
    fi
}

# Sparse methods under matched budgets.
for seed in ${SEEDS}; do
  for budget in ${BUDGETS}; do
    for mode in ${SPARSE_MODES}; do
      launch_job ${mode} ${budget} ${seed}
    done
  done

  # Dense full-upload reference once per seed.
  if [ "${RUN_DENSE}" = "true" ]; then
    launch_job dense_full 0 ${seed}
  fi
done

for pid in "${pids[@]}"; do
  wait ${pid}
done

echo "[DONE] Multi-seed key experiment finished. Results saved to ${OUT_ROOT}"
