#!/bin/bash
set -euo pipefail
set -x

export PYTHONWARNINGS="ignore"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/home/qiuwenqi/.cache/huggingface}
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

# ===============================================================
# Llama-3.1-8B-Instruct + GSM8K formal overnight experiments
#
# Default plan for 8 GPUs:
#   A) Main multi-seed comparison at 12.5% budget:
#        seeds = 42,28,45
#        methods = dense, ab_factor, ab_effective, ours_depth_balanced
#        budget = 1144 packets/client/round
#   B) Optional seed42 budget sweep for sparse methods:
#        budgets = 572,1144,1716,2288
#        methods = ab_factor, ab_effective, ours_depth_balanced
#
# Run in project root:
#   bash run_llama31_gsm8k_formal_overnight.sh
#
# Light version only main multi-seed:
#   RUN_BUDGET_SWEEP=false bash run_llama31_gsm8k_formal_overnight.sh
#
# Only budget sweep:
#   RUN_MAIN_MULTI_SEED=false RUN_BUDGET_SWEEP=true bash run_llama31_gsm8k_formal_overnight.sh
# ============================================================

MODEL_PATH=${MODEL_PATH:-/home/qiuwenqi/LLM/models/Llama-3.1-8B-Instruct}
DATA_DIR=${DATA_DIR:-/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data}
TASK_CONFIG_DIR=${TASK_CONFIG_DIR:-/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config}
ACC_CONFIG=${ACC_CONFIG:-script_accelerate/accelerate_config.yaml}
PY_SCRIPT=${PY_SCRIPT:-src/run_uie_lora.py}
OUT_ROOT=${OUT_ROOT:-results/Llama31_gsm8k/formal_overnight}

# 8-GPU default: two 4-GPU jobs in parallel.
GPU_GROUPS=${GPU_GROUPS:-"0,1,2,3 4,5,6,7"}
PARALLEL_JOBS=${PARALLEL_JOBS:-2}

# Main comparison: 12.5% budget for Llama-3.1-8B.
# Smoke log showed: full_upload_cost=9152, qv_block unit_cost=286,
# so 12.5% = 9152/8 = 1144 = 4 qv-blocks.
MAIN_SEEDS=${MAIN_SEEDS:-"42 28 45"}
MAIN_BUDGET=${MAIN_BUDGET:-1144}
MAIN_MODES=${MAIN_MODES:-"dense ab_factor ab_effective ours_depth_balanced"}
RUN_MAIN_MULTI_SEED=${RUN_MAIN_MULTI_SEED:-true}

# Extra seed42 budget curve. 572/1144/1716/2288 = 6.25/12.5/18.75/25%.
RUN_BUDGET_SWEEP=${RUN_BUDGET_SWEEP:-true}
SWEEP_SEED=${SWEEP_SEED:-42}
SWEEP_BUDGETS=${SWEEP_BUDGETS:-"572 1144 1716 2288"}
SWEEP_MODES=${SWEEP_MODES:-"ab_factor ab_effective ours_depth_balanced"}

SKIP_FINISHED=${SKIP_FINISHED:-true}

mkdir -p ${OUT_ROOT}/logs

if [ ! -d "${MODEL_PATH}" ]; then
  echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi
if [ ! -d "${DATA_DIR}" ]; then
  echo "[ERROR] DATA_DIR does not exist: ${DATA_DIR}" >&2
  exit 1
fi
if [ ! -d "${TASK_CONFIG_DIR}" ]; then
  echo "[ERROR] TASK_CONFIG_DIR does not exist: ${TASK_CONFIG_DIR}" >&2
  exit 1
fi

# Cheap tokenizer/config sanity check.
python - <<PY
from transformers import AutoConfig, AutoTokenizer
path = "${MODEL_PATH}"
config = AutoConfig.from_pretrained(path)
tok = AutoTokenizer.from_pretrained(path, use_fast=True)
print("[Preflight] model_type=", getattr(config, "model_type", None))
print("[Preflight] hidden_size=", getattr(config, "hidden_size", None), "layers=", getattr(config, "num_hidden_layers", None))
print("[Preflight] heads=", getattr(config, "num_attention_heads", None), "kv_heads=", getattr(config, "num_key_value_heads", None))
print("[Preflight] bos/eos/pad=", tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
print("[Preflight] tokenizer class=", tok.__class__.__name__)
PY

choose_gpu_group() {
  local idx=$1
  local arr=(${GPU_GROUPS})
  local n=${#arr[@]}
  echo ${arr[$((idx % n))]}
}

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL_JOBS}" ]; do
    sleep 30
  done
}

run_one() {
  local gpu_group=$1
  local mode=$2
  local budget=$3
  local seed=$4

  local budget_tag="budget${budget}"
  local out_dir=""
  local run_name=""
  local log_file=""
  local comm_budget=${budget}
  local atomic_args=""
  local sn_args=""

  if [ "${mode}" = "dense" ]; then
    comm_budget=0
    out_dir=${OUT_ROOT}/dense_full_K10_seed${seed}
    run_name=llama31_gsm8k_dense_full_K10_seed${seed}
    atomic_args="--upload_atomic_mode ab_pair --upload_score_mode factor_norm"
    sn_args=""
  elif [ "${mode}" = "ab_factor" ]; then
    out_dir=${OUT_ROOT}/ab_factor_${budget_tag}_K10_seed${seed}
    run_name=llama31_gsm8k_ab_factor_${budget_tag}_K10_seed${seed}
    atomic_args="--upload_atomic_mode ab_pair --upload_score_mode factor_norm"
    sn_args=""
  elif [ "${mode}" = "ab_effective" ]; then
    out_dir=${OUT_ROOT}/ab_effective_${budget_tag}_K10_seed${seed}
    run_name=llama31_gsm8k_ab_effective_${budget_tag}_K10_seed${seed}
    atomic_args="--upload_atomic_mode ab_pair --upload_score_mode effective_norm"
    sn_args=""
  elif [ "${mode}" = "ours_depth_balanced" ]; then
    out_dir=${OUT_ROOT}/ours_depth_balanced_${budget_tag}_K10_seed${seed}
    run_name=llama31_gsm8k_ours_depth_balanced_${budget_tag}_K10_seed${seed}
    atomic_args="--upload_atomic_mode qv_block --upload_score_mode sn_p1p2"
    sn_args="--sn_gap_eta 0.0 --sn_force_full_budget True --sn_save_diagnostics True --sn_p1_norm_mode depth_balanced --sn_depth_group_ratios 1,1,2"
  else
    echo "[ERROR] Unknown mode: ${mode}" >&2
    exit 1
  fi

  log_file=${OUT_ROOT}/logs/${run_name}.log

  if [ "${SKIP_FINISHED}" = "true" ] && [ -f "${out_dir}/all_results.json" ]; then
    echo "[SKIP] ${run_name}: all_results.json already exists."
    return 0
  fi

  local port
  port=$(shuf -i25000-30000 -n1)

  echo "[RUN] gpu=${gpu_group} mode=${mode} budget=${comm_budget} seed=${seed} out=${out_dir}"

  CUDA_VISIBLE_DEVICES=${gpu_group} accelerate launch --config_file ${ACC_CONFIG} \
    --main_process_port ${port} \
    ${PY_SCRIPT} \
    --report_to none \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --lora_dim 8 \
    --model_name_or_path ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --task_config_dir ${TASK_CONFIG_DIR} \
    --output_dir ${out_dir} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --global_rounds 5 \
    --local_epochs 10 \
    --num_clients 50 \
    --clients_per_round 10 \
    --dirichlet_alpha 10 \
    --partition_strategy quantity \
    --comm_budget ${comm_budget} \
    --learning_rate 1e-04 \
    --run_name ${run_name} \
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
    --logging_steps 5 \
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
    ${atomic_args} \
    ${sn_args} \
    --upload_diversity_mode none \
    --diagnose_pair_saliency False \
    --diagnose_residual_errors False \
    --lora_residual_accumulation False \
    > ${log_file} 2>&1

  echo "[OK] ${run_name} finished. Log: ${log_file}"
  if [ -f "${out_dir}/all_results.json" ]; then
    cat "${out_dir}/all_results.json"
  fi
}

job_idx=0

if [ "${RUN_MAIN_MULTI_SEED}" = "true" ]; then
  echo "[PLAN] Main multi-seed comparison: seeds=${MAIN_SEEDS}, budget=${MAIN_BUDGET}, modes=${MAIN_MODES}"
  for seed in ${MAIN_SEEDS}; do
    for mode in ${MAIN_MODES}; do
      wait_for_slot
      gpu=$(choose_gpu_group ${job_idx})
      run_one ${gpu} ${mode} ${MAIN_BUDGET} ${seed} &
      job_idx=$((job_idx + 1))
    done
  done
fi

if [ "${RUN_BUDGET_SWEEP}" = "true" ]; then
  echo "[PLAN] Seed${SWEEP_SEED} budget sweep: budgets=${SWEEP_BUDGETS}, modes=${SWEEP_MODES}"
  for budget in ${SWEEP_BUDGETS}; do
    for mode in ${SWEEP_MODES}; do
      wait_for_slot
      gpu=$(choose_gpu_group ${job_idx})
      run_one ${gpu} ${mode} ${budget} ${SWEEP_SEED} &
      job_idx=$((job_idx + 1))
    done
  done
fi

wait

echo "[DONE] Llama-3.1 GSM8K formal overnight experiments finished. Outputs: ${OUT_ROOT}"
