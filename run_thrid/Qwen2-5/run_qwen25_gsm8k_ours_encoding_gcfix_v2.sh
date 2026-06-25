#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/qiuwenqi/LLM/Third_work}"
cd "${ROOT}"

MODEL_PATH="${MODEL_PATH:-/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct}"

# Robust path defaults. The old script used configs/gsm8k, but your existing GSM8K
# scripts use data_gsm8k/gsm8k_uie_task_config.
DATA_DIR="${DATA_DIR:-${ROOT}/data_gsm8k/gsm8k_uie_data}"
TASK_CONFIG_DIR="${TASK_CONFIG_DIR:-${ROOT}/data_gsm8k/gsm8k_uie_task_config}"
INSTRUCTION_FILE="${INSTRUCTION_FILE:-${ROOT}/configs/instruction_config.json}"

# Fallbacks if users intentionally keep relative project paths.
if [ ! -d "${DATA_DIR}" ] && [ -d "data_gsm8k/gsm8k_uie_data" ]; then
  DATA_DIR="data_gsm8k/gsm8k_uie_data"
fi
if [ ! -d "${TASK_CONFIG_DIR}" ]; then
  for cand in \
    "data_gsm8k/gsm8k_uie_task_config" \
    "configs/gsm8k" \
    "src/configs/gsm8k" \
    "run_thrid/Qwen2-5/configs/gsm8k"; do
    if [ -d "$cand" ]; then
      TASK_CONFIG_DIR="$cand"
      break
    fi
  done
fi
if [ ! -f "${INSTRUCTION_FILE}" ]; then
  for cand in \
    "configs/instruction_config.json" \
    "src/configs/instruction_config.json" \
    "run_thrid/Qwen2-5/configs/instruction_config.json"; do
    if [ -f "$cand" ]; then
      INSTRUCTION_FILE="$cand"
      break
    fi
  done
fi

# Preflight: fail before accelerate launches 4 processes, with useful path hints.
echo "===== Path check ====="
echo "ROOT=${ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "TASK_CONFIG_DIR=${TASK_CONFIG_DIR}"
echo "INSTRUCTION_FILE=${INSTRUCTION_FILE}"

if [ ! -d "${DATA_DIR}" ]; then
  echo "[ERROR] DATA_DIR does not exist: ${DATA_DIR}" >&2
  echo "Try: find ${ROOT} -maxdepth 4 -type d -iname '*gsm8k*' | sort" >&2
  exit 1
fi
if [ ! -d "${TASK_CONFIG_DIR}" ]; then
  echo "[ERROR] TASK_CONFIG_DIR does not exist: ${TASK_CONFIG_DIR}" >&2
  echo "Your previous GSM8K script used: ${ROOT}/data_gsm8k/gsm8k_uie_task_config" >&2
  echo "Try: find ${ROOT} -maxdepth 5 -type d -iname '*task*config*' -o -type d -iname '*gsm8k*' | sort" >&2
  exit 1
fi
if [ -n "${INSTRUCTION_FILE}" ] && [ ! -f "${INSTRUCTION_FILE}" ]; then
  echo "[WARN] INSTRUCTION_FILE does not exist: ${INSTRUCTION_FILE}. It will not be passed."
  INSTRUCTION_FILE=""
fi

OUT_ROOT="${OUT_ROOT:-results/Qwen2_gsm8k/ours_encoding_gcfix_seed42}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

SEED="${SEED:-42}"
NUM_CLIENTS="${NUM_CLIENTS:-50}"
CLIENTS_PER_ROUND="${CLIENTS_PER_ROUND:-10}"
ROUNDS="${ROUNDS:-5}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
BUDGET="${BUDGET:-2112}"

# First verify one variant. After it succeeds, set VARIANTS to run more.
VARIANTS="${VARIANTS:-enc_compeft_m3}"

GPU_GROUP="${GPU_GROUP:-0,1,2,3}"
MAIN_PORT_BASE="${MAIN_PORT_BASE:-29730}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"

run_variant() {
  local variant="$1"
  local port="$2"

  local enc_mode="none"
  local mult="1"
  local p1_norm="depth_balanced"
  local depth_ratios="1,1,2"
  local score_mode="sn_p1p2"
  local method="lora_origin"
  local extra_args=()

  case "${variant}" in
    enc_compeft_m2)
      enc_mode="compeft"; mult="2";;
    enc_compeft_m3)
      enc_mode="compeft"; mult="3";;
    enc_compeft_m4)
      enc_mode="compeft"; mult="4";;
    enc_flasc_m3)
      enc_mode="flasc"; mult="3";;
    depth_rank_compeft_m3)
      enc_mode="compeft"; mult="3"; p1_norm="depth_rank"; depth_ratios="";;
    rawp1_compeft_m3)
      enc_mode="compeft"; mult="3"; p1_norm="raw"; depth_ratios="";;
    flm_topk_fast)
      method="flm_topk"
      extra_args+=(
        --baseline_packet_num "${BUDGET}"
        --baseline_flm_opt_max_iter 40
        --baseline_flm_max_blocks 256
      )
      ;;
    raw_ours)
      enc_mode="none"; mult="1";;
    *)
      echo "[ERROR] Unknown variant: ${variant}" >&2
      exit 1
      ;;
  esac

  local OUT_DIR="${OUT_ROOT}/${variant}_budget${BUDGET}_K${CLIENTS_PER_ROUND}_seed${SEED}"
  local LOG_FILE="${LOG_DIR}/qwen25_gsm8k_${variant}_budget${BUDGET}_K${CLIENTS_PER_ROUND}_seed${SEED}.log"

  echo "============================================================"
  echo "[RUN] ${variant}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
  echo "GPU_GROUP=${GPU_GROUP}, PORT=${port}"
  echo "============================================================"

  rm -rf "${OUT_DIR}"
  mkdir -p "${OUT_DIR}"

  local instruction_args=()
  if [ -n "${INSTRUCTION_FILE}" ]; then
    instruction_args+=(--instruction_file "${INSTRUCTION_FILE}")
  fi

  CUDA_VISIBLE_DEVICES="${GPU_GROUP}" \
  NCCL_DEBUG=WARN \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  accelerate launch \
    --main_process_port "${port}" \
    --num_processes 4 \
    --mixed_precision bf16 \
    src/run_uie_lora.py \
    --mode federated \
    --model_name_or_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --task_config_dir "${TASK_CONFIG_DIR}" \
    "${instruction_args[@]}" \
    --instruction_strategy single \
    --output_dir "${OUT_DIR}" \
    --overwrite_output_dir True \
    --do_train True \
    --do_predict True \
    --predict_with_generate True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --num_train_epochs "${LOCAL_EPOCHS}" \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_steps 10 \
    --save_strategy no \
    --evaluation_strategy no \
    --bf16 True \
    --fp16 False \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --ddp_find_unused_parameters False \
    --max_source_length 512 \
    --max_target_length 16 \
    --generation_max_length 16 \
    --num_beams 1 \
    --lora_dim 8 \
    --num_clients "${NUM_CLIENTS}" \
    --clients_per_round "${CLIENTS_PER_ROUND}" \
    --global_rounds "${ROUNDS}" \
    --federated_seed "${SEED}" \
    --seed "${SEED}" \
    --dirichlet_alpha 10.0 \
    --partition_strategy quantity \
    --comm_budget "${BUDGET}" \
    --method "${method}" \
    --upload_atomic_mode qv_block \
    --upload_score_mode "${score_mode}" \
    --sn_p1_norm_mode "${p1_norm}" \
    --sn_depth_group_ratios "${depth_ratios}" \
    --sn_gap_eta 0.0 \
    --sn_force_full_budget True \
    --sn_encoder_mode "${enc_mode}" \
    --sn_candidate_budget_multiplier "${mult}" \
    --sn_encoder_packet_num "${BUDGET}" \
    --sn_encoder_bit 18 \
    "${extra_args[@]}" \
    2>&1 | tee "${LOG_FILE}"
}

idx=0
for v in ${VARIANTS}; do
  run_variant "$v" "$((MAIN_PORT_BASE + idx))"
  idx=$((idx + 1))
done

echo "===== packing results ====="
rm -f qwen25_gsm8k_ours_encoding_gcfix_pack.zip
zip -r qwen25_gsm8k_ours_encoding_gcfix_pack.zip \
  "${OUT_ROOT}" \
  run_qwen25_gsm8k_ours_encoding_gcfix_v2.sh \
  src/baseline_compressors.py \
  src/run_uie_lora.py \
  src/federated_uie_lora.py \
  src/uie_trainer_lora.py \
  src/uie_collator.py \
  src/gsm8k/gsm8k_metrics.py \
  -x "*/adapter/*" "*/client_states/*" "*/checkpoint*" "*.safetensors" "*.bin"

du -h qwen25_gsm8k_ours_encoding_gcfix_pack.zip
