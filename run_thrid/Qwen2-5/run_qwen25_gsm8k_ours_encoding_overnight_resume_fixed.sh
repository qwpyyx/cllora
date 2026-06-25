#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/qiuwenqi/LLM/Third_work}"
cd "${ROOT}" || exit 1

MODEL_PATH="${MODEL_PATH:-/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct}"
DATA_DIR="${DATA_DIR:-${ROOT}/data_gsm8k/gsm8k_uie_data}"
TASK_CONFIG_DIR="${TASK_CONFIG_DIR:-${ROOT}/data_gsm8k/gsm8k_uie_task_config}"
INSTRUCTION_FILE="${INSTRUCTION_FILE:-${ROOT}/configs/instruction_config.json}"

# Fallbacks for relative project paths.
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

echo "===== Path check ====="
echo "ROOT=${ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "TASK_CONFIG_DIR=${TASK_CONFIG_DIR}"
echo "INSTRUCTION_FILE=${INSTRUCTION_FILE}"

if [ ! -d "${DATA_DIR}" ]; then
  echo "[ERROR] DATA_DIR does not exist: ${DATA_DIR}" >&2
  exit 1
fi
if [ ! -d "${TASK_CONFIG_DIR}" ]; then
  echo "[ERROR] TASK_CONFIG_DIR does not exist: ${TASK_CONFIG_DIR}" >&2
  exit 1
fi
if [ -n "${INSTRUCTION_FILE}" ] && [ ! -f "${INSTRUCTION_FILE}" ]; then
  echo "[WARN] INSTRUCTION_FILE does not exist: ${INSTRUCTION_FILE}. It will not be passed."
  INSTRUCTION_FILE=""
fi

# Basic experiment settings, aligned with previous Qwen2.5 GSM8K runs.
NUM_CLIENTS="${NUM_CLIENTS:-50}"
CLIENTS_PER_ROUND="${CLIENTS_PER_ROUND:-10}"
ROUNDS="${ROUNDS:-5}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
DEFAULT_BUDGET="${DEFAULT_BUDGET:-2112}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
OUT_ROOT="${OUT_ROOT:-results/Qwen2_gsm8k/ours_encoding_overnight}"
LOG_DIR="${OUT_ROOT}/logs"
STATUS_DIR="${OUT_ROOT}/status"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

# GPU groups. Default: two 4-GPU jobs in parallel on an 8-GPU machine.
GPU_GROUPS_RAW="${GPU_GROUPS:-0,1,2,3 4,5,6,7}"
read -r -a GPU_GROUPS_ARR <<< "${GPU_GROUPS_RAW}"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_GROUPS_ARR[@]}}"
MAIN_PORT_BASE="${MAIN_PORT_BASE:-29810}"
FORCE="${FORCE:-false}"
RUN_FLM_FAST="${RUN_FLM_FAST:-false}"

# Job format: variant:seed:budget
# Default overnight set:
#   1) m2/m4 on seed42 to compare candidate multiplier around the completed m3 result.
#   2) m3 on seed28/45 for multi-seed evidence.
#   3) alternative scheduling variants on seed42.
#   4) budget curve for m3 on seed42.
JOBS=(
  # Remaining jobs after the partial run.
  # Completed in the partial pack: enc_compeft_m2 seed42, enc_compeft_m4 seed42,
  # enc_compeft_m3 seed45, enc_flasc_m3 seed42.
  # Previously completed in gcfix root: enc_compeft_m3 seed42.
  "enc_compeft_m3:28:${DEFAULT_BUDGET}"
  "depth_rank_compeft_m3:42:${DEFAULT_BUDGET}"
  "rawp1_compeft_m3:42:${DEFAULT_BUDGET}"
  "enc_compeft_m3:42:1760"
  "enc_compeft_m3:42:2816"
)

if [ "${RUN_FLM_FAST}" = "true" ]; then
  JOBS+=("flm_topk_fast:42:${DEFAULT_BUDGET}")
fi

variant_to_args() {
  local variant="$1"
  ENC_MODE="none"
  MULT="1"
  P1_NORM="depth_balanced"
  DEPTH_RATIOS="1,1,2"
  SCORE_MODE="sn_p1p2"
  METHOD="lora_origin"
  EXTRA_ARGS=()

  case "${variant}" in
    enc_compeft_m2)
      ENC_MODE="compeft"; MULT="2";;
    enc_compeft_m3)
      ENC_MODE="compeft"; MULT="3";;
    enc_compeft_m4)
      ENC_MODE="compeft"; MULT="4";;
    enc_flasc_m3)
      ENC_MODE="flasc"; MULT="3";;
    depth_rank_compeft_m3)
      ENC_MODE="compeft"; MULT="3"; P1_NORM="depth_rank"; DEPTH_RATIOS="";;
    rawp1_compeft_m3)
      ENC_MODE="compeft"; MULT="3"; P1_NORM="raw"; DEPTH_RATIOS="";;
    raw_ours)
      ENC_MODE="none"; MULT="1";;
    flm_topk_fast)
      METHOD="flm_topk"
      EXTRA_ARGS+=(
        --baseline_packet_num "${BUDGET}"
        --baseline_flm_opt_max_iter 40
        --baseline_flm_max_blocks 256
      )
      ;;
    *)
      echo "[ERROR] Unknown variant: ${variant}" >&2
      return 1
      ;;
  esac
}

run_job() {
  local job="$1"
  local gpu_group="$2"
  local port="$3"

  IFS=":" read -r VARIANT SEED BUDGET <<< "${job}"
  variant_to_args "${VARIANT}" || return 1

  local OUT_DIR="${OUT_ROOT}/${VARIANT}_budget${BUDGET}_K${CLIENTS_PER_ROUND}_seed${SEED}"
  local LOG_FILE="${LOG_DIR}/qwen25_gsm8k_${VARIANT}_budget${BUDGET}_K${CLIENTS_PER_ROUND}_seed${SEED}.log"
  local STATUS_FILE="${STATUS_DIR}/${VARIANT}_budget${BUDGET}_K${CLIENTS_PER_ROUND}_seed${SEED}.status"

  if [ "${FORCE}" != "true" ] && [ -f "${OUT_DIR}/all_results.json" ] && [ -f "${OUT_DIR}/predict_results.json" ]; then
    echo "[SKIP] ${job} already has all_results.json and predict_results.json. Set FORCE=true to rerun."
    echo "SKIPPED" > "${STATUS_FILE}"
    return 0
  fi

  echo "============================================================"
  echo "[RUN] ${VARIANT} | seed=${SEED} | budget=${BUDGET}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
  echo "GPU_GROUP=${gpu_group}, PORT=${port}"
  echo "GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
  echo "============================================================"

  rm -rf "${OUT_DIR}"
  mkdir -p "${OUT_DIR}"

  local instruction_args=()
  if [ -n "${INSTRUCTION_FILE}" ]; then
    instruction_args+=(--instruction_file "${INSTRUCTION_FILE}")
  fi

  set +e
  CUDA_VISIBLE_DEVICES="${gpu_group}" \
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
    --method "${METHOD}" \
    --upload_atomic_mode qv_block \
    --upload_score_mode "${SCORE_MODE}" \
    --sn_p1_norm_mode "${P1_NORM}" \
    --sn_depth_group_ratios "${DEPTH_RATIOS}" \
    --sn_gap_eta 0.0 \
    --sn_force_full_budget True \
    --sn_encoder_mode "${ENC_MODE}" \
    --sn_candidate_budget_multiplier "${MULT}" \
    --sn_encoder_packet_num "${BUDGET}" \
    --sn_encoder_bit 18 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"
  local rc=${PIPESTATUS[0]}
  set -e

  if [ "${rc}" -eq 0 ]; then
    echo "SUCCESS" > "${STATUS_FILE}"
  else
    echo "FAILED rc=${rc}" > "${STATUS_FILE}"
  fi
  return "${rc}"
}

run_batch() {
  local start_idx="$1"
  local total="${#JOBS[@]}"
  local pids=()
  local labels=()

  for ((slot=0; slot<MAX_PARALLEL && start_idx+slot<total; slot++)); do
    local job="${JOBS[$((start_idx+slot))]}"
    local gpu="${GPU_GROUPS_ARR[$((slot % ${#GPU_GROUPS_ARR[@]}))]}"
    local port="$((MAIN_PORT_BASE + start_idx + slot))"
    (run_job "${job}" "${gpu}" "${port}") &
    pids+=("$!")
    labels+=("${job}")
    sleep 5
  done

  local failed=0
  local wait_idx
  for wait_idx in "${!pids[@]}"; do
    if wait "${pids[$wait_idx]}"; then
      echo "[DONE] ${labels[$wait_idx]}"
    else
      echo "[FAILED] ${labels[$wait_idx]}"
      failed=1
    fi
  done
  return "${failed}"
}

echo "===== Planned jobs ====="
printf '  %s\n' "${JOBS[@]}"
echo "MAX_PARALLEL=${MAX_PARALLEL}, GPU_GROUPS=${GPU_GROUPS_RAW}"

overall_failed=0
for ((i=0; i<${#JOBS[@]}; i+=MAX_PARALLEL)); do
  run_batch "${i}" || overall_failed=1
done

# Summarize status and errors.
SUMMARY_FILE="${OUT_ROOT}/overnight_summary.txt"
{
  echo "===== finished at $(date) ====="
  echo "overall_failed=${overall_failed}"
  echo
  echo "===== status files ====="
  find "${STATUS_DIR}" -type f -name '*.status' -print -exec cat {} \; | sed 's/^/  /'
  echo
  echo "===== result files ====="
  find "${OUT_ROOT}" -maxdepth 3 -type f \( -name 'all_results.json' -o -name 'predict_results.json' -o -name 'sn_encoder_compression_summary.json' -o -name 'baseline_compression_summary.json' \) | sort
  echo
  echo "===== recent errors ====="
  grep -R "Traceback\|Error\|Exception\|ValueError\|RuntimeError\|TypeError\|AttributeError\|KeyError\|UnboundLocalError" "${LOG_DIR}" 2>/dev/null | tail -200 || true
} > "${SUMMARY_FILE}"

cat "${SUMMARY_FILE}"

echo "===== packing results ====="
PACK="qwen25_gsm8k_ours_encoding_overnight_resume_pack.zip"
rm -f "${PACK}"
rm -f /tmp/qwen25_ours_encoding_overnight_pack_files.txt

find "${OUT_ROOT}" \
  -type f \
  \( -name "*.log" -o -name "*.json" -o -name "*.jsonl" -o -name "*.txt" -o -name "*.status" \) \
  ! -path "*/adapter/*" \
  ! -path "*/client_states/*" \
  ! -path "*/checkpoint*" \
  ! -path "*/cache/*" \
  ! -name "*.safetensors" \
  ! -name "*.bin" \
  > /tmp/qwen25_ours_encoding_overnight_pack_files.txt

printf "%s\n" \
  src/baseline_compressors.py \
  src/run_uie_lora.py \
  src/federated_uie_lora.py \
  src/uie_trainer_lora.py \
  src/uie_collator.py \
  src/gsm8k/gsm8k_metrics.py \
  >> /tmp/qwen25_ours_encoding_overnight_pack_files.txt

# Add this script when it is launched with a relative path under ROOT.
if [ -f "$0" ]; then
  printf "%s\n" "$0" >> /tmp/qwen25_ours_encoding_overnight_pack_files.txt
fi

sort -u /tmp/qwen25_ours_encoding_overnight_pack_files.txt -o /tmp/qwen25_ours_encoding_overnight_pack_files.txt
zip -@ "${PACK}" < /tmp/qwen25_ours_encoding_overnight_pack_files.txt

du -h "${PACK}"

if [ "${overall_failed}" -ne 0 ]; then
  echo "[WARN] Some jobs failed. Upload ${PACK}; the logs and summary are included."
  exit 1
fi
