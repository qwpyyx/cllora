#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/qiuwenqi/LLM/Third_work}"
cd "${ROOT}" || exit 1

LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-/home/qiuwenqi/LLM/models/Llama-3.1-8B-Instruct}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct}"

DATA_DIR="${DATA_DIR:-${ROOT}/data_gsm8k/gsm8k_uie_data}"
TASK_CONFIG_DIR="${TASK_CONFIG_DIR:-${ROOT}/data_gsm8k/gsm8k_uie_task_config}"
INSTRUCTION_FILE="${INSTRUCTION_FILE:-${ROOT}/configs/instruction_config.json}"

NUM_CLIENTS="${NUM_CLIENTS:-50}"
CLIENTS_PER_ROUND="${CLIENTS_PER_ROUND:-10}"
ROUNDS="${ROUNDS:-5}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"

GPU_GROUPS_RAW="${GPU_GROUPS:-0,1,2,3 4,5,6,7}"
read -r -a GPU_GROUPS_ARR <<< "${GPU_GROUPS_RAW}"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_GROUPS_ARR[@]}}"
MAIN_PORT_BASE="${MAIN_PORT_BASE:-30210}"

RUN_LLAMA="${RUN_LLAMA:-true}"
RUN_QWEN="${RUN_QWEN:-true}"
FORCE="${FORCE:-false}"

LLAMA_OUT_ROOT="${LLAMA_OUT_ROOT:-results/Llama31_gsm8k/ours_encoding_followup_depthrank_m6}"
QWEN_OUT_ROOT="${QWEN_OUT_ROOT:-results/Qwen2_gsm8k/ours_encoding_followup_depthrank_m6}"

echo "===== Path check ====="
echo "ROOT=${ROOT}"
echo "LLAMA_MODEL_PATH=${LLAMA_MODEL_PATH}"
echo "QWEN_MODEL_PATH=${QWEN_MODEL_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "TASK_CONFIG_DIR=${TASK_CONFIG_DIR}"
echo "INSTRUCTION_FILE=${INSTRUCTION_FILE}"
echo "GPU_GROUPS=${GPU_GROUPS_RAW}, MAX_PARALLEL=${MAX_PARALLEL}"

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

variant_to_args() {
  local variant="$1"
  ENC_MODE="compeft"
  MULT="3"
  P1_NORM="depth_balanced"
  DEPTH_RATIOS="1,1,2"
  SCORE_MODE="sn_p1p2"
  METHOD="lora_origin"
  EXTRA_ARGS=()

  case "${variant}" in
    enc_compeft_m2) ENC_MODE="compeft"; MULT="2";;
    enc_compeft_m3) ENC_MODE="compeft"; MULT="3";;
    enc_compeft_m4) ENC_MODE="compeft"; MULT="4";;
    enc_compeft_m5) ENC_MODE="compeft"; MULT="5";;
    enc_compeft_m6) ENC_MODE="compeft"; MULT="6";;
    enc_compeft_m7) ENC_MODE="compeft"; MULT="7";;
    enc_compeft_m8) ENC_MODE="compeft"; MULT="8";;

    depth_rank_compeft_m3) ENC_MODE="compeft"; MULT="3"; P1_NORM="depth_rank"; DEPTH_RATIOS="";;
    depth_rank_compeft_m4) ENC_MODE="compeft"; MULT="4"; P1_NORM="depth_rank"; DEPTH_RATIOS="";;
    depth_rank_compeft_m5) ENC_MODE="compeft"; MULT="5"; P1_NORM="depth_rank"; DEPTH_RATIOS="";;
    depth_rank_compeft_m6) ENC_MODE="compeft"; MULT="6"; P1_NORM="depth_rank"; DEPTH_RATIOS="";;
    depth_rank_compeft_m7) ENC_MODE="compeft"; MULT="7"; P1_NORM="depth_rank"; DEPTH_RATIOS="";;

    rawp1_compeft_m3) ENC_MODE="compeft"; MULT="3"; P1_NORM="raw"; DEPTH_RATIOS="";;
    rawp1_compeft_m4) ENC_MODE="compeft"; MULT="4"; P1_NORM="raw"; DEPTH_RATIOS="";;
    enc_flasc_m3) ENC_MODE="flasc"; MULT="3";;
    enc_flasc_m4) ENC_MODE="flasc"; MULT="4";;
    *)
      echo "[ERROR] Unknown variant: ${variant}" >&2
      return 1
      ;;
  esac
}

run_job() {
  local phase="$1"
  local model_path="$2"
  local out_root="$3"
  local log_prefix="$4"
  local job="$5"
  local gpu_group="$6"
  local port="$7"

  IFS=":" read -r VARIANT SEED BUDGET <<< "${job}"
  variant_to_args "${VARIANT}" || return 1

  local log_dir="${out_root}/logs"
  local status_dir="${out_root}/status"
  local out_dir="${out_root}/${VARIANT}_budget${BUDGET}_K${CLIENTS_PER_ROUND}_seed${SEED}"
  local log_file="${log_dir}/${log_prefix}_${VARIANT}_budget${BUDGET}_K${CLIENTS_PER_ROUND}_seed${SEED}.log"
  local status_file="${status_dir}/${VARIANT}_budget${BUDGET}_K${CLIENTS_PER_ROUND}_seed${SEED}.status"

  if [ "${FORCE}" != "true" ] && [ -f "${out_dir}/all_results.json" ] && [ -f "${out_dir}/predict_results.json" ]; then
    echo "[SKIP][${phase}] ${job} already completed. Set FORCE=true to rerun."
    echo "SKIPPED" > "${status_file}"
    return 0
  fi

  echo "============================================================"
  echo "[RUN][${phase}] ${VARIANT} | seed=${SEED} | budget=${BUDGET}"
  echo "OUT_DIR=${out_dir}"
  echo "LOG_FILE=${log_file}"
  echo "GPU_GROUP=${gpu_group}, PORT=${port}"
  echo "MODEL_PATH=${model_path}"
  echo "============================================================"

  mkdir -p "${out_dir}"
  local instruction_args=()
  if [ -n "${INSTRUCTION_FILE}" ]; then
    instruction_args+=(--instruction_file "${INSTRUCTION_FILE}")
  fi

  set +e
  CUDA_VISIBLE_DEVICES="${gpu_group}" \
  WANDB_DISABLED=true \
  NCCL_DEBUG=WARN \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  accelerate launch \
    --main_process_port "${port}" \
    --num_processes 4 \
    --mixed_precision bf16 \
    src/run_uie_lora.py \
    --mode federated \
    --model_name_or_path "${model_path}" \
    --data_dir "${DATA_DIR}" \
    --task_config_dir "${TASK_CONFIG_DIR}" \
    "${instruction_args[@]}" \
    --instruction_strategy single \
    --output_dir "${out_dir}" \
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
    2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e

  if [ "${rc}" -eq 0 ]; then
    echo "SUCCESS" > "${status_file}"
  else
    echo "FAILED rc=${rc}" > "${status_file}"
  fi
  return "${rc}"
}

run_phase() {
  local phase="$1"
  local model_path="$2"
  local out_root="$3"
  local log_prefix="$4"
  local port_base="$5"
  shift 5
  local jobs=("$@")

  mkdir -p "${out_root}/logs" "${out_root}/status"
  echo
  echo "################################################################"
  echo "===== Phase ${phase}: ${#jobs[@]} jobs ====="
  printf '  %s\n' "${jobs[@]}"
  echo "OUT_ROOT=${out_root}"
  echo "################################################################"

  local overall_failed=0
  local batch_start
  for ((batch_start=0; batch_start<${#jobs[@]}; batch_start+=MAX_PARALLEL)); do
    local pids=()
    local labels=()
    local slot
    for ((slot=0; slot<MAX_PARALLEL && batch_start+slot<${#jobs[@]}; slot++)); do
      local job="${jobs[$((batch_start+slot))]}"
      local gpu="${GPU_GROUPS_ARR[$((slot % ${#GPU_GROUPS_ARR[@]}))]}"
      local port="$((port_base + batch_start + slot))"
      (run_job "${phase}" "${model_path}" "${out_root}" "${log_prefix}" "${job}" "${gpu}" "${port}") &
      pids+=("$!")
      labels+=("${job}")
      sleep 5
    done

    local wait_idx
    for wait_idx in "${!pids[@]}"; do
      if wait "${pids[$wait_idx]}"; then
        echo "[DONE][${phase}] ${labels[$wait_idx]}"
      else
        echo "[FAILED][${phase}] ${labels[$wait_idx]}"
        overall_failed=1
      fi
    done
  done

  local summary_file="${out_root}/followup_summary.txt"
  {
    echo "===== ${phase} finished at $(date) ====="
    echo "overall_failed=${overall_failed}"
    echo
    echo "===== status files ====="
    find "${out_root}/status" -type f -name '*.status' -print -exec cat {} \; | sed 's/^/  /'
    echo
    echo "===== result files ====="
    find "${out_root}" -maxdepth 3 -type f \( -name 'all_results.json' -o -name 'predict_results.json' -o -name 'sn_encoder_compression_summary.json' \) | sort
    echo
    echo "===== recent errors ====="
    grep -R "Traceback\|Error\|Exception\|ValueError\|RuntimeError\|TypeError\|AttributeError\|KeyError\|UnboundLocalError" "${out_root}/logs" 2>/dev/null | tail -200 || true
  } > "${summary_file}"
  cat "${summary_file}"
  PHASE_FAILED="${overall_failed}"
}

# Llama: verify whether depth-rank remains best under higher budgets and whether
# m4/m6 are competitive. Full qv-block cost is 9152, so 1144/1716/2288/2860/3432
# are 12.5/18.75/25/31.25/37.5%.
LLAMA_JOBS=(
  "depth_rank_compeft_m3:28:1716"
  "depth_rank_compeft_m3:45:1716"
  "depth_rank_compeft_m3:28:2288"
  "depth_rank_compeft_m3:45:2288"
  "depth_rank_compeft_m3:42:2860"
  "depth_rank_compeft_m3:42:3432"

  "depth_rank_compeft_m4:28:1144"
  "depth_rank_compeft_m4:42:1144"
  "depth_rank_compeft_m4:45:1144"
  "depth_rank_compeft_m4:28:2288"
  "depth_rank_compeft_m4:42:2288"
  "depth_rank_compeft_m4:45:2288"

  "enc_compeft_m6:28:1144"
  "enc_compeft_m6:45:1144"
  "enc_compeft_m3:42:2860"
  "enc_compeft_m3:42:3432"
)

# Qwen: m6 single seed was strongest; this phase gives it multi-seed evidence
# and checks whether depth-rank with larger candidate regions is even better.
QWEN_JOBS=(
  "enc_compeft_m5:28:2112"
  "enc_compeft_m5:45:2112"
  "enc_compeft_m6:28:2112"
  "enc_compeft_m6:45:2112"
  "enc_compeft_m7:28:2112"
  "enc_compeft_m7:45:2112"
  "enc_compeft_m8:28:2112"
  "enc_compeft_m8:45:2112"

  "depth_rank_compeft_m5:28:2112"
  "depth_rank_compeft_m5:42:2112"
  "depth_rank_compeft_m5:45:2112"
  "depth_rank_compeft_m6:28:2112"
  "depth_rank_compeft_m6:42:2112"
  "depth_rank_compeft_m6:45:2112"
  "depth_rank_compeft_m7:42:2112"

  "depth_rank_compeft_m3:28:1056"
  "depth_rank_compeft_m3:45:1056"
  "depth_rank_compeft_m3:28:1408"
  "depth_rank_compeft_m3:45:1408"
  "depth_rank_compeft_m3:28:1760"
  "depth_rank_compeft_m3:45:1760"

  "enc_compeft_m4:28:2816"
  "enc_compeft_m4:45:2816"
  "enc_compeft_m6:42:1760"
  "enc_compeft_m6:42:2816"
  "depth_rank_compeft_m6:42:2816"
)

ALL_FAILED=0

if [ "${RUN_LLAMA}" = "true" ]; then
  if [ -d "${LLAMA_MODEL_PATH}" ]; then
    run_phase "Llama31" "${LLAMA_MODEL_PATH}" "${LLAMA_OUT_ROOT}" "llama31_gsm8k" "${MAIN_PORT_BASE}" "${LLAMA_JOBS[@]}"
    if [ "${PHASE_FAILED}" -ne 0 ]; then ALL_FAILED=1; fi
  else
    echo "[WARN] LLAMA_MODEL_PATH does not exist: ${LLAMA_MODEL_PATH}. Skip Llama."
  fi
fi

if [ "${RUN_QWEN}" = "true" ]; then
  if [ -d "${QWEN_MODEL_PATH}" ]; then
    run_phase "Qwen25" "${QWEN_MODEL_PATH}" "${QWEN_OUT_ROOT}" "qwen25_gsm8k" "$((MAIN_PORT_BASE + 1000))" "${QWEN_JOBS[@]}"
    if [ "${PHASE_FAILED}" -ne 0 ]; then ALL_FAILED=1; fi
  else
    echo "[WARN] QWEN_MODEL_PATH does not exist: ${QWEN_MODEL_PATH}. Skip Qwen."
    echo "[WARN] Set QWEN_MODEL_PATH=/path/to/Qwen2.5-14B-Instruct if needed."
  fi
fi

echo
echo "===== packing follow-up results ====="
PACK="${PACK:-gsm8k_ours_encoding_followup_depthrank_m6_pack.zip}"
LIST_FILE="/tmp/gsm8k_ours_encoding_followup_depthrank_m6_pack_files.txt"
: > "${LIST_FILE}"

for out_root in "${LLAMA_OUT_ROOT}" "${QWEN_OUT_ROOT}"; do
  if [ -d "${out_root}" ]; then
    find "${out_root}" \
      -type f \
      \( -name "*.log" -o -name "*.json" -o -name "*.jsonl" -o -name "*.txt" -o -name "*.status" \) \
      ! -path "*/adapter/*" \
      ! -path "*/client_states/*" \
      ! -path "*/checkpoint*" \
      ! -path "*/cache/*" \
      ! -name "*.safetensors" \
      ! -name "*.bin" \
      >> "${LIST_FILE}"
  fi
done

for src_file in \
  src/baseline_compressors.py \
  src/run_uie_lora.py \
  src/federated_uie_lora.py \
  src/uie_trainer_lora.py \
  src/uie_collator.py \
  src/gsm8k/gsm8k_metrics.py; do
  if [ -f "${src_file}" ]; then
    printf "%s\n" "${src_file}" >> "${LIST_FILE}"
  fi
done

if [ -f "$0" ]; then
  printf "%s\n" "$0" >> "${LIST_FILE}"
fi

sort -u "${LIST_FILE}" -o "${LIST_FILE}"
zip -@ "${PACK}" < "${LIST_FILE}"
du -h "${PACK}"

if [ "${ALL_FAILED}" -ne 0 ]; then
  echo "[WARN] Some jobs failed. Upload ${PACK}; logs and summaries are included."
  exit 1
fi

echo "[OK] Follow-up finished successfully."
