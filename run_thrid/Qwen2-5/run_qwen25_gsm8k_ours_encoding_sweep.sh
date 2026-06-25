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

# =============================================================
# Qwen2.5-14B + GSM8K: Ours + fine-grained encoding designs
#
# Purpose:
#   Test whether signal-noise-aware scheduling becomes stronger when combined
#   with value-level encoding inside SN-selected candidate qv-blocks.
#
# Fair setting aligned with previous Qwen2.5 K10 results:
#   model              = Qwen2.5-14B-Instruct
#   dataset            = GSM8K
#   num_clients        = 50
#   clients_per_round  = 10
#   global_rounds      = 5
#   local_epochs       = 10
#   lora_dim           = 8
#   lr                 = 1e-4
#   partition_strategy = quantity
#   dirichlet_alpha    = 10
#   main budget        = 2112 packets/client/round = 12.5% of full upload cost
#   seed               = 42
#
# Default variants:
#   current_raw            : original Ours, raw qv-block upload
#   enc_compeft_m2/m3/m4   : SN-P1P2 candidate gate, then ComPEFT TopK+PQ
#   enc_flasc_m3           : SN-P1P2 candidate gate, then global TopK
#   depth_rank_compeft_m3  : depth-rank P1 instead of depth-balanced
#   rawp1_compeft_m3       : raw P1 instead of depth-balanced
#   flm_topk_fast          : fast capped FLM-TopK for runtime sanity check
#
# Run:
#   bash run_qwen25_gsm8k_ours_encoding_sweep.sh
#
# Overrides:
#   VARIANTS="enc_compeft_m3 enc_compeft_m4" bash run_qwen25_gsm8k_ours_encoding_sweep.sh
#   GPU_GROUPS="0,1,2,3 4,5,6,7" PARALLEL_JOBS=2 bash run_qwen25_gsm8k_ours_encoding_sweep.sh
# ============================================================

MODEL_PATH=${MODEL_PATH:-/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct}
DATA_DIR=${DATA_DIR:-/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data}
TASK_CONFIG_DIR=${TASK_CONFIG_DIR:-/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config}
ACC_CONFIG=${ACC_CONFIG:-script_accelerate/accelerate_config.yaml}
PY_SCRIPT=${PY_SCRIPT:-src/run_uie_lora.py}
OUT_ROOT=${OUT_ROOT:-results/Qwen2_gsm8k/ours_encoding_seed42}

GPU_GROUPS=${GPU_GROUPS:-"0,1,2,3 4,5,6,7"}
PARALLEL_JOBS=${PARALLEL_JOBS:-2}

SEED=${SEED:-42}
BUDGET=${BUDGET:-2112}
VARIANTS=${VARIANTS:-"enc_compeft_m2 enc_compeft_m3 enc_compeft_m4 enc_flasc_m3 depth_rank_compeft_m3 rawp1_compeft_m3 flm_topk_fast"}
RUN_CURRENT_RAW=${RUN_CURRENT_RAW:-false}
SKIP_FINISHED=${SKIP_FINISHED:-true}
AUTO_PACK=${AUTO_PACK:-false}

NUM_CLIENTS=${NUM_CLIENTS:-50}
CLIENTS_PER_ROUND=${CLIENTS_PER_ROUND:-10}
GLOBAL_ROUNDS=${GLOBAL_ROUNDS:-5}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-10}
DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-10}
PARTITION_STRATEGY=${PARTITION_STRATEGY:-quantity}
LR=${LR:-1e-4}
LORA_DIM=${LORA_DIM:-8}

TRAIN_BS=${TRAIN_BS:-16}
EVAL_BS=${EVAL_BS:-16}
GRAD_ACC=${GRAD_ACC:-2}
MAX_SOURCE_LENGTH=${MAX_SOURCE_LENGTH:-512}
MAX_TARGET_LENGTH=${MAX_TARGET_LENGTH:-50}
GENERATION_MAX_LENGTH=${GENERATION_MAX_LENGTH:-50}
LOGGING_STEPS=${LOGGING_STEPS:-2}

# Encoding / FLM knobs
SN_ENCODER_BIT=${SN_ENCODER_BIT:-18}
SN_ENCODER_MIN_BIT=${SN_ENCODER_MIN_BIT:-4}
SN_ENCODER_BLOCKS=${SN_ENCODER_BLOCKS:-192}
BASELINE_PACKET_NUM=${BASELINE_PACKET_NUM:-0}
BASELINE_BLOCKS=${BASELINE_BLOCKS:-192}
BASELINE_BIT=${BASELINE_BIT:-18}
BASELINE_MIN_BIT=${BASELINE_MIN_BIT:-4}
BASELINE_TOPK_METHOD=${BASELINE_TOPK_METHOD:-gradient}
BASELINE_FLM_OPT_MAX_ITER=${BASELINE_FLM_OPT_MAX_ITER:-40}
BASELINE_FLM_MAX_BLOCKS=${BASELINE_FLM_MAX_BLOCKS:-256}
BASELINE_FLM_DISABLE_OPTIM=${BASELINE_FLM_DISABLE_OPTIM:-false}

mkdir -p ${OUT_ROOT}/logs

# -------- Preflight --------
for p in "${MODEL_PATH}" "${DATA_DIR}" "${TASK_CONFIG_DIR}"; do
  if [ ! -e "$p" ]; then
    echo "[ERROR] Missing path: $p" >&2
    exit 1
  fi
done
if ! grep -q "sn_encoder_mode" src/run_uie_lora.py; then
  echo "[ERROR] src/run_uie_lora.py does not include sn_encoder_mode. Please replace code first." >&2
  exit 1
fi
if ! grep -q "SNEncoder" src/federated_uie_lora.py; then
  echo "[ERROR] src/federated_uie_lora.py does not include SNEncoder logic. Please replace code first." >&2
  exit 1
fi
if ! grep -q "flm_opt_max_iter" src/baseline_compressors.py; then
  echo "[ERROR] src/baseline_compressors.py does not include fast FLM knobs. Please replace code first." >&2
  exit 1
fi

choose_gpu_group() {
  local idx=$1
  read -r -a arr <<< "${GPU_GROUPS}"
  local n=${#arr[@]}
  echo "${arr[$((idx % n))]}"
}

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL_JOBS}" ]; do
    wait -n
  done
}

common_args() {
  local out_dir=$1
  local run_name=$2
  local budget=$3
  echo \
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
    --seed ${SEED} \
    --federated_seed ${SEED} \
    --bf16 True \
    --fp16 False \
    --gradient_checkpointing True \
    --use_fast_tokenizer True
}

run_variant() {
  local variant=$1
  local gpu_group=$2
  local port
  port=$(shuf -i25000-30000 -n1)

  local method="lora_origin"
  local out_dir=${OUT_ROOT}/${variant}_budget${BUDGET}_K${CLIENTS_PER_ROUND}_seed${SEED}
  local run_name=qwen25_gsm8k_${variant}_budget${BUDGET}_K${CLIENTS_PER_ROUND}_seed${SEED}
  local log_file=${OUT_ROOT}/logs/${run_name}.log

  local extra=""
  case ${variant} in
    current_raw)
      extra="--method lora_origin --upload_atomic_mode qv_block --upload_score_mode sn_p1p2 --sn_p1_norm_mode depth_balanced --sn_depth_group_ratios 1,1,2 --sn_gap_eta 0.0 --sn_force_full_budget True --sn_encoder_mode none"
      ;;
    enc_compeft_m2)
      extra="--method lora_origin --upload_atomic_mode qv_block --upload_score_mode sn_p1p2 --sn_p1_norm_mode depth_balanced --sn_depth_group_ratios 1,1,2 --sn_gap_eta 0.0 --sn_force_full_budget True --sn_encoder_mode compeft --sn_candidate_budget_multiplier 2 --sn_encoder_packet_num ${BUDGET} --sn_encoder_bit ${SN_ENCODER_BIT} --sn_encoder_min_bit ${SN_ENCODER_MIN_BIT} --sn_encoder_blocks ${SN_ENCODER_BLOCKS}"
      ;;
    enc_compeft_m3)
      extra="--method lora_origin --upload_atomic_mode qv_block --upload_score_mode sn_p1p2 --sn_p1_norm_mode depth_balanced --sn_depth_group_ratios 1,1,2 --sn_gap_eta 0.0 --sn_force_full_budget True --sn_encoder_mode compeft --sn_candidate_budget_multiplier 3 --sn_encoder_packet_num ${BUDGET} --sn_encoder_bit ${SN_ENCODER_BIT} --sn_encoder_min_bit ${SN_ENCODER_MIN_BIT} --sn_encoder_blocks ${SN_ENCODER_BLOCKS}"
      ;;
    enc_compeft_m4)
      extra="--method lora_origin --upload_atomic_mode qv_block --upload_score_mode sn_p1p2 --sn_p1_norm_mode depth_balanced --sn_depth_group_ratios 1,1,2 --sn_gap_eta 0.0 --sn_force_full_budget True --sn_encoder_mode compeft --sn_candidate_budget_multiplier 4 --sn_encoder_packet_num ${BUDGET} --sn_encoder_bit ${SN_ENCODER_BIT} --sn_encoder_min_bit ${SN_ENCODER_MIN_BIT} --sn_encoder_blocks ${SN_ENCODER_BLOCKS}"
      ;;
    enc_flasc_m3)
      extra="--method lora_origin --upload_atomic_mode qv_block --upload_score_mode sn_p1p2 --sn_p1_norm_mode depth_balanced --sn_depth_group_ratios 1,1,2 --sn_gap_eta 0.0 --sn_force_full_budget True --sn_encoder_mode flasc --sn_candidate_budget_multiplier 3 --sn_encoder_packet_num ${BUDGET}"
      ;;
    depth_rank_compeft_m3)
      extra="--method lora_origin --upload_atomic_mode qv_block --upload_score_mode sn_p1p2 --sn_p1_norm_mode depth_rank --sn_gap_eta 0.0 --sn_force_full_budget True --sn_encoder_mode compeft --sn_candidate_budget_multiplier 3 --sn_encoder_packet_num ${BUDGET} --sn_encoder_bit ${SN_ENCODER_BIT} --sn_encoder_min_bit ${SN_ENCODER_MIN_BIT} --sn_encoder_blocks ${SN_ENCODER_BLOCKS}"
      ;;
    rawp1_compeft_m3)
      extra="--method lora_origin --upload_atomic_mode qv_block --upload_score_mode sn_p1p2 --sn_p1_norm_mode raw --sn_gap_eta 0.0 --sn_force_full_budget True --sn_encoder_mode compeft --sn_candidate_budget_multiplier 3 --sn_encoder_packet_num ${BUDGET} --sn_encoder_bit ${SN_ENCODER_BIT} --sn_encoder_min_bit ${SN_ENCODER_MIN_BIT} --sn_encoder_blocks ${SN_ENCODER_BLOCKS}"
      ;;
    flm_topk_fast)
      method="flm_topk"
      extra="--method flm_topk --baseline_packet_num ${BASELINE_PACKET_NUM} --baseline_blocks ${BASELINE_BLOCKS} --baseline_bit ${BASELINE_BIT} --baseline_min_bit ${BASELINE_MIN_BIT} --baseline_topk_method ${BASELINE_TOPK_METHOD} --baseline_flm_opt_max_iter ${BASELINE_FLM_OPT_MAX_ITER} --baseline_flm_max_blocks ${BASELINE_FLM_MAX_BLOCKS} --baseline_flm_disable_optim ${BASELINE_FLM_DISABLE_OPTIM}"
      ;;
    *)
      echo "[ERROR] Unknown variant: ${variant}" >&2
      exit 1
      ;;
  esac

  if [ "${SKIP_FINISHED}" = "true" ] && [ -f "${out_dir}/all_results.json" ]; then
    if [[ "${variant}" == flm_topk_fast ]]; then
      [ -f "${out_dir}/baseline_compression_history.json" ] && { echo "[SKIP] ${variant}"; return 0; }
    else
      if [ "${variant}" = "current_raw" ]; then
        [ -f "${out_dir}/signal_noise_schedule_history.json" ] && { echo "[SKIP] ${variant}"; return 0; }
      else
        [ -f "${out_dir}/sn_encoder_compression_history.json" ] && { echo "[SKIP] ${variant}"; return 0; }
      fi
    fi
  fi

  echo "[RUN] variant=${variant}, gpu=${gpu_group}, out=${out_dir}"
  CUDA_VISIBLE_DEVICES=${gpu_group} accelerate launch --config_file ${ACC_CONFIG} \
    --main_process_port ${port} \
    ${PY_SCRIPT} \
    $(common_args ${out_dir} ${run_name} ${BUDGET}) \
    ${extra} 2>&1 | tee ${log_file}
}

job_idx=0
if [ "${RUN_CURRENT_RAW}" = "true" ]; then
  wait_for_slot
  run_variant current_raw "$(choose_gpu_group ${job_idx})" &
  job_idx=$((job_idx + 1))
fi

for v in ${VARIANTS}; do
  wait_for_slot
  run_variant ${v} "$(choose_gpu_group ${job_idx})" &
  job_idx=$((job_idx + 1))
done
wait

if [ "${AUTO_PACK}" = "true" ]; then
  pack_name=qwen25_gsm8k_ours_encoding_seed42_pack.zip
  rm -f ${pack_name}
  zip -r ${pack_name} \
    ${OUT_ROOT} \
    run_qwen25_gsm8k_ours_encoding_sweep.sh \
    src/baseline_compressors.py \
    src/run_uie_lora.py \
    src/federated_uie_lora.py \
    src/uie_trainer_lora.py \
    src/uie_collator.py \
    src/gsm8k/gsm8k_metrics.py \
    -x "*/adapter/*" "*/client_states/*" "*/checkpoint*" "*.safetensors" "*.bin"
  du -h ${pack_name}
fi
