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

# ============================================================
# Qwen2.5-14B + GSM8K migrated mature baseline test
#
# Purpose:
#   Run mature baselines on the same setting as previous Qwen2.5 K10 results.
#   This is a single-seed check first, not a full multi-seed run.
#
# Fair setting aligned with previous Qwen2.5 key results:
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
#   generation length  = 50  (keep Qwen2.5 protocol unchanged)
#   main budget        = 2112 packets/client/round = 12.5% of full upload cost 16896
#   seed               = 42
#
# Baselines:
#   flasc     = global TopK
#   compeft   = global TopK + PQ, no blocks
#   flm_topk  = FLM-TopK / block_opt, uses BASELINE_BLOCKS
#   fedcomp   = row-vector selection + residual replay
#
# Default 8-GPU plan:
#   two 4-GPU jobs in parallel.
#
# Run:
#   bash run_qwen25_gsm8k_baselines_seed42.sh
#
# Common overrides:
#   METHODS="flasc compeft" bash run_qwen25_gsm8k_baselines_seed42.sh
#   RUN_BUDGET_SWEEP=true bash run_qwen25_gsm8k_baselines_seed42.sh
#   GPU_GROUPS="0,1 2,3 4,5 6,7" PARALLEL_JOBS=4 bash run_qwen25_gsm8k_baselines_seed42.sh
# ============================================================

MODEL_PATH=${MODEL_PATH:-/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct}
DATA_DIR=${DATA_DIR:-/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data}
TASK_CONFIG_DIR=${TASK_CONFIG_DIR:-/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config}
ACC_CONFIG=${ACC_CONFIG:-script_accelerate/accelerate_config.yaml}
PY_SCRIPT=${PY_SCRIPT:-src/run_uie_lora.py}
OUT_ROOT=${OUT_ROOT:-results/Qwen2_gsm8k/baselines_seed42}

GPU_GROUPS=${GPU_GROUPS:-"0,1,2,3 4,5,6,7"}
PARALLEL_JOBS=${PARALLEL_JOBS:-2}

METHODS=${METHODS:-"flasc compeft flm_topk fedcomp"}
SEED=${SEED:-42}
MAIN_BUDGET=${MAIN_BUDGET:-2112}
RUN_MAIN=${RUN_MAIN:-true}

# Optional single-seed budget sweep. Default false to first check the main 12.5% setting.
RUN_BUDGET_SWEEP=${RUN_BUDGET_SWEEP:-false}
SWEEP_BUDGETS=${SWEEP_BUDGETS:-"1056 1408 1760 2112 3168"}

SKIP_FINISHED=${SKIP_FINISHED:-true}
AUTO_PACK=${AUTO_PACK:-false}

# FL settings: aligned with previous Qwen2.5 K10 results.
NUM_CLIENTS=${NUM_CLIENTS:-50}
CLIENTS_PER_ROUND=${CLIENTS_PER_ROUND:-10}
GLOBAL_ROUNDS=${GLOBAL_ROUNDS:-5}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-10}
DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-10}
PARTITION_STRATEGY=${PARTITION_STRATEGY:-quantity}
LR=${LR:-1e-4}
LORA_DIM=${LORA_DIM:-8}

# Keep Qwen2.5 training protocol. Override if OOM.
TRAIN_BS=${TRAIN_BS:-16}
EVAL_BS=${EVAL_BS:-16}
GRAD_ACC=${GRAD_ACC:-2}
MAX_SOURCE_LENGTH=${MAX_SOURCE_LENGTH:-512}
MAX_TARGET_LENGTH=${MAX_TARGET_LENGTH:-50}
GENERATION_MAX_LENGTH=${GENERATION_MAX_LENGTH:-50}
LOGGING_STEPS=${LOGGING_STEPS:-2}

# Baseline hyperparameters.
# baseline_packet_num=0 means use --comm_budget as packet budget.
# ComPEFT ignores BASELINE_BLOCKS in the corrected code; FLM-TopK uses it.
BASELINE_PACKET_NUM=${BASELINE_PACKET_NUM:-0}
BASELINE_BLOCKS=${BASELINE_BLOCKS:-192}
BASELINE_BIT=${BASELINE_BIT:-18}
BASELINE_MIN_BIT=${BASELINE_MIN_BIT:-4}
BASELINE_TOPK_METHOD=${BASELINE_TOPK_METHOD:-gradient}
FEDCOMP_USE_RESIDUAL=${FEDCOMP_USE_RESIDUAL:-true}

mkdir -p ${OUT_ROOT}/logs

# -------- Preflight --------
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
if [ ! -f "src/baseline_compressors.py" ]; then
  echo "[ERROR] Missing src/baseline_compressors.py" >&2
  exit 1
fi
if ! grep -q "BaselineCompressor" src/federated_uie_lora.py; then
  echo "[ERROR] src/federated_uie_lora.py does not seem to call BaselineCompressor." >&2
  exit 1
fi
if ! grep -q "baseline_compression_history" src/federated_uie_lora.py; then
  echo "[ERROR] src/federated_uie_lora.py does not contain baseline_compression_history saving logic." >&2
  exit 1
fi
if ! grep -q "baseline_packet_num" src/run_uie_lora.py; then
  echo "[ERROR] src/run_uie_lora.py does not contain migrated baseline arguments." >&2
  exit 1
fi
if ! grep -q "flm_topk" src/uie_trainer_lora.py || ! grep -q "fedcomp" src/uie_trainer_lora.py; then
  echo "[ERROR] src/uie_trainer_lora.py may not include migrated baselines in standard local train methods." >&2
  exit 1
fi

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
  read -r -a arr <<< "${GPU_GROUPS}"
  local n=${#arr[@]}
  echo "${arr[$((idx % n))]}"
}

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL_JOBS}" ]; do
    wait -n
  done
}

run_one() {
  local method=$1
  local seed=$2
  local budget=$3
  local gpu_group=$4
  local port
  port=$(shuf -i25000-30000 -n1)

  local out_dir=${OUT_ROOT}/${method}_budget${budget}_K${CLIENTS_PER_ROUND}_seed${seed}
  local run_name=qwen25_gsm8k_${method}_baseline_budget${budget}_K${CLIENTS_PER_ROUND}_seed${seed}
  local log_file=${OUT_ROOT}/logs/${run_name}.log

  if [ "${SKIP_FINISHED}" = "true" ] && [ -f "${out_dir}/all_results.json" ] && [ -f "${out_dir}/baseline_compression_history.json" ]; then
    echo "[SKIP] ${method}, budget=${budget}, seed=${seed}: ${out_dir}"
    return 0
  fi

  echo "[RUN] method=${method}, seed=${seed}, budget=${budget}, gpu=${gpu_group}, out=${out_dir}"

  CUDA_VISIBLE_DEVICES=${gpu_group} accelerate launch --config_file ${ACC_CONFIG} \
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
    --baseline_packet_num ${BASELINE_PACKET_NUM} \
    --baseline_blocks ${BASELINE_BLOCKS} \
    --baseline_bit ${BASELINE_BIT} \
    --baseline_min_bit ${BASELINE_MIN_BIT} \
    --baseline_topk_method ${BASELINE_TOPK_METHOD} \
    --fedcomp_use_residual ${FEDCOMP_USE_RESIDUAL} \
    --upload_diversity_mode none \
    --diagnose_pair_saliency False \
    --diagnose_residual_errors False \
    --lora_residual_accumulation False \
    > ${log_file} 2>&1

  echo "[OK] ${run_name} finished. Log: ${log_file}"
  if [ -f "${out_dir}/all_results.json" ]; then
    cat "${out_dir}/all_results.json"
  fi
  if [ ! -f "${out_dir}/baseline_compression_history.json" ]; then
    echo "[ERROR] ${method} finished but missing baseline_compression_history.json in ${out_dir}" >&2
    exit 1
  fi
  if [ ! -f "${out_dir}/baseline_compression_summary.json" ]; then
    echo "[ERROR] ${method} finished but missing baseline_compression_summary.json in ${out_dir}" >&2
    exit 1
  fi
}

job_idx=0

if [ "${RUN_MAIN}" = "true" ]; then
  echo "[INFO] Running Qwen2.5 baseline comparison: seed=${SEED}, budget=${MAIN_BUDGET}, methods=${METHODS}"
  for method in ${METHODS}; do
    wait_for_slot
    gpu=$(choose_gpu_group ${job_idx})
    run_one ${method} ${SEED} ${MAIN_BUDGET} ${gpu} &
    job_idx=$((job_idx + 1))
  done
fi

if [ "${RUN_BUDGET_SWEEP}" = "true" ]; then
  echo "[INFO] Running Qwen2.5 seed${SEED} budget sweep: budgets=${SWEEP_BUDGETS}, methods=${METHODS}"
  for budget in ${SWEEP_BUDGETS}; do
    for method in ${METHODS}; do
      wait_for_slot
      gpu=$(choose_gpu_group ${job_idx})
      run_one ${method} ${SEED} ${budget} ${gpu} &
      job_idx=$((job_idx + 1))
    done
  done
fi

wait

# -------- Build lightweight summary --------
python - <<PY
import glob, json, os
out_root = "${OUT_ROOT}"
rows = []
for d in sorted(glob.glob(os.path.join(out_root, "*"))):
    if not os.path.isdir(d) or os.path.basename(d) == "logs":
        continue
    ar = os.path.join(d, "all_results.json")
    if not os.path.exists(ar):
        continue
    with open(ar, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    summ_path = os.path.join(d, "baseline_compression_summary.json")
    hist_path = os.path.join(d, "baseline_compression_history.json")
    comp_summary = None
    if os.path.exists(summ_path):
        with open(summ_path, "r", encoding="utf-8") as f:
            comp_summary = json.load(f)
    rows.append({
        "run_dir": d,
        "run_name": os.path.basename(d),
        "predict_gsm8k_em": all_results.get("predict_gsm8k_em"),
        "predict_gen_len": all_results.get("predict_gen_len"),
        "full_upload_cost": all_results.get("full_upload_cost"),
        "baseline_compression_summary": comp_summary,
        "has_history": os.path.exists(hist_path),
    })

out = os.path.join(out_root, "baseline_qwen25_seed42_summary.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)
print(f"[SUMMARY] wrote {out} with {len(rows)} runs")
print("run_name\tEM\tGenLen")
for r in rows:
    print(f"{r['run_name']}\t{r['predict_gsm8k_em']}\t{r['predict_gen_len']}")
PY

if [ "${AUTO_PACK}" = "true" ]; then
  PACK_NAME=${PACK_NAME:-qwen25_gsm8k_baselines_seed42_pack.zip}
  rm -f ${PACK_NAME}
  rm -f /tmp/qwen25_baseline_seed42_pack_files.txt
  find ${OUT_ROOT} \
    -type f \
    \( -name "*.log" -o -name "*.json" -o -name "*.jsonl" -o -name "*.txt" \) \
    ! -path "*/adapter/*" \
    ! -path "*/client_states/*" \
    ! -path "*/checkpoint*" \
    ! -path "*/cache/*" \
    ! -name "*.safetensors" \
    ! -name "*.bin" \
    > /tmp/qwen25_baseline_seed42_pack_files.txt
  printf "%s\n" \
    run_qwen25_gsm8k_baselines_seed42.sh \
    src/baseline_compressors.py \
    src/run_uie_lora.py \
    src/federated_uie_lora.py \
    src/uie_trainer_lora.py \
    src/uie_collator.py \
    src/gsm8k/gsm8k_metrics.py \
    >> /tmp/qwen25_baseline_seed42_pack_files.txt
  zip -@ ${PACK_NAME} < /tmp/qwen25_baseline_seed42_pack_files.txt
  du -h ${PACK_NAME}
fi
