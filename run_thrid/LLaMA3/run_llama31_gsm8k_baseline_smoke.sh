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
# Llama-3.1-8B + GSM8K migrated-baseline smoke test
# Purpose:
#   1) Verify migrated baselines can run end-to-end in the current FedLoRA code.
#   2) Verify baseline_compression_history.json / summary json are saved.
#   3) Verify ComPEFT no longer depends on block-wise selection, while FLM-TopK does.
#
# Baselines tested by default:
#   flasc     = global TopK
#   compeft   = global TopK + PQ
#   flm_topk  = block_opt / FLM-TopK
#   fedcomp   = row-vector FedComp + residual
#
# Run in project root:
#   bash run_llama31_gsm8k_baseline_smoke.sh
#
# Optional overrides:
#   MODEL_PATH=/path/to/Llama-3.1-8B-Instruct bash run_llama31_gsm8k_baseline_smoke.sh
#   METHODS="flasc compeft" bash run_llama31_gsm8k_baseline_smoke.sh
#   BUDGET=1144 BASELINE_BLOCKS=192 bash run_llama31_gsm8k_baseline_smoke.sh
# ==============================================================

MODEL_PATH=${MODEL_PATH:-/home/qiuwenqi/LLM/models/Llama-3.1-8B-Instruct}
DATA_DIR=${DATA_DIR:-/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data}
TASK_CONFIG_DIR=${TASK_CONFIG_DIR:-/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config}
ACC_CONFIG=${ACC_CONFIG:-script_accelerate/accelerate_config.yaml}
PY_SCRIPT=${PY_SCRIPT:-src/run_uie_lora.py}
OUT_ROOT=${OUT_ROOT:-results/Llama31_gsm8k/baseline_smoke}
GPUS=${GPUS:-0,1,2,3}
SEED=${SEED:-42}

# Llama-3.1 formal result showed full_upload_cost=9152 and qv-block unit_cost=286.
# BUDGET=1144 corresponds to 12.5% of full upload and is used here as a matched packet budget.
BUDGET=${BUDGET:-1144}
METHODS=${METHODS:-"flasc compeft flm_topk fedcomp"}
RUN_DENSE_REF=${RUN_DENSE_REF:-false}
SKIP_FINISHED=${SKIP_FINISHED:-false}

# Baseline hyperparameters.
# ComPEFT ignores BASELINE_BLOCKS in the corrected code; FLM-TopK uses it.
BASELINE_PACKET_NUM=${BASELINE_PACKET_NUM:-0}   # 0 means reuse --comm_budget
BASELINE_BLOCKS=${BASELINE_BLOCKS:-192}
BASELINE_BIT=${BASELINE_BIT:-18}
BASELINE_MIN_BIT=${BASELINE_MIN_BIT:-4}
BASELINE_TOPK_METHOD=${BASELINE_TOPK_METHOD:-gradient}
FEDCOMP_USE_RESIDUAL=${FEDCOMP_USE_RESIDUAL:-true}

# Keep this small: the goal is code validation, not final performance.
SMOKE_NUM_CLIENTS=${SMOKE_NUM_CLIENTS:-10}
SMOKE_CLIENTS_PER_ROUND=${SMOKE_CLIENTS_PER_ROUND:-2}
SMOKE_GLOBAL_ROUNDS=${SMOKE_GLOBAL_ROUNDS:-1}
SMOKE_LOCAL_EPOCHS=${SMOKE_LOCAL_EPOCHS:-1}

# FedComp residual replay only triggers from the second communication round.
# Keep it on by default to test residual path as well.
FEDCOMP_TEST_RESIDUAL_REPLAY=${FEDCOMP_TEST_RESIDUAL_REPLAY:-true}
FEDCOMP_NUM_CLIENTS=${FEDCOMP_NUM_CLIENTS:-2}
FEDCOMP_CLIENTS_PER_ROUND=${FEDCOMP_CLIENTS_PER_ROUND:-2}
FEDCOMP_GLOBAL_ROUNDS=${FEDCOMP_GLOBAL_ROUNDS:-2}

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
  echo "        Please copy federated_uie_lora_migrated_baseline_history.py to src/federated_uie_lora.py first." >&2
  exit 1
fi
if ! grep -q "baseline_compression_history" src/federated_uie_lora.py; then
  echo "[ERROR] src/federated_uie_lora.py does not contain baseline_compression_history saving logic." >&2
  exit 1
fi
if ! grep -q "baseline_packet_num" src/run_uie_lora.py; then
  echo "[ERROR] src/run_uie_lora.py does not contain migrated baseline arguments." >&2
  echo "        Please copy run_uie_lora_migrated_baseline_history.py to src/run_uie_lora.py first." >&2
  exit 1
fi

python - <<PY
from transformers import AutoConfig, AutoTokenizer
path = "${MODEL_PATH}"
config = AutoConfig.from_pretrained(path)
tok = AutoTokenizer.from_pretrained(path, use_fast=True)
print("[Preflight] model_type=", getattr(config, "model_type", None))
print("[Preflight] hidden_size=", getattr(config, "hidden_size", None), "layers=", getattr(config, "num_hidden_layers", None))
print("[Preflight] bos/eos/pad=", tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
print("[Preflight] tokenizer class=", tok.__class__.__name__)
PY

run_one() {
  local method=$1
  local rounds=$2
  local num_clients=$3
  local clients_per_round=$4
  local port
  port=$(shuf -i25000-30000 -n1)

  local out_dir=${OUT_ROOT}/${method}_budget${BUDGET}_rounds${rounds}_seed${SEED}
  local run_name=llama31_gsm8k_${method}_baseline_smoke_budget${BUDGET}_rounds${rounds}_seed${SEED}
  local log_file=${OUT_ROOT}/logs/${run_name}.log

  if [ "${SKIP_FINISHED}" = "true" ] && [ -f "${out_dir}/all_results.json" ]; then
    if [ "${method}" = "dense_ref" ] || [ -f "${out_dir}/baseline_compression_history.json" ]; then
      echo "[SKIP] ${method} already finished: ${out_dir}"
      return 0
    fi
  fi

  local method_arg="${method}"
  local budget_arg="${BUDGET}"
  local extra_baseline_args="--baseline_packet_num ${BASELINE_PACKET_NUM} --baseline_blocks ${BASELINE_BLOCKS} --baseline_bit ${BASELINE_BIT} --baseline_min_bit ${BASELINE_MIN_BIT} --baseline_topk_method ${BASELINE_TOPK_METHOD} --fedcomp_use_residual ${FEDCOMP_USE_RESIDUAL}"

  if [ "${method}" = "dense_ref" ]; then
    method_arg="lora_origin"
    budget_arg="0"
    extra_baseline_args=""
  fi

  CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch --config_file ${ACC_CONFIG} \
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
    --global_rounds ${rounds} \
    --local_epochs ${SMOKE_LOCAL_EPOCHS} \
    --num_clients ${num_clients} \
    --clients_per_round ${clients_per_round} \
    --dirichlet_alpha 10 \
    --partition_strategy quantity \
    --comm_budget ${budget_arg} \
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
    --logging_steps 1 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 150 \
    --lamda_1 0 \
    --lamda_2 0 \
    --federated_seed ${SEED} \
    --method ${method_arg} \
    --task 1 \
    --radius 0 \
    --gradient_checkpointing True \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --upload_diversity_mode none \
    --diagnose_pair_saliency False \
    --diagnose_residual_errors False \
    --lora_residual_accumulation False \
    ${extra_baseline_args} \
    > ${log_file} 2>&1

  echo "[OK] ${method} finished. Log: ${log_file}"
  if [ -f "${out_dir}/all_results.json" ]; then
    echo "[all_results] ${out_dir}/all_results.json"
    cat "${out_dir}/all_results.json"
  fi
  if [ "${method}" != "dense_ref" ]; then
    if [ ! -f "${out_dir}/baseline_compression_history.json" ]; then
      echo "[ERROR] ${method} finished but baseline_compression_history.json was not found in ${out_dir}" >&2
      exit 1
    fi
    if [ ! -f "${out_dir}/baseline_compression_summary.json" ]; then
      echo "[ERROR] ${method} finished but baseline_compression_summary.json was not found in ${out_dir}" >&2
      exit 1
    fi
    echo "[baseline_summary] ${out_dir}/baseline_compression_summary.json"
    cat "${out_dir}/baseline_compression_summary.json"
  fi
}

if [ "${RUN_DENSE_REF}" = "true" ]; then
  run_one dense_ref ${SMOKE_GLOBAL_ROUNDS} ${SMOKE_NUM_CLIENTS} ${SMOKE_CLIENTS_PER_ROUND}
fi

for m in ${METHODS}; do
  if [ "${m}" = "fedcomp" ] && [ "${FEDCOMP_TEST_RESIDUAL_REPLAY}" = "true" ]; then
    run_one ${m} ${FEDCOMP_GLOBAL_ROUNDS} ${FEDCOMP_NUM_CLIENTS} ${FEDCOMP_CLIENTS_PER_ROUND}
  else
    run_one ${m} ${SMOKE_GLOBAL_ROUNDS} ${SMOKE_NUM_CLIENTS} ${SMOKE_CLIENTS_PER_ROUND}
  fi
done

python - <<PY
import glob, json, os
out_root = "${OUT_ROOT}"
rows = []
for d in sorted(glob.glob(os.path.join(out_root, "*"))):
    if not os.path.isdir(d) or os.path.basename(d) == "logs":
        continue
    row = {"dir": d, "name": os.path.basename(d)}
    p = os.path.join(d, "all_results.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            ar = json.load(f)
        row.update({
            "predict_gsm8k_em": ar.get("predict_gsm8k_em"),
            "predict_gen_len": ar.get("predict_gen_len"),
            "full_upload_cost": ar.get("full_upload_cost"),
        })
    sp = os.path.join(d, "baseline_compression_summary.json")
    if os.path.exists(sp):
        with open(sp, "r", encoding="utf-8") as f:
            bs = json.load(f)
        row["baseline_summary"] = bs
    rows.append(row)
summary_path = os.path.join(out_root, "baseline_smoke_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print("[DONE] Wrote", summary_path)
for r in rows:
    print(r.get("name"), "EM=", r.get("predict_gsm8k_em"), "gen_len=", r.get("predict_gen_len"), "full_cost=", r.get("full_upload_cost"))
    if "baseline_summary" in r:
        print("  baseline_summary=", r["baseline_summary"])
PY

echo "[DONE] Migrated baseline smoke test finished. Outputs: ${OUT_ROOT}"
