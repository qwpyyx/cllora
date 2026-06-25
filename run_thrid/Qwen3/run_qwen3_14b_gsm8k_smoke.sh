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
# Qwen3-14B-Instruct + GSM8K smoke test
# Purpose:
#   1) Verify Qwen3-14B can be loaded by the current Qwen branch.
#   2) Verify tokenizer/pad/eos/generation/GSM8K extraction.
#   3) Verify qv-block Ours can build LoRA units and finish one FL round.
#
# Run in project root:
#   bash run_qwen3_14b_gsm8k_smoke.sh
#
# Override model path if needed:
#   MODEL_PATH=/path/to/Qwen3-14B-Instruct bash run_qwen3_14b_gsm8k_smoke.sh
#
# Run dense only:
#   RUN_OURS_SMOKE=false bash run_qwen3_14b_gsm8k_smoke.sh
# ==============================================================

MODEL_PATH=${MODEL_PATH:-/home/qiuwenqi/LLM/models/Qwen3-14B-Instruct}
DATA_DIR=${DATA_DIR:-/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data}
TASK_CONFIG_DIR=${TASK_CONFIG_DIR:-/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config}
ACC_CONFIG=${ACC_CONFIG:-script_accelerate/accelerate_config.yaml}
PY_SCRIPT=${PY_SCRIPT:-src/run_uie_lora.py}
OUT_ROOT=${OUT_ROOT:-results/Qwen3_14B_gsm8k/smoke_test}
GPUS=${GPUS:-0,1,2,3}
SEED=${SEED:-42}
RUN_DENSE_SMOKE=${RUN_DENSE_SMOKE:-true}
RUN_OURS_SMOKE=${RUN_OURS_SMOKE:-true}
SMOKE_BUDGET=${SMOKE_BUDGET:-2112}

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
if ! grep -q "is_qwen" src/federated_uie_lora.py; then
  echo "[ERROR] src/federated_uie_lora.py does not seem to contain Qwen branch." >&2
  exit 1
fi
if ! grep -q "qv_block" src/federated_uie_lora.py; then
  echo "[ERROR] src/federated_uie_lora.py does not seem to contain qv_block logic." >&2
  exit 1
fi

# Tokenizer/config-only check: cheap, does not load 14B weights.
python - <<PY
from transformers import AutoConfig, AutoTokenizer
path = "${MODEL_PATH}"
config = AutoConfig.from_pretrained(path)
tok = AutoTokenizer.from_pretrained(path, use_fast=True)
print("[Preflight] model_type=", getattr(config, "model_type", None))
print("[Preflight] hidden_size=", getattr(config, "hidden_size", None), "layers=", getattr(config, "num_hidden_layers", None))
print("[Preflight] heads=", getattr(config, "num_attention_heads", None), "kv_heads=", getattr(config, "num_key_value_heads", None))
print("[Preflight] bos/eos/pad=", tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
print("[Preflight] eos_token=", repr(tok.eos_token), "pad_token=", repr(tok.pad_token))
print("[Preflight] tokenizer class=", tok.__class__.__name__)
print("[Preflight] chat_template exists=", bool(getattr(tok, "chat_template", None)))
PY

run_one() {
  local mode=$1
  local port
  port=$(shuf -i25000-30000 -n1)

  local budget=0
  local out_dir=${OUT_ROOT}/${mode}_seed${SEED}
  local run_name=qwen3_14b_gsm8k_${mode}_seed${SEED}
  local log_file=${OUT_ROOT}/logs/${run_name}.log

  local atomic_args="--upload_atomic_mode ab_pair --upload_score_mode factor_norm"
  local sn_args=""

  if [ "${mode}" = "dense_smoke" ]; then
    budget=0
  elif [ "${mode}" = "ours_qv_smoke" ]; then
    budget=${SMOKE_BUDGET}
    atomic_args="--upload_atomic_mode qv_block --upload_score_mode sn_p1p2"
    sn_args="--sn_gap_eta 0.0 --sn_force_full_budget True --sn_save_diagnostics True --sn_p1_norm_mode depth_balanced --sn_depth_group_ratios 1,1,2"
  else
    echo "[ERROR] Unknown mode: ${mode}" >&2
    exit 1
  fi

  echo "[RUN] mode=${mode} budget=${budget} gpus=${GPUS} out=${out_dir}"

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
    --global_rounds 1 \
    --local_epochs 1 \
    --num_clients 10 \
    --clients_per_round 2 \
    --dirichlet_alpha 10 \
    --partition_strategy quantity \
    --comm_budget ${budget} \
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

  echo "[OK] ${mode} finished. Log: ${log_file}"
  if [ -f "${out_dir}/all_results.json" ]; then
    cat "${out_dir}/all_results.json"
  fi
}

if [ "${RUN_DENSE_SMOKE}" = "true" ]; then
  run_one dense_smoke
fi

if [ "${RUN_OURS_SMOKE}" = "true" ]; then
  run_one ours_qv_smoke
fi

echo "[DONE] Qwen3-14B GSM8K smoke test finished. Outputs: ${OUT_ROOT}"
