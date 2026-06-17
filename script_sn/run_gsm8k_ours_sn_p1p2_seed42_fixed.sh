#!/bin/bash
set -euo pipefail
set -x

export PYTHONWARNINGS="ignore"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/qiuwenqi/.cache/huggingface
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

# =============================================================
# GSM8K: Ours only
#   Ours = A/B-pair atomic upload + signal-noise P1 allocation
#          + gap-aware P2 assignment.
#
# Before running, copy the fixed Python files into src/:
#   cp run_uie_lora_sn_fixed.py src/run_uie_lora.py
#   cp federated_uie_lora_sn_fixed.py src/federated_uie_lora.py
#
# This script only runs ours_sn_p1p2 because the other baselines
# have already been completed.
# ============================================================

if ! grep -q "sn_p1p2" src/run_uie_lora.py; then
    echo "[ERROR] src/run_uie_lora.py does not contain sn_p1p2 arguments." >&2
    echo "Please copy run_uie_lora_sn_fixed.py to src/run_uie_lora.py first." >&2
    exit 1
fi

if ! grep -q "greedy_generalized_assignment_heterogeneous_cost" src/federated_uie_lora.py; then
    echo "[ERROR] src/federated_uie_lora.py does not contain the heterogeneous-cost SN-P1P2 fix." >&2
    echo "Please copy federated_uie_lora_sn_fixed.py to src/federated_uie_lora.py first." >&2
    exit 1
fi

# ===== Basic config =====
method=lora_origin
lora_rank=8
lamda_2=0
lamda_1=0
lr=1e-04
radius=0
seed=42

# GSM8K setting, kept consistent with previous runs.
di_alpha=10
num_clients=50
clients_per_round=20
global_rounds=5
local_epochs=10

# Sparse per-client packet budget.
com_budget=2200

# ===== Paths =====
data_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data
task_config_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config
model_path=/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct
ACC_CONFIG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py

OUT_ROOT=results/Qwen2_gsm8k/main5_sn_seed42
mkdir -p ${OUT_ROOT}/logs

mode=ours_sn_p1p2
budget=${com_budget}
gpus=${CUDA_VISIBLE_DEVICES_OVERRIDE:-0,1,2,3}
port=$(shuf -i25000-30000 -n1)

out_dir=${OUT_ROOT}/${mode}_budget${budget}_K${clients_per_round}_seed${seed}
run_name=gsm8k_${mode}_budget${budget}_K${clients_per_round}_seed${seed}
log_file=${OUT_ROOT}/logs/${run_name}.log

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
   --comm_budget ${budget} \
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
   --upload_atomic_mode ab_pair \
   --upload_score_mode sn_p1p2 \
   --sn_gap_eta 1.0 \
   --sn_force_full_budget False \
   --sn_save_diagnostics True \
   --upload_diversity_mode none \
   --diagnose_pair_saliency False \
   --diagnose_residual_errors False \
   --lora_residual_accumulation False \
   > ${log_file} 2>&1

echo "Ours GSM8K run finished. Output: ${out_dir}"
echo "Log: ${log_file}"
