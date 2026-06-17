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

# =============================================================
# GSM8K Ours-qvblock + P1 normalization/depth-balance sweep
# Purpose:
#   Test whether P1 scale/depth normalization fixes the high-layer
#   concentration observed in raw qv-block Ours.
#
# Required code version:
#   src/run_uie_lora.py and src/federated_uie_lora.py must support:
#     --upload_atomic_mode qv_block
#     --upload_score_mode sn_p1p2
#     --sn_p1_norm_mode
#     --sn_depth_group_ratios
#
# Use:
#   cp run_uie_lora_sn_qvblock_norm.py src/run_uie_lora.py
#   cp federated_uie_lora_sn_qvblock_norm.py src/federated_uie_lora.py
#   bash run_gsm8k_ours_qvblock_norm_sweep_seed42.sh
# =============================================================

if ! grep -q "sn_p1_norm_mode" src/run_uie_lora.py; then
    echo "[ERROR] src/run_uie_lora.py does not support --sn_p1_norm_mode." >&2
    echo "Please copy run_uie_lora_sn_qvblock_norm.py to src/run_uie_lora.py first." >&2
    exit 1
fi
if ! grep -q "depth_balanced" src/federated_uie_lora.py; then
    echo "[ERROR] src/federated_uie_lora.py does not contain P1 normalization/depth-balance support." >&2
    echo "Please copy federated_uie_lora_sn_qvblock_norm.py to src/federated_uie_lora.py first." >&2
    exit 1
fi

# ===== Basic config =====
method=lora_origin
lora_rank=8
lamda_2=0
lamda_1=0
lr=${LR:-1e-04}
radius=0
seed=${SEED:-42}

di_alpha=${DIRICHLET_ALPHA:-10}
num_clients=${NUM_CLIENTS:-50}
clients_per_round=${CLIENTS_PER_ROUND:-10}
global_rounds=${GLOBAL_ROUNDS:-5}
local_epochs=${LOCAL_EPOCHS:-10}

# For Qwen2.5-14B qv_block, one qv block is usually about 352 packets.
# 1408/1760/2112 correspond roughly to 4/5/6 qv-blocks per client.
BUDGETS=${BUDGETS:-"1408 1760 2112"}
GAP_ETAS=${GAP_ETAS:-"0.0"}

# raw has already been run in the previous qv-block sweep.  Here we focus on
# P1 normalization variants:
#   depth_rank     : rank-normalize a_m and b_m inside lower/middle/upper groups;
#   depth_balanced : depth_rank + reserve P1 quota by lower/middle/upper ratios.
P1_NORM_MODES=${P1_NORM_MODES:-"depth_rank depth_balanced"}
DEPTH_GROUP_RATIOS=${DEPTH_GROUP_RATIOS:-"1,1,2"}

# ===== Paths =====
data_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data
task_config_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config
model_path=/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct
ACC_CONFIG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py

OUT_ROOT=results/Qwen2_gsm8k/ours_qvblock_norm_sweep_seed42
mkdir -p ${OUT_ROOT}/logs

GPU_GROUP_0=${GPU_GROUP_0:-0,1,2,3}
GPU_GROUP_1=${GPU_GROUP_1:-4,5,6,7}
PARALLEL_JOBS=${PARALLEL_JOBS:-2}

run_one() {
    local gpus=$1
    local budget=$2
    local gap_eta=$3
    local p1_norm_mode=$4
    local depth_ratios=$5

    local port
    port=$(shuf -i25000-30000 -n1)

    local gap_tag
    gap_tag=$(echo "${gap_eta}" | tr '.' 'p')
    local ratio_tag
    ratio_tag=$(echo "${depth_ratios}" | tr ',' '_')

    local out_dir=${OUT_ROOT}/ours_qvblock_${p1_norm_mode}_r${ratio_tag}_gap${gap_tag}_budget${budget}_K${clients_per_round}_seed${seed}
    local run_name=gsm8k_ours_qvblock_${p1_norm_mode}_r${ratio_tag}_gap${gap_tag}_budget${budget}_K${clients_per_round}_seed${seed}
    local log_file=${OUT_ROOT}/logs/${run_name}.log

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
       --upload_atomic_mode qv_block \
       --upload_score_mode sn_p1p2 \
       --sn_gap_eta ${gap_eta} \
       --sn_force_full_budget True \
       --sn_save_diagnostics True \
       --sn_p1_norm_mode ${p1_norm_mode} \
       --sn_depth_group_ratios ${depth_ratios} \
       --upload_diversity_mode none \
       --diagnose_pair_saliency False \
       --diagnose_residual_errors False \
       --lora_residual_accumulation False \
       > ${log_file} 2>&1
}

job_idx=0
pids=()
for norm_mode in ${P1_NORM_MODES}; do
  for budget in ${BUDGETS}; do
    for gap_eta in ${GAP_ETAS}; do
      if (( job_idx % 2 == 0 )); then
        gpus=${GPU_GROUP_0}
      else
        gpus=${GPU_GROUP_1}
      fi
      run_one ${gpus} ${budget} ${gap_eta} ${norm_mode} ${DEPTH_GROUP_RATIOS} &
      pids+=("$!")
      job_idx=$((job_idx + 1))

      if (( ${#pids[@]} >= PARALLEL_JOBS )); then
        for pid in "${pids[@]}"; do
          wait ${pid}
        done
        pids=()
      fi
    done
  done
done

for pid in "${pids[@]}"; do
  wait ${pid}
done

echo "[DONE] qv-block normalized/depth-balanced Ours sweep finished. Results saved to ${OUT_ROOT}"
