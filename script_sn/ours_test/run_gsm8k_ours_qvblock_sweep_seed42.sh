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
# GSM8K Ours-qvblock sweep
# Purpose:
#   Test the qv-block version of our signal-noise P1/P2 scheduler.
#   In qv_block mode, q_proj and v_proj LoRA A/B pairs in the same
#   transformer layer are uploaded together as one structured unit.
#
# Required code version:
#   src/run_uie_lora.py and src/federated_uie_lora.py must support:
#     --upload_atomic_mode qv_block
#     --upload_score_mode sn_p1p2
#   Use:
#     cp run_uie_lora_sn_qvblock.py src/run_uie_lora.py
#     cp federated_uie_lora_sn_qvblock.py src/federated_uie_lora.py
# =============================================================

if ! grep -q "qv_block" src/run_uie_lora.py; then
    echo "[ERROR] src/run_uie_lora.py does not support qv_block." >&2
    exit 1
fi
if ! grep -q "atomic_mode=ab_pair or qv_block" src/federated_uie_lora.py; then
    echo "[ERROR] src/federated_uie_lora.py does not contain qv-block sn_p1p2 fix." >&2
    echo "Please copy federated_uie_lora_sn_qvblock.py to src/federated_uie_lora.py first." >&2
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

# For Qwen2.5-14B qv_block, one qv block is usually about 352 packets
# (q pair + v pair). These budgets correspond roughly to 3/4/5/6 blocks.
BUDGETS=${BUDGETS:-"1056 1408 1760 2112"}
GAP_ETAS=${GAP_ETAS:-"0.0"}

# ===== Paths =====
data_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data
task_config_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config
model_path=/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct
ACC_CONFIG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py

OUT_ROOT=results/Qwen2_gsm8k/ours_qvblock_sweep_seed42
mkdir -p ${OUT_ROOT}/logs

GPU_GROUP_0=${GPU_GROUP_0:-0,1,2,3}
GPU_GROUP_1=${GPU_GROUP_1:-4,5,6,7}
PARALLEL_JOBS=${PARALLEL_JOBS:-2}

run_one() {
    local gpus=$1
    local budget=$2
    local gap_eta=$3

    local port
    port=$(shuf -i25000-30000 -n1)

    local gap_tag
    gap_tag=$(echo "${gap_eta}" | tr '.' 'p')

    local out_dir=${OUT_ROOT}/ours_qvblock_gap${gap_tag}_budget${budget}_K${clients_per_round}_seed${seed}
    local run_name=gsm8k_ours_qvblock_gap${gap_tag}_budget${budget}_K${clients_per_round}_seed${seed}
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
       --upload_diversity_mode none \
       --diagnose_pair_saliency False \
       --diagnose_residual_errors False \
       --lora_residual_accumulation False \
       > ${log_file} 2>&1
}

job_idx=0
pids=()
for budget in ${BUDGETS}; do
  for gap_eta in ${GAP_ETAS}; do
    if (( job_idx % 2 == 0 )); then
      gpus=${GPU_GROUP_0}
    else
      gpus=${GPU_GROUP_1}
    fi
    run_one ${gpus} ${budget} ${gap_eta} &
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

for pid in "${pids[@]}"; do
  wait ${pid}
done

echo "[DONE] qv-block Ours sweep finished. Results saved to ${OUT_ROOT}"
