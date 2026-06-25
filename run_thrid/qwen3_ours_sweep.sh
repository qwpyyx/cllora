#!/bin/bash
set -euo pipefail
set -x
# Ours parameter sweep: find working config for Qwen3-14B

export PYTHONWARNINGS="ignore"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

PY=/home/qiuwenqi/.conda/envs/qwen3/bin/python
ACCEL="${PY} -m accelerate.commands.launch"
MODEL=/home/qiuwenqi/LLM/models/Qwen3-14B-Instruct
DATA_DIR=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data
TASK_DIR=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config
ACC_CFG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py
OUT=results/Qwen3_ours_sweep
SEED=28
BUDGET=1760

COMMON="--report_to none --do_train --do_predict --predict_with_generate \
 --lora_dim 8 --model_name_or_path ${MODEL} --data_dir ${DATA_DIR} \
 --task_config_dir ${TASK_DIR} --per_device_train_batch_size 16 \
 --per_device_eval_batch_size 16 --gradient_accumulation_steps 2 \
 --global_rounds 5 --local_epochs 10 --num_clients 50 --clients_per_round 10 \
 --dirichlet_alpha 10 --partition_strategy quantity --comm_budget ${BUDGET} \
 --learning_rate 1e-04 --max_source_length 512 --max_target_length 50 \
 --generation_max_length 50 --add_task_name False --add_dataset_name False \
 --overwrite_output_dir --overwrite_cache --lr_scheduler_type constant \
 --warmup_steps 0 --logging_strategy steps --logging_steps 2 \
 --evaluation_strategy no --save_strategy no --save_steps 150 \
 --lamda_1 0 --lamda_2 0 --federated_seed ${SEED} --method lora_origin \
 --task 1 --radius 0 --gradient_checkpointing True --bf16 True \
 --ddp_find_unused_parameters False \
 --upload_atomic_mode qv_block --upload_score_mode sn_p1p2 \
 --sn_save_diagnostics True \
 --upload_diversity_mode none --diagnose_pair_saliency False \
 --diagnose_residual_errors False --lora_residual_accumulation False"

mkdir -p ${OUT}/logs

run_ours() {
    local tag=$1; local gpus=$2; shift 2
    local port=$(shuf -i25000-30000 -n1)
    local out_dir=${OUT}/${tag}_seed${SEED}
    local log=${OUT}/logs/${tag}.log

    echo "[RUN] ${tag} gpus=${gpus}"
    CUDA_VISIBLE_DEVICES=${gpus} ${ACCEL} --config_file ${ACC_CFG} \
       --main_process_port ${port} ${PY_SCRIPT} ${COMMON} \
       --output_dir ${out_dir} --run_name qwen3_ours_${tag} \
       "$@" > ${log} 2>&1

    if [ -f "${out_dir}/all_results.json" ]; then
        local em=$(grep -o '"predict_gsm8k_em": [0-9.]*' "${out_dir}/all_results.json")
        echo "[DONE] ${tag} ${em}"
    else
        echo "[FAIL] ${tag}"
    fi
}

# Config 1: Default (depth_balanced, gap_eta=1.0, no force) — baseline
run_ours "db_gap1_noforce" "0,1,2,3" \
    --sn_p1_norm_mode depth_balanced --sn_depth_group_ratios 1,1,2 \
    --sn_gap_eta 1.0 --sn_force_full_budget False

# Config 2: depth_balanced, gap_eta=0, no force (plain P2-L)
run_ours "db_gap0_noforce" "0,1,2,3" \
    --sn_p1_norm_mode depth_balanced --sn_depth_group_ratios 1,1,2 \
    --sn_gap_eta 0.0 --sn_force_full_budget False

# Config 3: depth_balanced, gap_eta=0, force=True (fill budget)
run_ours "db_gap0_force" "0,1,2,3" \
    --sn_p1_norm_mode depth_balanced --sn_depth_group_ratios 1,1,2 \
    --sn_gap_eta 0.0 --sn_force_full_budget True

# Config 4: depth_rank, gap_eta=1.0, no force
run_ours "dr_gap1_noforce" "0,1,2,3" \
    --sn_p1_norm_mode depth_rank \
    --sn_gap_eta 1.0 --sn_force_full_budget False

# Summary
echo "=== Ours Sweep Results ==="
for tag in db_gap1_noforce db_gap0_noforce db_gap0_force dr_gap1_noforce; do
    f=${OUT}/${tag}_seed${SEED}/all_results.json
    if [ -f "$f" ]; then
        em=$(grep -o '"predict_gsm8k_em": [0-9.]*' "$f")
        echo "  ${tag}: ${em}"
    else
        echo "  ${tag}: MISSING"
    fi
done
