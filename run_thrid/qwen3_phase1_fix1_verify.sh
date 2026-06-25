#!/bin/bash
set -euo pipefail
set -x
# Quick re-run: flasc, compeft, ours for seed28 after extractor fix

export PYTHONWARNINGS="ignore"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/qiuwenqi/.cache/huggingface
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

PY=/home/qiuwenqi/.conda/envs/qwen3/bin/python
ACCELERATE="${PY} -m accelerate.commands.launch"
MODEL_PATH=/home/qiuwenqi/LLM/models/Qwen3-14B-Instruct
DATA_DIR=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data
TASK_CONFIG_DIR=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config
ACC_CONFIG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py
OUT_ROOT=results/Qwen3_phase1_fix1
SEED=28
BUDGET=1760

mkdir -p ${OUT_ROOT}/logs

run_one() {
    local method=$1
    local gpus=$2
    local port=$(shuf -i25000-30000 -n1)

    local out_dir=${OUT_ROOT}/${method}_seed${SEED}
    local log_file=${OUT_ROOT}/logs/${method}_seed${SEED}.log

    echo "[RUN] ${method} seed=${SEED} gpus=${gpus}"

    local extra_args=""
    if [ "${method}" = "ours" ]; then
        extra_args="--upload_atomic_mode qv_block --upload_score_mode sn_p1p2 --sn_p1_norm_mode depth_balanced --sn_depth_group_ratios 1,1,2 --sn_gap_eta 1.0 --sn_force_full_budget False --sn_save_diagnostics True"
        method="lora_origin"
    fi

    CUDA_VISIBLE_DEVICES=${gpus} ${ACCELERATE} \
       --config_file ${ACC_CONFIG} --main_process_port ${port} \
       ${PY_SCRIPT} \
       --report_to none --do_train --do_predict --predict_with_generate \
       --lora_dim 8 --model_name_or_path ${MODEL_PATH} \
       --data_dir ${DATA_DIR} --task_config_dir ${TASK_CONFIG_DIR} \
       --output_dir ${out_dir} \
       --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
       --gradient_accumulation_steps 2 \
       --global_rounds 5 --local_epochs 10 \
       --num_clients 50 --clients_per_round 10 \
       --dirichlet_alpha 10 --partition_strategy quantity \
       --comm_budget ${BUDGET} --learning_rate 1e-04 \
       --run_name qwen3_fix1_${method} \
       --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
       --add_task_name False --add_dataset_name False \
       --overwrite_output_dir --overwrite_cache \
       --lr_scheduler_type constant --warmup_steps 0 \
       --logging_strategy steps --logging_steps 2 \
       --evaluation_strategy no --save_strategy no --save_steps 150 \
       --lamda_1 0 --lamda_2 0 --federated_seed ${SEED} \
       --method ${method} --task 1 --radius 0 \
       --gradient_checkpointing True --bf16 True \
       --ddp_find_unused_parameters False \
       --baseline_packet_num 0 --baseline_blocks 192 --baseline_bit 18 --baseline_min_bit 4 \
       --baseline_topk_method gradient --fedcomp_use_residual True \
       ${extra_args} \
       --upload_diversity_mode none \
       --diagnose_pair_saliency False --diagnose_residual_errors False \
       --lora_residual_accumulation False \
       > ${log_file} 2>&1

    local ec=$?
    if [ -f "${out_dir}/all_results.json" ]; then
        em=$(grep -o '"predict_gsm8k_em": [0-9.]*' "${out_dir}/all_results.json")
        echo "[DONE] ${method} exit=${ec} ${em}"
    else
        echo "[FAIL] ${method} exit=${ec}"
    fi
}

# Run sequentially for clean results
run_one flasc "0,1,2,3"
run_one compeft "4,5,6,7"
run_one ours "0,1,2,3"

echo "=== Fix verification done ==="
for m in flasc compeft ours; do
    f=${OUT_ROOT}/${m}_seed${SEED}/all_results.json
    if [ -f "$f" ]; then
        echo -n "[${m}]: "
        grep '"predict_gsm8k_em"' "$f" | head -1
    fi
done
