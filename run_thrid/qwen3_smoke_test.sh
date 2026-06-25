#!/bin/bash
# Qwen3 Smoke Test: verify algorithm + baselines work with Qwen3-14B
# Lightweight: 2 rounds, 10 clients, 2 local epochs
set -x

export PYTHONWARNINGS="ignore"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/qiuwenqi/.cache/huggingface
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

# ===== Use qwen3 Python explicitly =====
PY=/home/qiuwenqi/.conda/envs/qwen3/bin/python

# ===== Lightweight config =====
method=lora_origin
lora_rank=8
lr=1e-04
seed=42
di_alpha=10
num_clients=10
clients_per_round=4
global_rounds=2
local_epochs=2
com_budget=440

# ===== Qwen3 paths =====
data_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data
task_config_dir=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config
model_path=/home/qiuwenqi/LLM/models/Qwen3-14B-Instruct
ACC_CONFIG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py
OUT_ROOT=results/Qwen3_smoke_test

mkdir -p ${OUT_ROOT}/logs

run_one() {
    local gpus=$1
    local mode=$2
    local budget=$3
    local tag=$4

    local port
    port=$(shuf -i25000-30000 -n1)

    local budget_label=${budget}
    if [ "${budget}" = "-1" ] || [ "${budget}" = "0" ]; then
        budget_label=full
    fi

    local out_dir=${OUT_ROOT}/${mode}_budget${budget_label}_K${clients_per_round}_seed${seed}
    local run_name=qwen3smoke_${tag}
    local log_file=${OUT_ROOT}/logs/${run_name}.log

    local atomic_param=""
    local score_param=""
    local sn_params=""

    if [ "${mode}" = "tensor_topk" ]; then
        atomic_param="--upload_atomic_mode tensor"
        score_param="--upload_score_mode factor_norm"
    elif [ "${mode}" = "ours_sn_p1p2" ]; then
        atomic_param="--upload_atomic_mode ab_pair"
        score_param="--upload_score_mode sn_p1p2"
        sn_params="--sn_gap_eta 1.0 --sn_force_full_budget False --sn_save_diagnostics True"
    else
        echo "Unknown mode: ${mode}" >&2
        return 1
    fi

    echo "=== Starting ${tag} on GPUs ${gpus} at $(date) ==="
    CUDA_VISIBLE_DEVICES=${gpus} ${PY} -m accelerate.commands.launch \
       --config_file ${ACC_CONFIG} \
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
       --eval_strategy no \
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
       ${atomic_param} \
       ${score_param} \
       ${sn_params} \
       --upload_diversity_mode none \
       --diagnose_pair_saliency False \
       --diagnose_residual_errors False \
       --lora_residual_accumulation False \
       > ${log_file} 2>&1

    local ec=$?
    echo "=== ${tag} exit code: ${ec} at $(date) ==="
    return ${ec}
}

echo "============================================"
echo "  Qwen3 Smoke Test"
echo "  Python: $(${PY} --version)"
echo "  Model: ${model_path}"
echo "  Rounds: ${global_rounds} | Clients: ${num_clients} | Per-round: ${clients_per_round}"
echo "============================================"

# Run baseline (tensor_topk) on GPUs 0,1,2,3
if run_one 0,1,2,3 tensor_topk ${com_budget} "baseline_tensor_topk"; then
    echo "=== tensor_topk PASSED ==="
else
    echo "=== tensor_topk FAILED (exit $?) - check logs ==="
fi

echo ""

# Run ours (sn_p1p2) on GPUs 4,5,6,7
if run_one 4,5,6,7 ours_sn_p1p2 ${com_budget} "ours_sn_p1p2"; then
    echo "=== ours_sn_p1p2 PASSED ==="
else
    echo "=== ours_sn_p1p2 FAILED (exit $?) - check logs ==="
fi

echo ""
echo "============================================"
echo "  Qwen3 smoke test done! Logs: ${OUT_ROOT}/logs/"
echo "============================================"
