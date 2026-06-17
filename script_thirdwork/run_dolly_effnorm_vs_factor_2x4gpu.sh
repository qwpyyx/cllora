#!/bin/bash
set -x

export PYTHONWARNINGS="ignore"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/qiuwenqi/.cache/huggingface
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

method=lora_origin
lora_rank=8
lamda_2=0
lamda_1=0
lr=1e-04
radius=0
di_alpha=0.5
seed=42

# Three budgets for the minimal method-validation experiment.
# 880: low budget; 2200: main budget; 3080: higher budget. 1
BUDGETS=(880 2200 3080)

data_dir=/home/qiuwenqi/LLM/Third_work/data_dolly/dolly_uie/data
task_config_dir=/home/qiuwenqi/LLM/Third_work/data_dolly/dolly_uie/task_config
model_path=/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct

ACC_CONFIG=script_accelerate/accelerate_config.yaml
PY_SCRIPT=src/run_uie_lora.py
OUT_ROOT=results/Qwen2_dolly/lora_exp_dolly_alpha05/effnorm_vs_factor_seed42_2x4gpu
mkdir -p ${OUT_ROOT}/logs

# Guard: this experiment requires the modified code supporting --upload_score_mode.
if ! grep -q "upload_score_mode" ${PY_SCRIPT}; then
  echo "ERROR: ${PY_SCRIPT} does not support --upload_score_mode."
  echo "Please copy the modified files first:"
  echo "  cp run_uie_lora_effective.py src/run_uie_lora.py"
  echo "  cp federated_uie_lora_effective.py src/federated_uie_lora.py"
  exit 1
fi

run_one() {
  local gpus=$1
  local score_mode=$2
  local budget=$3

  local port=$(shuf -i25000-30000 -n1)
  local out_dir=${OUT_ROOT}/budget_${budget}_${score_mode}_abpair_5client
  local run_name=dolly_${score_mode}_abpair_budget${budget}_seed42_5client
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
     --global_rounds 5 \
     --local_epochs 10 \
     --num_clients 50 \
     --clients_per_round 5 \
     --dirichlet_alpha ${di_alpha} \
     --partition_strategy label \
     --partition_label_key Dataset \
     --comm_budget ${budget} \
     --learning_rate ${lr} \
     --run_name ${run_name} \
     --max_source_length 512 \
     --max_target_length 128 \
     --generation_max_length 128 \
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
     --upload_score_mode ${score_mode} \
     --upload_diversity_mode none \
     --diagnose_pair_saliency False \
     --diagnose_residual_errors False \
     --lora_residual_accumulation False \
     > ${log_file} 2>&1
}

run_group() {
  local gpus=$1
  local score_mode=$2
  for budget in "${BUDGETS[@]}"; do
    run_one ${gpus} ${score_mode} ${budget}
    sleep 5
  done
}

# Run factor-norm baseline and effective-norm candidate in parallel.
# Group A uses GPUs 0,1,2,3; Group B uses GPUs 4,5,6,7.
run_group 0,1,2,3 factor_norm &
PID_A=$!

run_group 4,5,6,7 effective_norm &
PID_B=$!

wait ${PID_A}
wait ${PID_B}

echo "All Dolly factor_norm vs effective_norm experiments finished."
echo "Results are under: ${OUT_ROOT}"
