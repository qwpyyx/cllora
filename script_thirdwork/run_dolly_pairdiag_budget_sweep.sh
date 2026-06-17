#!/bin/bash
set -x
set -e

export PYTHONWARNINGS="ignore"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/qiuwenqi/.cache/huggingface
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

# -----------------------------
# Basic settings
# -----------------------------
CUDA_DEVICES=4,5,6,7
ACCEL_CONFIG=script_accelerate/accelerate_config.yaml

method=lora_origin
lora_rank=8
lamda_2=0
lamda_1=0
lr=1e-04
radius=0
is_random_select=False
di_alpha=0.5
seed=42

# Budget sweep. 2200 is your previous main setting.
# Since one A/B pair is often around 220 packet-cost in your current setup,
# these roughly correspond to 2/4/6/8/10/14 selected pairs, but the actual
# number can vary because q/v modules may have different costs.
BUDGETS=(440 880 1320 1760 2200 3080)

# Diagnostic mode: prediction is not required for pair-saliency motivation.
# Set DO_PREDICT=True if you also want final Dolly metrics for every budget.
DO_PREDICT=False

pair_scales="0.25,0.5,2,4"

base_out=results/Qwen2_dolly/lora_exp_dolly_alpha05/pairdiag_budget_sweep_seed${seed}
data_dir=/home/qiuwenqi/LLM/Third_work/data_dolly/dolly_uie/data
task_config_dir=/home/qiuwenqi/LLM/Third_work/data_dolly/dolly_uie/task_config
model_path=/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct

if [ "$is_random_select" = "True" ]; then
    random_layer_param="--random_layer_selection True"
else
    random_layer_param=""
fi

if [ "$DO_PREDICT" = "True" ]; then
    predict_param="--do_predict --predict_with_generate"
else
    predict_param=""
fi

# Make sure the pairdiag code has already been copied into src/.
if ! grep -q "diagnose_pair_saliency" src/run_uie_lora.py; then
    echo "ERROR: src/run_uie_lora.py does not contain diagnose_pair_saliency."
    echo "Please first run: cp run_uie_lora_pairdiag.py src/run_uie_lora.py"
    exit 1
fi

if ! grep -q "pair_saliency_diagnostics_summary" src/federated_uie_lora.py; then
    echo "ERROR: src/federated_uie_lora.py does not contain pair_saliency diagnostics."
    echo "Please first run: cp federated_uie_lora_pairdiag.py src/federated_uie_lora.py"
    exit 1
fi

mkdir -p "$base_out"

for com_budget in "${BUDGETS[@]}"; do
    echo "============================================"
    echo "Running pair-saliency diagnostics: budget=${com_budget}, seed=${seed}"
    echo "============================================"

    port=$(shuf -i25000-30000 -n1)
    out_dir=${base_out}/budget_${com_budget}_abpair_pairdiag_5client

    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch --config_file ${ACCEL_CONFIG} \
       --main_process_port $port \
       src/run_uie_lora.py \
       --report_to none \
       --do_train \
       $predict_param \
       --lora_dim $lora_rank \
       --model_name_or_path $model_path \
       --data_dir $data_dir \
       --task_config_dir $task_config_dir \
       --output_dir $out_dir \
       --per_device_train_batch_size 16 \
       --per_device_eval_batch_size 16 \
       --gradient_accumulation_steps 2 \
       --global_rounds 5 \
       --local_epochs 10 \
       --num_clients 50 \
       --clients_per_round 5 \
       --dirichlet_alpha $di_alpha \
       --partition_strategy label \
       --partition_label_key Dataset \
       --comm_budget $com_budget \
       --learning_rate $lr \
       --run_name dolly_pairdiag_budget${com_budget}_seed${seed}_5client \
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
       --lamda_1 $lamda_1 \
       --lamda_2 $lamda_2 \
       --federated_seed $seed \
       --method $method \
       --task 1 \
       --radius $radius \
       --gradient_checkpointing True \
       --bf16 True \
       --ddp_find_unused_parameters False \
       --upload_atomic_mode ab_pair \
       --upload_diversity_mode none \
       --diagnose_residual_errors False \
       --lora_residual_accumulation False \
       --diagnose_pair_saliency True \
       --pair_saliency_reparam_scales $pair_scales \
       --pair_saliency_save_top_units True \
       --pair_saliency_top_n 20 \
       $random_layer_param

    sleep 5
done

echo "All pair-saliency budget-sweep diagnostics finished."
echo "Results are under: ${base_out}"
