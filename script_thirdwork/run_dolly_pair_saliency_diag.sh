#!/bin/bash
set -euo pipefail
set -x

export PYTHONWARNINGS="ignore"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/qiuwenqi/.cache/huggingface
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

# ==================1==========================================
# Dolly A/B-pair saliency diagnostics
#
# This single run produces BOTH diagnostics:
#   Diag 1: Equivalent reparameterization changes factor-norm Top-K selection.
#   Diag 2: Factor-norm pair ranking vs effective-update pair ranking mismatch.
#
# Required code version:
#   src/run_uie_lora.py and src/federated_uie_lora.py must include the
#   pair-saliency diagnostic code, i.e., arguments/functions containing
#   "pair_saliency".
# ============================================================

if ! grep -q "pair_saliency" src/run_uie_lora.py; then
    echo "[ERROR] src/run_uie_lora.py does not contain pair_saliency arguments." >&2
    echo "Please replace it with run_uie_lora_pairdiag.py first." >&2
    exit 1
fi

if ! grep -q "pair_saliency" src/federated_uie_lora.py; then
    echo "[ERROR] src/federated_uie_lora.py does not contain pair_saliency diagnostics." >&2
    echo "Please replace it with federated_uie_lora_pairdiag.py first." >&2
    exit 1
fi

method=lora_origin
lora_rank=8
lamda_2=0
lamda_1=0
lr=1e-04
radius=0
com_budget=2200
is_random_select=False
di_alpha=0.5
seed=42

# GPUs and paths follow your original Dolly script.
gpus=4,5,6,7
data_dir=/home/qiuwenqi/LLM/Third_work/data_dolly/dolly_uie/data
task_config_dir=/home/qiuwenqi/LLM/Third_work/data_dolly/dolly_uie/task_config
model_path=/home/qiuwenqi/LLM/models/Qwen2.5-14B-Instruct

# Pair-saliency diagnostic settings.
# The scales are used for Diag 1. Diag 2 is computed automatically in the same run.
reparam_scales=0.25,0.5,2,4
save_top_units=True
top_n=20

if [ "$is_random_select" = "True" ]; then
    random_layer_param="--random_layer_selection True"
else
    random_layer_param=""
fi

############################################
# Diag 1 + Diag 2: Dolly A/B pair Top-K
############################################

port=$(shuf -i25000-30000 -n1)
output_dir=results/Qwen2_dolly/lora_exp_dolly_alpha05/2200_seed42_abpair_pair_saliency_diag_5client

CUDA_VISIBLE_DEVICES=$gpus accelerate launch --config_file script_accelerate/accelerate_config.yaml \
   --main_process_port $port \
   src/run_uie_lora.py \
   --report_to none \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --lora_dim $lora_rank \
   --model_name_or_path $model_path \
   --data_dir $data_dir \
   --task_config_dir $task_config_dir \
   --output_dir $output_dir \
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
   --run_name dolly_abpair_pair_saliency_diag_seed42_5client \
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
   --pair_saliency_reparam_scales $reparam_scales \
   --pair_saliency_save_top_units $save_top_units \
   --pair_saliency_top_n $top_n \
   $random_layer_param

echo "Dolly A/B-pair saliency diagnostics finished."
echo "Output directory: $output_dir"
echo "Expected diagnostic files:"
echo "  $output_dir/pair_saliency_diagnostics_history.json"
echo "  $output_dir/pair_saliency_diagnostics_summary.json"
