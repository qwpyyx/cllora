#!/bin/bash
# Fix: SN-P1/P2 with raw (no depth normalization)
# Hypothesis: depth_balanced rank_norm over-amplifies small differences in a_hat
set -euo pipefail; set -x
export PYTHONWARNINGS="ignore"; export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_ASYNC_ERROR_HANDLING=1; export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN; export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1; export WANDB_DISABLED=true
PY=/home/qiuwenqi/.conda/envs/qwen3/bin/python
ACC="${PY} -m accelerate.commands.launch --config_file script_accelerate/accelerate_config.yaml"
C="--report_to none --do_train --do_predict --predict_with_generate --lora_dim 8 --model_name_or_path /home/qiuwenqi/LLM/models/Qwen3-14B-Instruct --data_dir /home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data --task_config_dir /home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 2 --global_rounds 5 --local_epochs 10 --num_clients 50 --clients_per_round 10 --dirichlet_alpha 10 --partition_strategy quantity --comm_budget 1760 --learning_rate 1e-04 --max_source_length 512 --max_target_length 50 --generation_max_length 50 --add_task_name False --add_dataset_name False --overwrite_output_dir --overwrite_cache --lr_scheduler_type constant --warmup_steps 0 --logging_strategy steps --logging_steps 2 --evaluation_strategy no --save_strategy no --save_steps 150 --lamda_1 0 --lamda_2 0 --federated_seed 28 --method lora_origin --task 1 --radius 0 --gradient_checkpointing True --bf16 True --ddp_find_unused_parameters False --upload_atomic_mode qv_block --upload_score_mode sn_p1p2 --sn_gap_eta 1.0 --sn_force_full_budget False --sn_save_diagnostics True --upload_diversity_mode none --diagnose_pair_saliency False --diagnose_residual_errors False --lora_residual_accumulation False"
OUT=results/Qwen3_ours_fix_snraw; rm -rf ${OUT}; mkdir -p ${OUT}/logs

# Fix 1: raw mode (no normalization)
P1=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES=0,1,2,3 ${ACC} --main_process_port ${P1} src/run_uie_lora.py ${C} \
  --output_dir ${OUT}/sn_raw_seed28 --run_name fix_sn_raw \
  --sn_p1_norm_mode raw \
  > ${OUT}/logs/sn_raw.log 2>&1 &
PID1=$!

# Fix 2: raw + force_full_budget (use all budget even with negative marginal gains)
P2=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES=4,5,6,7 ${ACC} --main_process_port ${P2} src/run_uie_lora.py ${C} \
  --output_dir ${OUT}/sn_raw_force_seed28 --run_name fix_sn_raw_force \
  --sn_p1_norm_mode raw --sn_force_full_budget True \
  > ${OUT}/logs/sn_raw_force.log 2>&1 &
PID2=$!

wait ${PID1}; wait ${PID2}
echo "=== Results ==="
for t in sn_raw sn_raw_force; do
  f=${OUT}/${t}_seed28/all_results.json
  [ -f "$f" ] && echo "${t}: $(grep -o 'predict_gsm8k_em.: [0-9.]*' $f)"
done
