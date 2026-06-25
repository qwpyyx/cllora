#!/bin/bash
# Phase 1 followup: alpha=1 + lower budget tests
set -euo pipefail; set -x
export PYTHONWARNINGS="ignore"; export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_ASYNC_ERROR_HANDLING=1; export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN; export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1; export WANDB_DISABLED=true
PY=/home/qiuwenqi/.conda/envs/qwen3/bin/python
ACC="${PY} -m accelerate.commands.launch --config_file script_accelerate/accelerate_config.yaml"
M=/home/qiuwenqi/LLM/models/Qwen3-14B-Instruct
D=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_data
T=/home/qiuwenqi/LLM/Third_work/data_gsm8k/gsm8k_uie_task_config
C="--report_to none --do_train --do_predict --predict_with_generate --lora_dim 8 --model_name_or_path ${M} --data_dir ${D} --task_config_dir ${T} --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 2 --global_rounds 5 --local_epochs 10 --num_clients 50 --clients_per_round 10 --partition_strategy quantity --learning_rate 1e-04 --max_source_length 512 --max_target_length 50 --generation_max_length 50 --add_task_name False --add_dataset_name False --overwrite_output_dir --overwrite_cache --lr_scheduler_type constant --warmup_steps 0 --logging_strategy steps --logging_steps 2 --evaluation_strategy no --save_strategy no --save_steps 150 --lamda_1 0 --lamda_2 0 --federated_seed 28 --method lora_origin --task 1 --radius 0 --gradient_checkpointing True --bf16 True --ddp_find_unused_parameters False --upload_diversity_mode none --diagnose_pair_saliency False --diagnose_residual_errors False --lora_residual_accumulation False"
OUT=results/Qwen3_followup; rm -rf ${OUT}; mkdir -p ${OUT}/logs

# 1) alpha=1, qv_factor_norm, 12.5% budget
P1=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES=0,1,2,3 ${ACC} --main_process_port ${P1} src/run_uie_lora.py ${C} \
  --dirichlet_alpha 1 --comm_budget 1760 \
  --output_dir ${OUT}/a1_qvfac_seed28 --run_name a1_qvfac \
  --upload_atomic_mode qv_block --upload_score_mode factor_norm \
  > ${OUT}/logs/a1_qvfac.log 2>&1 &
PID1=$!

# 2) alpha=1, SN-P1/P2 depth_rank
P2=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES=4,5,6,7 ${ACC} --main_process_port ${P2} src/run_uie_lora.py ${C} \
  --dirichlet_alpha 1 --comm_budget 1760 \
  --output_dir ${OUT}/a1_sndr_seed28 --run_name a1_sndr \
  --upload_atomic_mode qv_block --upload_score_mode sn_p1p2 \
  --sn_p1_norm_mode depth_rank --sn_gap_eta 1.0 --sn_force_full_budget False --sn_save_diagnostics True \
  > ${OUT}/logs/a1_sndr.log 2>&1 &
PID2=$!

wait ${PID1}; wait ${PID2}

# 3) alpha=10, lower budget 6.25% (880), qv_factor_norm
P3=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES=0,1,2,3 ${ACC} --main_process_port ${P3} src/run_uie_lora.py ${C} \
  --dirichlet_alpha 10 --comm_budget 880 \
  --output_dir ${OUT}/bud880_qvfac_seed28 --run_name bud880_qvfac \
  --upload_atomic_mode qv_block --upload_score_mode factor_norm \
  > ${OUT}/logs/bud880_qvfac.log 2>&1

# Summary
echo "=== Results ==="
for t in a1_qvfac a1_sndr bud880_qvfac; do
  f=${OUT}/${t}_seed28/all_results.json
  [ -f "$f" ] && echo "${t}: $(grep -o 'predict_gsm8k_em.: [0-9.]*' $f)"
done
