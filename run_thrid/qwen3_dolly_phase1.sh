#!/bin/bash
set -euo pipefail; set -x
export PYTHONWARNINGS="ignore"; export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_ASYNC_ERROR_HANDLING=1; export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN; export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1; export WANDB_DISABLED=true
PY=/home/qiuwenqi/.conda/envs/qwen3/bin/python
ACC="${PY} -m accelerate.commands.launch --config_file script_accelerate/accelerate_config.yaml"
M=/home/qiuwenqi/LLM/models/Qwen3-14B-Instruct
C="--report_to none --do_train --do_predict --predict_with_generate --lora_dim 8 --model_name_or_path ${M} --data_dir /home/qiuwenqi/LLM/Third_work/data_dolly/dolly_uie/data --task_config_dir /home/qiuwenqi/LLM/Third_work/data_dolly/dolly_uie/task_config --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 2 --global_rounds 5 --local_epochs 10 --num_clients 50 --clients_per_round 5 --dirichlet_alpha 0.5 --partition_strategy label --partition_label_key Dataset --learning_rate 1e-04 --max_source_length 512 --max_target_length 128 --generation_max_length 128 --add_task_name False --add_dataset_name False --overwrite_output_dir --overwrite_cache --lr_scheduler_type constant --warmup_steps 0 --logging_strategy steps --logging_steps 2 --evaluation_strategy no --save_strategy no --save_steps 150 --lamda_1 0 --lamda_2 0 --federated_seed 42 --task 1 --radius 0 --gradient_checkpointing True --bf16 True --ddp_find_unused_parameters False --upload_diversity_mode none --diagnose_pair_saliency False --diagnose_residual_errors False --lora_residual_accumulation False"
OUT=results/Qwen3_dolly_phase1; rm -rf ${OUT}; mkdir -p ${OUT}/logs

# Batch 1: Dense + FLM-TopK
P1=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES=0,1,2,3 ${ACC} --main_process_port ${P1} src/run_uie_lora.py ${C} \
  --comm_budget 0 --method lora_origin --output_dir ${OUT}/dense --run_name dl_dense \
  > ${OUT}/logs/dense.log 2>&1 &
PID1=$!

P2=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES=4,5,6,7 ${ACC} --main_process_port ${P2} src/run_uie_lora.py ${C} \
  --comm_budget 2200 --method flm_topk --baseline_packet_num 0 --baseline_blocks 192 --baseline_bit 18 --baseline_min_bit 4 --baseline_topk_method gradient --fedcomp_use_residual True --output_dir ${OUT}/flm_topk --run_name dl_flmtopk \
  > ${OUT}/logs/flm_topk.log 2>&1 &
PID2=$!
wait ${PID1} ${PID2}

# Batch 2: FLASC + Ours
P3=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES=0,1,2,3 ${ACC} --main_process_port ${P3} src/run_uie_lora.py ${C} \
  --comm_budget 2200 --method flasc --baseline_packet_num 0 --baseline_blocks 192 --baseline_bit 18 --baseline_min_bit 4 --baseline_topk_method gradient --fedcomp_use_residual True --output_dir ${OUT}/flasc --run_name dl_flasc \
  > ${OUT}/logs/flasc.log 2>&1 &
PID3=$!

P4=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES=4,5,6,7 ${ACC} --main_process_port ${P4} src/run_uie_lora.py ${C} \
  --comm_budget 2200 --method lora_origin --upload_atomic_mode qv_block --upload_score_mode sn_p1p2 --sn_p1_norm_mode depth_rank --sn_gap_eta 1.0 --sn_force_full_budget False --sn_save_diagnostics True --output_dir ${OUT}/ours --run_name dl_ours \
  > ${OUT}/logs/ours.log 2>&1 &
PID4=$!
wait ${PID3} ${PID4}

echo "=== ALL DONE ==="
for m in dense flm_topk flasc ours; do
  f=${OUT}/${m}/all_results.json
  [ -f "$f" ] && echo "${m}: $(grep -o 'predict_rouge[^:]*: [0-9.]*' $f | head -2) $(grep -o 'predict_gen_len[^:]*: [0-9.]*' $f)"
done
