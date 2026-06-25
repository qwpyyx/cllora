#!/bin/bash
# Dolly Full Phase 1: 6 methods × 3 seeds, 8 GPUs parallel
set -euo pipefail; set -x
export PYTHONWARNINGS="ignore"; export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_ASYNC_ERROR_HANDLING=1; export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN; export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1; export WANDB_DISABLED=true
PY=/home/qiuwenqi/.conda/envs/qwen3/bin/python
ACC="${PY} -m accelerate.commands.launch --config_file script_accelerate/accelerate_config.yaml"
M=/home/qiuwenqi/LLM/models/Qwen3-14B-Instruct
D=/home/qiuwenqi/LLM/Third_work/data_dolly/dolly_uie/data
T=/home/qiuwenqi/LLM/Third_work/data_dolly/dolly_uie/task_config
C="--report_to none --do_train --do_predict --predict_with_generate --lora_dim 8 --model_name_or_path ${M} --data_dir ${D} --task_config_dir ${T} --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 2 --global_rounds 5 --local_epochs 10 --num_clients 50 --clients_per_round 5 --dirichlet_alpha 0.5 --partition_strategy label --partition_label_key Dataset --learning_rate 1e-04 --max_source_length 512 --max_target_length 128 --generation_max_length 128 --add_task_name False --add_dataset_name False --overwrite_output_dir --overwrite_cache --lr_scheduler_type constant --warmup_steps 0 --logging_strategy steps --logging_steps 2 --evaluation_strategy no --save_strategy no --save_steps 150 --lamda_1 0 --lamda_2 0 --task 1 --radius 0 --gradient_checkpointing True --bf16 True --ddp_find_unused_parameters False --upload_diversity_mode none --diagnose_pair_saliency False --diagnose_residual_errors False --lora_residual_accumulation False"
OUT=results/Qwen3_dolly_full; rm -rf ${OUT}; mkdir -p ${OUT}/logs

run_dense() {
    local seed=$1; local gpus=$2; local port=$(shuf -i25000-30000 -n1)
    CUDA_VISIBLE_DEVICES=${gpus} ${ACC} --main_process_port ${port} src/run_uie_lora.py ${C} \
      --federated_seed ${seed} --comm_budget 0 --method lora_origin \
      --output_dir ${OUT}/dense_s${seed} --run_name dl_dense_s${seed} \
      > ${OUT}/logs/dense_s${seed}.log 2>&1
}

run_flasc() {
    local seed=$1; local gpus=$2; local port=$(shuf -i25000-30000 -n1)
    CUDA_VISIBLE_DEVICES=${gpus} ${ACC} --main_process_port ${port} src/run_uie_lora.py ${C} \
      --federated_seed ${seed} --comm_budget 2200 --method flasc \
      --baseline_packet_num 0 --baseline_blocks 192 --baseline_bit 18 --baseline_min_bit 4 \
      --baseline_topk_method gradient --fedcomp_use_residual True \
      --output_dir ${OUT}/flasc_s${seed} --run_name dl_flasc_s${seed} \
      > ${OUT}/logs/flasc_s${seed}.log 2>&1
}

run_compeft() {
    local seed=$1; local gpus=$2; local port=$(shuf -i25000-30000 -n1)
    CUDA_VISIBLE_DEVICES=${gpus} ${ACC} --main_process_port ${port} src/run_uie_lora.py ${C} \
      --federated_seed ${seed} --comm_budget 2200 --method compeft \
      --baseline_packet_num 0 --baseline_blocks 192 --baseline_bit 18 --baseline_min_bit 4 \
      --baseline_topk_method gradient --fedcomp_use_residual True \
      --output_dir ${OUT}/compeft_s${seed} --run_name dl_compeft_s${seed} \
      > ${OUT}/logs/compeft_s${seed}.log 2>&1
}

run_flmtopk() {
    local seed=$1; local gpus=$2; local port=$(shuf -i25000-30000 -n1)
    CUDA_VISIBLE_DEVICES=${gpus} ${ACC} --main_process_port ${port} src/run_uie_lora.py ${C} \
      --federated_seed ${seed} --comm_budget 2200 --method flm_topk \
      --baseline_packet_num 0 --baseline_blocks 192 --baseline_bit 18 --baseline_min_bit 4 \
      --baseline_topk_method gradient --fedcomp_use_residual True \
      --output_dir ${OUT}/flmtopk_s${seed} --run_name dl_flmtopk_s${seed} \
      > ${OUT}/logs/flmtopk_s${seed}.log 2>&1
}

run_fedcomp() {
    local seed=$1; local gpus=$2; local port=$(shuf -i25000-30000 -n1)
    CUDA_VISIBLE_DEVICES=${gpus} ${ACC} --main_process_port ${port} src/run_uie_lora.py ${C} \
      --federated_seed ${seed} --comm_budget 2200 --method fedcomp \
      --baseline_packet_num 0 --baseline_blocks 192 --baseline_bit 18 --baseline_min_bit 4 \
      --baseline_topk_method gradient --fedcomp_use_residual True \
      --output_dir ${OUT}/fedcomp_s${seed} --run_name dl_fedcomp_s${seed} \
      > ${OUT}/logs/fedcomp_s${seed}.log 2>&1
}

run_ours() {
    local seed=$1; local gpus=$2; local port=$(shuf -i25000-30000 -n1)
    CUDA_VISIBLE_DEVICES=${gpus} ${ACC} --main_process_port ${port} src/run_uie_lora.py ${C} \
      --federated_seed ${seed} --comm_budget 2200 --method lora_origin \
      --upload_atomic_mode qv_block --upload_score_mode sn_p1p2 \
      --sn_p1_norm_mode depth_rank --sn_gap_eta 1.0 --sn_force_full_budget False --sn_save_diagnostics True \
      --output_dir ${OUT}/ours_s${seed} --run_name dl_ours_s${seed} \
      > ${OUT}/logs/ours_s${seed}.log 2>&1
}

for seed in 28 42 45; do
    echo "=== Seed ${seed} ==="
    # Batch A: Dense + FLM-TopK (FLM-TopK slow, runs parallel to Dense)
    run_dense ${seed} "0,1,2,3" & PID1=$!
    run_flmtopk ${seed} "4,5,6,7" & PID2=$!
    wait $PID1 $PID2

    # Batch B: FLASC + ComPEFT
    run_flasc ${seed} "0,1,2,3" & PID3=$!
    run_compeft ${seed} "4,5,6,7" & PID4=$!
    wait $PID3 $PID4

    # Batch C: FedComp + Ours
    run_fedcomp ${seed} "0,1,2,3" & PID5=$!
    run_ours ${seed} "4,5,6,7" & PID6=$!
    wait $PID5 $PID6
done

echo "=== ALL DONE ==="
for m in dense flasc compeft flmtopk fedcomp ours; do
    for s in 28 42 45; do
        f=${OUT}/${m}_s${s}/all_results.json
        [ -f "$f" ] && echo "${m}_s${s}: $(grep -o 'predict_rouge1.: [0-9.]*' $f) $(grep -o 'predict_rougeL.: [0-9.]*' $f) $(grep -o 'predict_gen_len.: [0-9.]*' $f)" || echo "${m}_s${s}: MISSING"
    done
done
