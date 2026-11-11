#!/usr/bin/env bash
set -e

PYTHON=/home/lcf/.conda/envs/harmony-clean/bin/python
PROJECT_ROOT=/home/lcf/MoE_Syn/HarMoEny-main
MODEL_NAME=/home/lcf/MoE_Syn/models/google_switch_base_64
OUT_ROOT=/home/lcf/MoE_Syn/HarMoEny-main/ablation_outputs/qosmoe_ablation_high_batch

mkdir -p "${OUT_ROOT}"

run_one () {
  NAME=$1
  ASYNC=$2
  LB=$3
  PREFETCH=$4
  PORT=$5

  echo "============================================================"
  echo "Running HighBatch/${NAME}: B48/L256 async=${ASYNC}, lb=${LB}, prefetch=${PREFETCH}"
  echo "Output: ${OUT_ROOT}/${NAME}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=2,3 \
  QOSMOE_ENABLE_ASYNC=${ASYNC} \
  QOSMOE_ENABLE_LOAD_BALANCE=${LB} \
  QOSMOE_ENABLE_PREFETCH=${PREFETCH} \
  ${PYTHON} src/run_all_moe_benchmarks.py \
    --project_root ${PROJECT_ROOT} \
    --outputs_root ${OUT_ROOT}/${NAME} \
    --systems qosmoe \
    --tag HighBatch_${NAME} \
    --model_name ${MODEL_NAME} \
    --dataset wikitext \
    --num_samples 2048 \
    --batch_size 48 \
    --seq_len 256 \
    --world_size 2 \
    --expert_cache_size 6 \
    --eq_tokens 150 \
    --qos_target_batch 48 \
    --enable_router_skew 1 \
    --router_skew 0.9 \
    --router_num_experts_skewed 12 \
    --capacity_factor 8.0 \
    --cuda_visible_devices 2,3 \
    --qos_default_deadline_s 0.34 \
    --qos_w1 0.1 \
    --qos_w2 0.0 \
    --qos_w3 1.0 \
    --qos_delta 0.02 \
    --qos_enable_prefetch 1 \
    --qos_prefetch_topk 1 \
    --qos_prefetch_budget_experts 1 \
    --base_port ${PORT}
}

run_one QoSMoE-Full        1 1 1 30401
run_one QoSMoE-wo-Async    0 1 1 30402
run_one QoSMoE-wo-LB       1 0 1 30403
run_one QoSMoE-wo-Prefetch 1 1 0 30404
