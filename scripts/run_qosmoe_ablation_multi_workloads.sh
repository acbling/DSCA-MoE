#!/usr/bin/env bash
set -e

PYTHON=/home/lcf/.conda/envs/harmony-clean/bin/python
PROJECT_ROOT=/home/lcf/MoE_Syn/HarMoEny-main
MODEL_NAME=/home/lcf/MoE_Syn/models/google_switch_base_64
OUT_ROOT=/home/lcf/MoE_Syn/HarMoEny-main/ablation_outputs/qosmoe_component_ablation_multi

mkdir -p "${OUT_ROOT}"

run_one () {
  WORKLOAD=$1
  BATCH=$2
  SEQLEN=$3
  NAME=$4
  ASYNC=$5
  LB=$6
  PREFETCH=$7
  PORT=$8

  echo "============================================================"
  echo "Running ${WORKLOAD}/${NAME}: batch=${BATCH}, seq=${SEQLEN}, async=${ASYNC}, lb=${LB}, prefetch=${PREFETCH}"
  echo "Output: ${OUT_ROOT}/${WORKLOAD}/${NAME}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=0,2 \
  QOSMOE_ENABLE_ASYNC=${ASYNC} \
  QOSMOE_ENABLE_LOAD_BALANCE=${LB} \
  QOSMOE_ENABLE_PREFETCH=${PREFETCH} \
  ${PYTHON} src/run_all_moe_benchmarks.py \
    --project_root ${PROJECT_ROOT} \
    --outputs_root ${OUT_ROOT}/${WORKLOAD}/${NAME} \
    --systems qosmoe \
    --tag ${WORKLOAD}_${NAME} \
    --model_name ${MODEL_NAME} \
    --dataset wikitext \
    --num_samples 2048 \
    --batch_size ${BATCH} \
    --seq_len ${SEQLEN} \
    --world_size 2 \
    --expert_cache_size 6 \
    --eq_tokens 150 \
    --qos_target_batch ${BATCH} \
    --enable_router_skew 1 \
    --router_skew 0.9 \
    --router_num_experts_skewed 12 \
    --capacity_factor 8.0 \
    --cuda_visible_devices 0,1 \
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

run_workload () {
  WORKLOAD=$1
  BATCH=$2
  SEQLEN=$3
  BASEPORT=$4

  run_one ${WORKLOAD} ${BATCH} ${SEQLEN} QoSMoE-Full        1 1 1 ${BASEPORT}
  run_one ${WORKLOAD} ${BATCH} ${SEQLEN} QoSMoE-wo-Async    0 1 1 $((BASEPORT + 1))
  run_one ${WORKLOAD} ${BATCH} ${SEQLEN} QoSMoE-wo-LB       1 0 1 $((BASEPORT + 2))
  run_one ${WORKLOAD} ${BATCH} ${SEQLEN} QoSMoE-wo-Prefetch 1 1 0 $((BASEPORT + 3))
}

run_workload B16_L256 16 256 29701
run_workload B32_L256 32 256 29711
run_workload B16_L384 16 384 29721
run_workload B32_L384 32 384 29731

echo "============================================================"
echo "All multi-workload QoSMoE ablation runs finished."
echo "Results are under: ${OUT_ROOT}"
echo "============================================================"
