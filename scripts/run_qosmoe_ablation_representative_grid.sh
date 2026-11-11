#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/lcf/.conda/envs/harmony-clean/bin/python
PROJECT_ROOT=/home/lcf/MoE_Syn/HarMoEny-main
MODEL_NAME=/home/lcf/MoE_Syn/models/google_switch_base_64

OUT_ROOT=${PROJECT_ROOT}/ablation_outputs/qosmoe_ablation_representative_grid
CUDA_DEVICES=${CUDA_DEVICES:-2,3}

ROUTER_SKEW=0.95
SKEWED_EXPERTS=12
WORLD_SIZE=2
WARMUP_ROUNDS=3
ITERS_PER_CONFIG=32

mkdir -p "${OUT_ROOT}"

deadline_by_seq () {
  local SEQ=$1
  if [ "$SEQ" -le 128 ]; then
    echo "0.28"
  elif [ "$SEQ" -le 256 ]; then
    echo "0.40"
  elif [ "$SEQ" -le 512 ]; then
    echo "0.70"
  else
    echo "1.00"
  fi
}

eqtokens_by_seq () {
  local SEQ=$1
  if [ "$SEQ" -le 128 ]; then
    echo "100"
  elif [ "$SEQ" -le 256 ]; then
    echo "150"
  elif [ "$SEQ" -le 512 ]; then
    echo "250"
  else
    echo "300"
  fi
}

already_done () {
  local DIR=$1
  local N
  N=$(find "${DIR}" -name "e2e.csv" 2>/dev/null | wc -l)
  [ "$N" -gt 0 ]
}

run_variant () {
  local ID=$1
  local BATCH=$2
  local SEQ_LEN=$3
  local CACHE=$4
  local VARIANT=$5
  local ASYNC=$6
  local LB=$7
  local PREFETCH=$8
  local PORT=$9

  local NUM_SAMPLES=$((BATCH * ITERS_PER_CONFIG))
  local DEADLINE
  local EQTOK
  DEADLINE=$(deadline_by_seq ${SEQ_LEN})
  EQTOK=$(eqtokens_by_seq ${SEQ_LEN})

  local SETTING="${ID}_B${BATCH}_L${SEQ_LEN}_C${CACHE}_S095_E${SKEWED_EXPERTS}"
  local OUT_DIR="${OUT_ROOT}/${SETTING}/${VARIANT}"

  mkdir -p "${OUT_DIR}"

  if already_done "${OUT_DIR}"; then
    echo "[SKIP] ${SETTING}/${VARIANT} already has e2e.csv"
    return 0
  fi

  echo "======================================================================"
  echo "[RUN] setting=${SETTING}"
  echo "      variant=${VARIANT}"
  echo "      batch=${BATCH}, seq=${SEQ_LEN}, cache=${CACHE}, skew=${ROUTER_SKEW}, skewed=${SKEWED_EXPERTS}"
  echo "      async=${ASYNC}, lb=${LB}, prefetch=${PREFETCH}"
  echo "      num_samples=${NUM_SAMPLES}, eq_tokens=${EQTOK}, deadline=${DEADLINE}, port=${PORT}"
  echo "      out=${OUT_DIR}"
  echo "======================================================================"

  set +e
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
  QOSMOE_ENABLE_ASYNC=${ASYNC} \
  QOSMOE_ENABLE_LOAD_BALANCE=${LB} \
  QOSMOE_ENABLE_PREFETCH=${PREFETCH} \
  ${PYTHON} src/run_all_moe_benchmarks.py \
    --project_root ${PROJECT_ROOT} \
    --outputs_root ${OUT_DIR} \
    --systems qosmoe \
    --tag "${SETTING}_${VARIANT}" \
    --model_name ${MODEL_NAME} \
    --dataset wikitext \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH} \
    --seq_len ${SEQ_LEN} \
    --world_size ${WORLD_SIZE} \
    --warmup_rounds ${WARMUP_ROUNDS} \
    --expert_cache_size ${CACHE} \
    --eq_tokens ${EQTOK} \
    --qos_target_batch ${BATCH} \
    --enable_router_skew 1 \
    --enable_router_random 0 \
    --enable_router_uniform 0 \
    --router_skew ${ROUTER_SKEW} \
    --router_num_experts_skewed ${SKEWED_EXPERTS} \
    --random_router_skew 0.0 \
    --capacity_factor 8.0 \
    --cuda_visible_devices ${CUDA_DEVICES} \
    --qos_default_deadline_s ${DEADLINE} \
    --qos_avg_token_cost 0.005 \
    --qos_w1 0.1 \
    --qos_w2 0.0 \
    --qos_w3 1.0 \
    --qos_alpha 1.0 \
    --qos_beta 0.1 \
    --qos_delta 0.02 \
    --qos_enable_prefetch 1 \
    --qos_prefetch_topk 1 \
    --qos_prefetch_budget_experts 1 \
    --qos_prefetch_score_power 1.0 \
    --qos_attention_mode router_var \
    --base_port ${PORT}
  RET=$?
  set -e

  if [ "$RET" -ne 0 ]; then
    echo "[WARN] ${SETTING}/${VARIANT} failed with exit code ${RET}; continue."
  fi
}

run_setting () {
  local ID=$1
  local BATCH=$2
  local SEQ_LEN=$3
  local CACHE=$4
  local PORT_BASE=$5

  run_variant ${ID} ${BATCH} ${SEQ_LEN} ${CACHE} "QoSMoE-Full"        1 1 1 $((PORT_BASE + 0))
  run_variant ${ID} ${BATCH} ${SEQ_LEN} ${CACHE} "QoSMoE-wo-Async"    0 1 1 $((PORT_BASE + 1))
  run_variant ${ID} ${BATCH} ${SEQ_LEN} ${CACHE} "QoSMoE-wo-LB"       1 0 1 $((PORT_BASE + 2))
  run_variant ${ID} ${BATCH} ${SEQ_LEN} ${CACHE} "QoSMoE-wo-Prefetch" 1 1 0 $((PORT_BASE + 3))
}

PORT=40000

# Batch scaling: fixed L=256, C=8
run_setting R01  32 256  8 ${PORT}; PORT=$((PORT + 10))
run_setting R02  64 256  8 ${PORT}; PORT=$((PORT + 10))
run_setting R03 128 256  8 ${PORT}; PORT=$((PORT + 10))
run_setting R04 256 256  8 ${PORT}; PORT=$((PORT + 10))

# Cache scaling: fixed B=64, L=256
run_setting R05  64 256  4 ${PORT}; PORT=$((PORT + 10))
run_setting R06  64 256 16 ${PORT}; PORT=$((PORT + 10))
run_setting R07  64 256 32 ${PORT}; PORT=$((PORT + 10))

# Sequence scaling: fixed B=64, C=8
run_setting R08  64 128  8 ${PORT}; PORT=$((PORT + 10))
run_setting R09  64 512  8 ${PORT}; PORT=$((PORT + 10))

# Combined pressure
run_setting R10 128 128 16 ${PORT}; PORT=$((PORT + 10))
run_setting R11 128 512 16 ${PORT}; PORT=$((PORT + 10))
run_setting R12 256 512 32 ${PORT}; PORT=$((PORT + 10))

echo "[DONE] QoSMoE representative ablation grid finished."
