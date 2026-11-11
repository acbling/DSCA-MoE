#!/usr/bin/env bash
set -e

PYTHON=/home/lcf/.conda/envs/harmony-clean/bin/python
PROJECT_ROOT=/home/lcf/MoE_Syn/HarMoEny-main
MODEL_NAME=/home/lcf/MoE_Syn/models/google_switch_base_64
OUT_ROOT=/home/lcf/MoE_Syn/HarMoEny-main/ablation_outputs/qosmoe_ablation_grid_v2_gpu23_resume
CUDA_DEVICES=2,3

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUT_ROOT}"

already_done () {
  SETTING=$1
  VARIANT=$2
  SAFE_SETTING=$(echo "${SETTING}" | tr ' /' '__')
  COUNT=$(find "${OUT_ROOT}/${SAFE_SETTING}/${VARIANT}" -name "e2e.csv" 2>/dev/null | wc -l)
  [ "$COUNT" -gt 0 ]
}


run_one_variant () {
  SETTING=$1
  BATCH=$2
  SEQ=$3
  CACHE=$4
  SKEW=$5
  SKEWED=$6
  EQTOK=$7
  DEADLINE=$8
  VARIANT=$9
  ASYNC=${10}
  LB=${11}
  PREFETCH=${12}
  PORT=${13}

  SAFE_SETTING=$(echo "${SETTING}" | tr ' /' '__')
  VARIANT_DIR="${OUT_ROOT}/${SAFE_SETTING}/${VARIANT}"

  if already_done "${SETTING}" "${VARIANT}"; then
    echo "[SKIP] ${SETTING}/${VARIANT} already has e2e.csv"
    return 0
  fi

  mkdir -p "${VARIANT_DIR}"

  echo "======================================================================"
  echo "[RUN] setting=${SETTING}"
  echo "      variant=${VARIANT}"
  echo "      batch=${BATCH}, seq=${SEQ}, cache=${CACHE}, skew=${SKEW}, skewed=${SKEWED}"
  echo "      async=${ASYNC}, lb=${LB}, prefetch=${PREFETCH}, port=${PORT}"
  echo "      out=${VARIANT_DIR}"
  echo "======================================================================"

  CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
  QOSMOE_ENABLE_ASYNC=${ASYNC} \
  QOSMOE_ENABLE_LOAD_BALANCE=${LB} \
  QOSMOE_ENABLE_PREFETCH=${PREFETCH} \
  ${PYTHON} src/run_all_moe_benchmarks.py \
    --project_root ${PROJECT_ROOT} \
    --outputs_root ${VARIANT_DIR} \
    --systems qosmoe \
    --tag "${SAFE_SETTING}_${VARIANT}" \
    --model_name ${MODEL_NAME} \
    --dataset wikitext \
    --num_samples 2048 \
    --batch_size ${BATCH} \
    --seq_len ${SEQ} \
    --world_size 2 \
    --expert_cache_size ${CACHE} \
    --eq_tokens ${EQTOK} \
    --qos_target_batch ${BATCH} \
    --enable_router_skew 1 \
    --router_skew ${SKEW} \
    --router_num_experts_skewed ${SKEWED} \
    --capacity_factor 8.0 \
    --cuda_visible_devices ${CUDA_DEVICES} \
    --qos_default_deadline_s ${DEADLINE} \
    --qos_w1 0.1 \
    --qos_w2 0.0 \
    --qos_w3 1.0 \
    --qos_delta 0.02 \
    --qos_enable_prefetch 1 \
    --qos_prefetch_topk 1 \
    --qos_prefetch_budget_experts 1 \
    --base_port ${PORT}
}

run_setting () {
  SETTING=$1
  BATCH=$2
  SEQ=$3
  CACHE=$4
  SKEW=$5
  SKEWED=$6
  EQTOK=$7
  DEADLINE=$8
  PORT_BASE=$9

  run_one_variant "${SETTING}" ${BATCH} ${SEQ} ${CACHE} ${SKEW} ${SKEWED} ${EQTOK} ${DEADLINE} QoSMoE-Full        1 1 1 $((PORT_BASE + 0))
  run_one_variant "${SETTING}" ${BATCH} ${SEQ} ${CACHE} ${SKEW} ${SKEWED} ${EQTOK} ${DEADLINE} QoSMoE-wo-Async    0 1 1 $((PORT_BASE + 1))
  run_one_variant "${SETTING}" ${BATCH} ${SEQ} ${CACHE} ${SKEW} ${SKEWED} ${EQTOK} ${DEADLINE} QoSMoE-wo-LB       1 0 1 $((PORT_BASE + 2))
  run_one_variant "${SETTING}" ${BATCH} ${SEQ} ${CACHE} ${SKEW} ${SKEWED} ${EQTOK} ${DEADLINE} QoSMoE-wo-Prefetch 1 1 0 $((PORT_BASE + 3))
}

# ----------------------------------------------------------------------
# 24 representative pressure settings
# Format:
# run_setting "SettingName" batch seq cache skew skewed_experts eq_tokens deadline port_base
# ----------------------------------------------------------------------

# A. Main representative pressure
run_setting "A01_main_B32_L256_C6_S090_E12"        32 256 6 0.90 12 150 0.34 31000
run_setting "A02_main_B32_L256_C4_S090_E12"        32 256 4 0.90 12 150 0.34 31010

# B. Cache pressure sweep
run_setting "B01_cache2_B32_L256_C2_S090_E12"      32 256 2 0.90 12 150 0.34 31020
run_setting "B02_cache4_B32_L256_C4_S090_E12"      32 256 4 0.90 12 150 0.34 31030
run_setting "B03_cache6_B32_L256_C6_S090_E12"      32 256 6 0.90 12 150 0.34 31040
run_setting "B04_cache8_B32_L256_C8_S090_E12"      32 256 8 0.90 12 150 0.34 31050

# C. Skew pressure sweep
run_setting "C01_skew070_B32_L256_C6_S070_E12"     32 256 6 0.70 12 150 0.34 31060
run_setting "C02_skew090_B32_L256_C6_S090_E12"     32 256 6 0.90 12 150 0.34 31070
run_setting "C03_skew095_B32_L256_C6_S095_E8"      32 256 6 0.95  8 150 0.34 31080
run_setting "C04_skew099_B32_L256_C6_S099_E4"      32 256 6 0.99  4 150 0.34 31090

# D. Hotspot width sweep
run_setting "D01_hotspot4_B32_L256_C6_S095_E4"     32 256 6 0.95  4 150 0.34 31100
run_setting "D02_hotspot8_B32_L256_C6_S095_E8"     32 256 6 0.95  8 150 0.34 31110
run_setting "D03_hotspot12_B32_L256_C6_S095_E12"   32 256 6 0.95 12 150 0.34 31120
run_setting "D04_hotspot16_B32_L256_C6_S095_E16"   32 256 6 0.95 16 150 0.34 31130

# E. Sequence pressure sweep
run_setting "E01_seq128_B32_L128_C6_S090_E12"      32 128 6 0.90 12 100 0.24 31140
run_setting "E02_seq256_B32_L256_C6_S090_E12"      32 256 6 0.90 12 150 0.34 31150
run_setting "E03_seq384_B32_L384_C6_S090_E12"      32 384 6 0.90 12 200 0.46 31160
run_setting "E04_seq512_B32_L512_C6_S090_E12"      32 512 6 0.90 12 250 0.58 31170

# F. Batch pressure sweep
run_setting "F01_batch16_B16_L256_C6_S090_E12"     16 256 6 0.90 12 150 0.34 31180
run_setting "F02_batch32_B32_L256_C6_S090_E12"     32 256 6 0.90 12 150 0.34 31190
run_setting "F03_batch48_B48_L256_C4_S095_E8"      48 256 4 0.95  8 150 0.34 31200
run_setting "F04_batch64_B64_L256_C4_S095_E8"      64 256 4 0.95  8 150 0.34 31210

# G. Combined pressure
run_setting "G01_extreme_cache_skew_B32_L256_C2_S099_E4" 32 256 2 0.99  4 150 0.34 31220
run_setting "G02_long_cache_skew_B32_L384_C2_S095_E8"    32 384 2 0.95  8 200 0.46 31230

echo "[DONE] all QoSMoE ablation grid v2 experiments finished."
