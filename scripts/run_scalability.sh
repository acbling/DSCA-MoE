#!/usr/bin/env bash

set -uo pipefail

# ============================================================
# DSCA-MoE Scalability Experiment
#
# Strong scaling:
#   1 GPU -> 2 GPUs -> 4 GPUs
#
# Systems:
#   DeepSpeed / HarMoEny / QoSMoE
#
# Fixed workload:
#   Switch-base-64
#   B16 / L256
#   4096 samples
# ============================================================


# ------------------------------------------------------------
# 0. Environment
# ------------------------------------------------------------

PYTHON=/home/lcf/.conda/envs/harmony-clean/bin/python

PROJECT_ROOT=/home/lcf/MoE_Syn/HarMoEny-main

MODEL_NAME=/home/lcf/MoE_Syn/models/google_switch_base_64

OUT_ROOT=${PROJECT_ROOT}/scalability_outputs

LOG_ROOT=${OUT_ROOT}/logs


cd "${PROJECT_ROOT}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${LOG_ROOT}"


# ------------------------------------------------------------
# 1. HuggingFace environment
#
# 不修改 flexible_dataset.py。
# 原代码使用 --dataset wikitext。
# 显式解除 offline，避免再次静默 fallback synthetic。
# ------------------------------------------------------------

export HF_HOME=/home/lcf/.cache/harmoeny/hf
export HF_DATASETS_CACHE=/home/lcf/.cache/harmoeny/datasets

unset HF_DATASETS_OFFLINE || true
unset HF_HUB_OFFLINE || true
unset TRANSFORMERS_OFFLINE || true


# ------------------------------------------------------------
# 2. Fixed workload
# ------------------------------------------------------------

DATASET=wikitext

NUM_SAMPLES=4096

BATCH_SIZE=16

SEQ_LEN=256

WARMUP_ROUNDS=10


# ------------------------------------------------------------
# 3. Model / MoE parameters
# ------------------------------------------------------------

D_MODEL=768

NUM_EXPERTS=64

EXPERT_CACHE_SIZE=8

EQ_TOKENS=150

CAPACITY_FACTOR=8.0


# ------------------------------------------------------------
# 4. Router workload
# ------------------------------------------------------------

ENABLE_ROUTER_SKEW=1

ROUTER_SKEW=0.9

HOT_EXPERTS=8


# ------------------------------------------------------------
# 5. QoSMoE parameters
#
# 与已有正式配置保持一致。
# Scalability 主要比较 throughput / P95，
# 不额外人为改变 SLO threshold。
# ------------------------------------------------------------

QOS_W1=0.1
QOS_W2=0.0
QOS_W3=1.0

QOS_DELTA=0.02

QOS_PREFETCH_TOPK=1
QOS_PREFETCH_BUDGET=1

QOS_TARGET_BATCH=${BATCH_SIZE}


# ------------------------------------------------------------
# 6. Repeats
#
# 正式论文默认 3 次。
#
# smoke test:
#   REPEATS=1 bash scripts/run_scalability.sh
# ------------------------------------------------------------

REPEATS=${REPEATS:-3}


# ------------------------------------------------------------
# 7. GPU configurations
# ------------------------------------------------------------

GPU_NUMS=(1 2 4)


get_devices () {

    local N=$1

    case "${N}" in

        1)
            echo "0"
            ;;

        2)
            echo "0,1"
            ;;

        4)
            echo "0,1,2,3"
            ;;

        *)
            echo "Unsupported GPU number: ${N}" >&2
            exit 1
            ;;

    esac
}


# ------------------------------------------------------------
# 8. Runtime cleanup
# ------------------------------------------------------------

cleanup_runtime () {

    pkill -9 -f "src/start_deepspeed.py" 2>/dev/null || true
    pkill -9 -f "src/start_harmony.py" 2>/dev/null || true
    pkill -9 -f "src/start_qosmoe.py" 2>/dev/null || true
    pkill -9 -f "run_all_moe_benchmarks.py" 2>/dev/null || true
    pkill -9 -f "deepspeed.launcher" 2>/dev/null || true
    pkill -9 -f "torchrun" 2>/dev/null || true

    sleep 3
}


# ------------------------------------------------------------
# 9. NCCL
#
# 与之前正式实验环境保持一致。
# ------------------------------------------------------------

export NCCL_DEBUG=WARN

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_LEVEL=SYS


# ------------------------------------------------------------
# 10. Dataset preflight
#
# 极其重要：
# 不允许正式 scalability 再 fallback synthetic。
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Checking WikiText dataset..."
echo "============================================================"

"${PYTHON}" - <<'PY'
import os
from datasets import load_dataset

cache_dir = os.path.expanduser(
    os.environ.get(
        "HF_DATASETS_CACHE",
        "~/.cache/harmoeny/datasets"
    )
)

print("HF_DATASETS_CACHE =", cache_dir)

try:
    ds = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train[:32]",
        cache_dir=cache_dir,
        download_mode="reuse_cache_if_exists",
    )

    print("[OK] WikiText loaded.")
    print("samples =", len(ds))

except Exception as e:

    print()
    print("[ERROR] WikiText cannot be loaded.")
    print(type(e).__name__ + ":", e)
    print()
    print("Stop here to avoid synthetic fallback.")
    raise SystemExit(1)
PY


if [[ $? -ne 0 ]]; then
    echo "[FATAL] Dataset preflight failed."
    exit 1
fi


# ------------------------------------------------------------
# 11. Metadata
# ------------------------------------------------------------

META=${OUT_ROOT}/scalability_meta.txt

cat > "${META}" <<EOF
Model=${MODEL_NAME}
Dataset=${DATASET}
NumExperts=${NUM_EXPERTS}

NumSamples=${NUM_SAMPLES}
BatchSize=${BATCH_SIZE}
SeqLen=${SEQ_LEN}
Warmup=${WARMUP_ROUNDS}

ExpertCacheSize=${EXPERT_CACHE_SIZE}
EqTokens=${EQ_TOKENS}
CapacityFactor=${CAPACITY_FACTOR}

RouterSkew=${ROUTER_SKEW}
HotExperts=${HOT_EXPERTS}

Repeats=${REPEATS}

GPUConfigs=1,2,4
EOF


# ------------------------------------------------------------
# 12. Run one experiment
# ------------------------------------------------------------

run_one () {

    SYSTEM=$1
    GPU_NUM=$2
    CUDA_DEVICES=$3
    REPEAT=$4
    PORT=$5

    TAG="scalability_${SYSTEM}_${GPU_NUM}gpu_r${REPEAT}"

    RUN_OUT="${OUT_ROOT}/${TAG}"

    LOG_FILE="${LOG_ROOT}/${TAG}.log"

    mkdir -p "${RUN_OUT}"


    echo
    echo "============================================================"
    echo "SYSTEM        : ${SYSTEM}"
    echo "GPU NUM       : ${GPU_NUM}"
    echo "CUDA DEVICES  : ${CUDA_DEVICES}"
    echo "EXPERTS/GPU   : $((NUM_EXPERTS / GPU_NUM))"
    echo "REPEAT        : ${REPEAT}/${REPEATS}"
    echo "TAG           : ${TAG}"
    echo "PORT          : ${PORT}"
    echo "============================================================"


    cleanup_runtime


    # --------------------------------------------------------
    # QoSMoE / DSCA-MoE
    # --------------------------------------------------------

    if [[ "${SYSTEM}" == "qosmoe" ]]; then

        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
        QOSMOE_ENABLE_ASYNC=1 \
        QOSMOE_ENABLE_LOAD_BALANCE=1 \
        QOSMOE_ENABLE_PREFETCH=1 \
        "${PYTHON}" src/run_all_moe_benchmarks.py \
            --project_root "${PROJECT_ROOT}" \
            --outputs_root "${RUN_OUT}" \
            --systems qosmoe \
            --tag "${TAG}" \
            --model_name "${MODEL_NAME}" \
            --dataset "${DATASET}" \
            --num_samples "${NUM_SAMPLES}" \
            --batch_size "${BATCH_SIZE}" \
            --seq_len "${SEQ_LEN}" \
            --world_size "${GPU_NUM}" \
            --warmup_rounds "${WARMUP_ROUNDS}" \
            --d_model "${D_MODEL}" \
            --num_experts "${NUM_EXPERTS}" \
            --expert_cache_size "${EXPERT_CACHE_SIZE}" \
            --eq_tokens "${EQ_TOKENS}" \
            --qos_target_batch "${QOS_TARGET_BATCH}" \
            --enable_router_skew "${ENABLE_ROUTER_SKEW}" \
            --router_skew "${ROUTER_SKEW}" \
            --router_num_experts_skewed "${HOT_EXPERTS}" \
            --capacity_factor "${CAPACITY_FACTOR}" \
            --cuda_visible_devices "${CUDA_DEVICES}" \
            --qos_w1 "${QOS_W1}" \
            --qos_w2 "${QOS_W2}" \
            --qos_w3 "${QOS_W3}" \
            --qos_delta "${QOS_DELTA}" \
            --qos_enable_prefetch 1 \
            --qos_prefetch_topk "${QOS_PREFETCH_TOPK}" \
            --qos_prefetch_budget_experts "${QOS_PREFETCH_BUDGET}" \
            --base_port "${PORT}" \
            --clean \
            2>&1 | tee "${LOG_FILE}"


    # --------------------------------------------------------
    # HarMoEny
    # --------------------------------------------------------

    elif [[ "${SYSTEM}" == "harmony" ]]; then

        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
        "${PYTHON}" src/run_all_moe_benchmarks.py \
            --project_root "${PROJECT_ROOT}" \
            --outputs_root "${RUN_OUT}" \
            --systems harmony \
            --tag "${TAG}" \
            --model_name "${MODEL_NAME}" \
            --dataset "${DATASET}" \
            --num_samples "${NUM_SAMPLES}" \
            --batch_size "${BATCH_SIZE}" \
            --seq_len "${SEQ_LEN}" \
            --world_size "${GPU_NUM}" \
            --warmup_rounds "${WARMUP_ROUNDS}" \
            --d_model "${D_MODEL}" \
            --num_experts "${NUM_EXPERTS}" \
            --expert_cache_size "${EXPERT_CACHE_SIZE}" \
            --eq_tokens "${EQ_TOKENS}" \
            --enable_router_skew "${ENABLE_ROUTER_SKEW}" \
            --router_skew "${ROUTER_SKEW}" \
            --router_num_experts_skewed "${HOT_EXPERTS}" \
            --capacity_factor "${CAPACITY_FACTOR}" \
            --cuda_visible_devices "${CUDA_DEVICES}" \
            --base_port "${PORT}" \
            --clean \
            2>&1 | tee "${LOG_FILE}"


    # --------------------------------------------------------
    # DeepSpeed
    # --------------------------------------------------------

    elif [[ "${SYSTEM}" == "deepspeed" ]]; then

        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
        "${PYTHON}" src/run_all_moe_benchmarks.py \
            --project_root "${PROJECT_ROOT}" \
            --outputs_root "${RUN_OUT}" \
            --systems deepspeed \
            --tag "${TAG}" \
            --model_name "${MODEL_NAME}" \
            --dataset "${DATASET}" \
            --num_samples "${NUM_SAMPLES}" \
            --batch_size "${BATCH_SIZE}" \
            --seq_len "${SEQ_LEN}" \
            --world_size "${GPU_NUM}" \
            --warmup_rounds "${WARMUP_ROUNDS}" \
            --d_model "${D_MODEL}" \
            --num_experts "${NUM_EXPERTS}" \
            --expert_cache_size "${EXPERT_CACHE_SIZE}" \
            --eq_tokens "${EQ_TOKENS}" \
            --enable_router_skew "${ENABLE_ROUTER_SKEW}" \
            --router_skew "${ROUTER_SKEW}" \
            --router_num_experts_skewed "${HOT_EXPERTS}" \
            --capacity_factor "${CAPACITY_FACTOR}" \
            --cuda_visible_devices "${CUDA_DEVICES}" \
            --base_port "${PORT}" \
            --clean \
            2>&1 | tee "${LOG_FILE}"

    fi


    EXIT_CODE=${PIPESTATUS[0]}


    if [[ ${EXIT_CODE} -eq 0 ]]; then

        echo "[SUCCESS] ${TAG}"

        echo \
            "${SYSTEM},${GPU_NUM},${REPEAT},SUCCESS" \
            >> "${OUT_ROOT}/run_status.csv"

    else

        echo "[FAILED] ${TAG}, exit=${EXIT_CODE}"

        echo \
            "${SYSTEM},${GPU_NUM},${REPEAT},FAILED" \
            >> "${OUT_ROOT}/run_status.csv"

    fi


    cleanup_runtime
}


# ------------------------------------------------------------
# 13. Main experiment
# ------------------------------------------------------------

echo "system,num_gpus,repeat,status" \
    > "${OUT_ROOT}/run_status.csv"


BASE_PORT=30201


# 固定系统顺序，避免不同 GPU 配置下方法顺序发生变化
SYSTEMS=(
    "deepspeed"
    "harmony"
    "qosmoe"
)


for GPU_NUM in "${GPU_NUMS[@]}"
do

    CUDA_DEVICES=$(get_devices "${GPU_NUM}")

    for REPEAT in $(seq 1 "${REPEATS}")
    do

        for SYSTEM in "${SYSTEMS[@]}"
        do

            run_one \
                "${SYSTEM}" \
                "${GPU_NUM}" \
                "${CUDA_DEVICES}" \
                "${REPEAT}" \
                "${BASE_PORT}"

            BASE_PORT=$((BASE_PORT + 10))

        done

    done

done


# ------------------------------------------------------------
# 14. Finish
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Scalability experiments finished."
echo
echo "Results:"
echo "  ${OUT_ROOT}"
echo
echo "Status:"
cat "${OUT_ROOT}/run_status.csv"
echo
echo "============================================================"