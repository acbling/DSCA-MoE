#!/usr/bin/env bash

set -uo pipefail

# ============================================================
# DSCA-MoE Comprehensive Scalability Evaluation
#
# Model    : Switch-base-64
# GPU      : A100-SXM4-40GB
# Systems  : DeepSpeed / HarMoEny / DSCA-MoE
#
# IMPORTANT:
#   local batch <= 16 to avoid DeepSpeed dispatch-buffer OOM.
# ============================================================


# ============================================================
# 0. Paths
# ============================================================

PYTHON=/home/lcf/.conda/envs/harmony-clean/bin/python

PROJECT_ROOT=/home/lcf/MoE_Syn/HarMoEny-main

MODEL_NAME=/home/lcf/MoE_Syn/models/google_switch_base_64

RUN_ID=$(date +%Y%m%d_%H%M%S)

OUT_ROOT=${PROJECT_ROOT}/experiment_outputs/complete_${RUN_ID}

LOG_ROOT=${OUT_ROOT}/logs


cd "${PROJECT_ROOT}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${LOG_ROOT}"


# ============================================================
# 1. Environment
# ============================================================

export HF_HOME=/home/lcf/.cache/harmoeny/hf
export HF_DATASETS_CACHE=/home/lcf/.cache/harmoeny/datasets

# 本地 cache 已经确认可用
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 减少 CUDA allocator 碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 保持与你之前实验环境一致
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_LEVEL=SYS


# ============================================================
# 2. Common parameters
# ============================================================

NUM_EXPERTS=64

D_MODEL=768

EQ_TOKENS=150

CAPACITY_FACTOR=8.0

WARMUP_ROUNDS=10

HOT_EXPERTS=8

DEFAULT_CACHE=8

DEFAULT_BATCH=16

DEFAULT_SEQ=256

DEFAULT_SAMPLES=4096

DEFAULT_DEADLINE=0.34


# QoSMoE
QOS_W1=0.1
QOS_W2=0.0
QOS_W3=1.0

QOS_DELTA=0.02

QOS_PREFETCH_TOPK=1
QOS_PREFETCH_BUDGET=1


# ============================================================
# 3. Repeats
# ============================================================

REPEATS=${REPEATS:-3}


# ============================================================
# 4. Experiment switches
#
# 可通过环境变量单独关闭，例如：
#
# RUN_SLO=0 bash scripts/run_complete_scalability.sh
# ============================================================

RUN_STRONG=${RUN_STRONG:-1}

RUN_WEAK=${RUN_WEAK:-1}

RUN_BATCH=${RUN_BATCH:-1}

RUN_SKEW=${RUN_SKEW:-1}

RUN_CACHE=${RUN_CACHE:-1}

RUN_SLO=${RUN_SLO:-1}

RUN_SEQUENCE=${RUN_SEQUENCE:-1}


# ============================================================
# 5. Status
# ============================================================

STATUS_FILE=${OUT_ROOT}/run_status.csv

echo \
"family,system,gpus,batch,num_samples,seq_len,cache,skew,deadline,repeat,status,log" \
> "${STATUS_FILE}"


# ============================================================
# 6. GPU mapping
# ============================================================

get_devices () {

    local world=$1

    case "${world}" in

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
            echo "Unsupported world size: ${world}" >&2
            exit 1
            ;;

    esac
}


# ============================================================
# 7. Runtime cleanup
# ============================================================

cleanup_runtime () {

    pkill -9 -f "src/start_deepspeed.py" 2>/dev/null || true
    pkill -9 -f "src/start_harmony.py" 2>/dev/null || true
    pkill -9 -f "src/start_qosmoe.py" 2>/dev/null || true
    pkill -9 -f "deepspeed.launcher" 2>/dev/null || true
    pkill -9 -f "torchrun" 2>/dev/null || true

    sleep 3
}


# ============================================================
# 8. Port
# ============================================================

BASE_PORT=32001


# ============================================================
# 9. Run one configuration
#
# args:
#   family
#   system
#   world
#   devices
#   batch
#   samples
#   seq
#   cache
#   skew_enable
#   skew
#   deadline
#   repeat
# ============================================================

run_one () {

    local FAMILY=$1
    local SYSTEM=$2
    local WORLD=$3
    local DEVICES=$4
    local BATCH=$5
    local NUM_SAMPLES=$6
    local SEQ=$7
    local CACHE=$8
    local SKEW_ENABLE=$9

    shift 9

    local SKEW=$1
    local DEADLINE=$2
    local REPEAT=$3


    local TAG="${FAMILY}_${SYSTEM}_${WORLD}gpu_r${REPEAT}"

    local RUN_OUT="${OUT_ROOT}/${FAMILY}/${TAG}"

    local FAMILY_LOG_DIR="${LOG_ROOT}/${FAMILY}"

    local LOG_FILE="${FAMILY_LOG_DIR}/${TAG}.log"


    mkdir -p "${RUN_OUT}"
    mkdir -p "${FAMILY_LOG_DIR}"


    echo
    echo "============================================================"
    echo "Family       : ${FAMILY}"
    echo "System       : ${SYSTEM}"
    echo "GPU          : ${WORLD}"
    echo "Devices      : ${DEVICES}"
    echo "Batch/rank   : ${BATCH}"
    echo "Global batch : $((BATCH * WORLD))"
    echo "Samples      : ${NUM_SAMPLES}"
    echo "Seq length   : ${SEQ}"
    echo "Cache        : ${CACHE}"
    echo "Router skew  : ${SKEW} (enable=${SKEW_ENABLE})"
    echo "Deadline     : ${DEADLINE}"
    echo "Repeat       : ${REPEAT}/${REPEATS}"
    echo "Port         : ${BASE_PORT}"
    echo "Output       : ${RUN_OUT}"
    echo "============================================================"


    cleanup_runtime


    CMD=(

        "${PYTHON}"
        src/run_all_moe_benchmarks.py

        --project_root "${PROJECT_ROOT}"

        --outputs_root "${RUN_OUT}"

        --systems "${SYSTEM}"

        --tag "${TAG}"

        --model_name "${MODEL_NAME}"

        --dataset wikitext

        --num_samples "${NUM_SAMPLES}"

        --batch_size "${BATCH}"

        --seq_len "${SEQ}"

        --world_size "${WORLD}"

        --warmup_rounds "${WARMUP_ROUNDS}"

        --d_model "${D_MODEL}"

        --num_experts "${NUM_EXPERTS}"

        --expert_cache_size "${CACHE}"

        --eq_tokens "${EQ_TOKENS}"

        --capacity_factor "${CAPACITY_FACTOR}"

        --enable_router_skew "${SKEW_ENABLE}"

        --enable_router_random 0

        --enable_router_uniform 0

        --router_skew "${SKEW}"

        --router_num_experts_skewed "${HOT_EXPERTS}"

        --cuda_visible_devices "${DEVICES}"

        --base_port "${BASE_PORT}"

    )


    ENV_ARGS=(

        "CUDA_VISIBLE_DEVICES=${DEVICES}"

    )


    # QoSMoE-specific parameters
    if [[ "${SYSTEM}" == "qosmoe" ]]; then

        CMD+=(

            --qos_target_batch "${BATCH}"

            --qos_default_deadline_s "${DEADLINE}"

            --qos_w1 "${QOS_W1}"

            --qos_w2 "${QOS_W2}"

            --qos_w3 "${QOS_W3}"

            --qos_delta "${QOS_DELTA}"

            --qos_enable_prefetch 1

            --qos_prefetch_topk "${QOS_PREFETCH_TOPK}"

            --qos_prefetch_budget_experts "${QOS_PREFETCH_BUDGET}"

        )


        ENV_ARGS+=(

            "QOSMOE_ENABLE_ASYNC=1"

            "QOSMOE_ENABLE_LOAD_BALANCE=1"

            "QOSMOE_ENABLE_PREFETCH=1"

        )

    fi


    echo
    echo "[COMMAND]"

    printf '%q ' env "${ENV_ARGS[@]}" "${CMD[@]}"

    echo
    echo


    if env "${ENV_ARGS[@]}" "${CMD[@]}" \
        2>&1 | tee "${LOG_FILE}"
    then

        RUN_STATUS="SUCCESS"

        echo
        echo "[SUCCESS] ${TAG}"

    else

        RUN_STATUS="FAILED"

        echo
        echo "[FAILED] ${TAG}"

    fi


    echo \
"${FAMILY},${SYSTEM},${WORLD},${BATCH},${NUM_SAMPLES},${SEQ},${CACHE},${SKEW},${DEADLINE},${REPEAT},${RUN_STATUS},${LOG_FILE}" \
    >> "${STATUS_FILE}"


    BASE_PORT=$((BASE_PORT + 1))


    cleanup_runtime
}


# ============================================================
# 10. Preflight
# ============================================================

echo
echo "============================================================"
echo "Dataset preflight"
echo "============================================================"

"${PYTHON}" - <<'PY'
import os
from datasets import load_dataset

cache = os.environ["HF_DATASETS_CACHE"]

ds = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="train[:16]",
    cache_dir=cache,
)

print("[OK] WikiText cache:", len(ds))
PY

if [[ $? -ne 0 ]]; then

    echo "[FATAL] WikiText preflight failed."

    exit 1

fi


# ============================================================
# 11. Metadata
# ============================================================

cat > "${OUT_ROOT}/experiment_meta.txt" <<EOF

Model=Switch-base-64
ModelPath=${MODEL_NAME}

GPU=A100-SXM4-40GB
GPUConfigs=1,2,4

NumExperts=${NUM_EXPERTS}
DModel=${D_MODEL}

DefaultBatch=${DEFAULT_BATCH}
DefaultSeq=${DEFAULT_SEQ}
DefaultSamples=${DEFAULT_SAMPLES}

CapacityFactor=${CAPACITY_FACTOR}
EqTokens=${EQ_TOKENS}

Repeats=${REPEATS}

NOTE:
batch_size is LOCAL batch per rank.
System throughput must be computed as:

system_throughput =
reported_samples_per_s * world_size

EOF


SYSTEMS=(

    "deepspeed"
    "harmony"
    "qosmoe"

)


# ============================================================
# 12. EXP-1: Strong scaling
#
# Fixed total dataset = 4096 samples
# local batch = 16
#
# world 1: 256 iterations/rank
# world 2: 128 iterations/rank
# world 4:  64 iterations/rank
# ============================================================

if [[ "${RUN_STRONG}" == "1" ]]; then

    echo
    echo "############################################################"
    echo "# EXP-1 Strong Scaling"
    echo "############################################################"


    for WORLD in 1 2 4
    do

        DEVICES=$(get_devices "${WORLD}")


        for REPEAT in $(seq 1 "${REPEATS}")
        do

            for SYSTEM in "${SYSTEMS[@]}"
            do

                run_one \
                    "strong" \
                    "${SYSTEM}" \
                    "${WORLD}" \
                    "${DEVICES}" \
                    16 \
                    4096 \
                    256 \
                    8 \
                    1 \
                    0.9 \
                    "${DEFAULT_DEADLINE}" \
                    "${REPEAT}"

            done

        done

    done

fi


# ============================================================
# 13. EXP-2: Weak scaling
#
# Fixed work per GPU.
#
# 1 GPU : 2048 total samples
# 2 GPU : 4096 total samples
# 4 GPU : 8192 total samples
#
# local batch is always 16.
# Each rank therefore runs ~128 iterations.
# ============================================================

if [[ "${RUN_WEAK}" == "1" ]]; then

    echo
    echo "############################################################"
    echo "# EXP-2 Weak Scaling"
    echo "############################################################"


    for WORLD in 1 2 4
    do

        DEVICES=$(get_devices "${WORLD}")

        NUM_SAMPLES=$((2048 * WORLD))


        for REPEAT in $(seq 1 "${REPEATS}")
        do

            for SYSTEM in "${SYSTEMS[@]}"
            do

                run_one \
                    "weak" \
                    "${SYSTEM}" \
                    "${WORLD}" \
                    "${DEVICES}" \
                    16 \
                    "${NUM_SAMPLES}" \
                    256 \
                    8 \
                    1 \
                    0.9 \
                    "${DEFAULT_DEADLINE}" \
                    "${REPEAT}"

            done

        done

    done

fi


# ============================================================
# 14. EXP-3: Batch sensitivity
#
# All are <= previously verified B16.
# ============================================================

if [[ "${RUN_BATCH}" == "1" ]]; then

    echo
    echo "############################################################"
    echo "# EXP-3 Batch Sensitivity"
    echo "############################################################"


    WORLD=4
    DEVICES="0,1,2,3"


    for BATCH in 4 8 16
    do

        for REPEAT in $(seq 1 "${REPEATS}")
        do

            for SYSTEM in "${SYSTEMS[@]}"
            do

                run_one \
                    "batch_b${BATCH}" \
                    "${SYSTEM}" \
                    "${WORLD}" \
                    "${DEVICES}" \
                    "${BATCH}" \
                    4096 \
                    256 \
                    8 \
                    1 \
                    0.9 \
                    "${DEFAULT_DEADLINE}" \
                    "${REPEAT}"

            done

        done

    done

fi


# ============================================================
# 15. EXP-4: Router skew sensitivity
#
# natural:
#   original router
#
# moderate:
#   skew = 0.7
#
# severe:
#   skew = 0.9
# ============================================================

if [[ "${RUN_SKEW}" == "1" ]]; then

    echo
    echo "############################################################"
    echo "# EXP-4 Router Skew Sensitivity"
    echo "############################################################"


    WORLD=4
    DEVICES="0,1,2,3"


    for SPEC in \
        "natural 0 0.0" \
        "moderate 1 0.7" \
        "severe 1 0.9"
    do

        NAME=$(echo "${SPEC}" | awk '{print $1}')

        ENABLE=$(echo "${SPEC}" | awk '{print $2}')

        SKEW=$(echo "${SPEC}" | awk '{print $3}')


        for REPEAT in $(seq 1 "${REPEATS}")
        do

            for SYSTEM in "${SYSTEMS[@]}"
            do

                run_one \
                    "skew_${NAME}" \
                    "${SYSTEM}" \
                    "${WORLD}" \
                    "${DEVICES}" \
                    16 \
                    4096 \
                    256 \
                    8 \
                    "${ENABLE}" \
                    "${SKEW}" \
                    "${DEFAULT_DEADLINE}" \
                    "${REPEAT}"

            done

        done

    done

fi


# ============================================================
# 16. EXP-5: Expert-cache pressure
#
# DeepSpeed has no equivalent dynamic expert cache mechanism,
# so this experiment compares HarMoEny and DSCA-MoE.
# ============================================================

if [[ "${RUN_CACHE}" == "1" ]]; then

    echo
    echo "############################################################"
    echo "# EXP-5 Expert Cache Pressure"
    echo "############################################################"


    WORLD=4
    DEVICES="0,1,2,3"


    for CACHE in 2 4 8
    do

        for REPEAT in $(seq 1 "${REPEATS}")
        do

            for SYSTEM in harmony qosmoe
            do

                run_one \
                    "cache_c${CACHE}" \
                    "${SYSTEM}" \
                    "${WORLD}" \
                    "${DEVICES}" \
                    16 \
                    4096 \
                    256 \
                    "${CACHE}" \
                    1 \
                    0.9 \
                    "${DEFAULT_DEADLINE}" \
                    "${REPEAT}"

            done

        done

    done

fi


# ============================================================
# 17. EXP-6: SLO sensitivity
#
# Baseline DeepSpeed / HarMoEny distributions can be reused
# from strong 4-GPU results and evaluated post-hoc against
# the same deadlines.
#
# QoSMoE must be rerun because deadline affects scheduling.
# ============================================================

if [[ "${RUN_SLO}" == "1" ]]; then

    echo
    echo "############################################################"
    echo "# EXP-6 SLO Sensitivity"
    echo "############################################################"


    WORLD=4
    DEVICES="0,1,2,3"


    for DEADLINE in 0.30 0.40 0.50 0.70
    do

        for REPEAT in $(seq 1 "${REPEATS}")
        do

            run_one \
                "slo_d${DEADLINE}" \
                "qosmoe" \
                "${WORLD}" \
                "${DEVICES}" \
                16 \
                4096 \
                256 \
                8 \
                1 \
                0.9 \
                "${DEADLINE}" \
                "${REPEAT}"

        done

    done

fi


# ============================================================
# 18. EXP-7: Sequence-length sensitivity
#
# Uses only lengths already known to be safe.
# ============================================================

if [[ "${RUN_SEQUENCE}" == "1" ]]; then

    echo
    echo "############################################################"
    echo "# EXP-7 Sequence Length Sensitivity"
    echo "############################################################"


    WORLD=4
    DEVICES="0,1,2,3"


    for SEQ in 64 128 256
    do

        for REPEAT in $(seq 1 "${REPEATS}")
        do

            for SYSTEM in "${SYSTEMS[@]}"
            do

                run_one \
                    "seq_l${SEQ}" \
                    "${SYSTEM}" \
                    "${WORLD}" \
                    "${DEVICES}" \
                    16 \
                    4096 \
                    "${SEQ}" \
                    8 \
                    1 \
                    0.9 \
                    "${DEFAULT_DEADLINE}" \
                    "${REPEAT}"

            done

        done

    done

fi


# ============================================================
# 19. Finish
# ============================================================

echo
echo "============================================================"
echo "ALL EXPERIMENTS FINISHED"
echo "============================================================"

echo
echo "Output:"
echo "${OUT_ROOT}"

echo
echo "Status:"
cat "${STATUS_FILE}"

echo
echo "Backup recommendation:"
echo "cp -a ${OUT_ROOT} ${OUT_ROOT}_backup"

echo
echo "============================================================"
