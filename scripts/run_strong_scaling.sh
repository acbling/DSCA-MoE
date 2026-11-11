#!/usr/bin/env bash

set -euo pipefail


cd /home/lcf/MoE_Syn/HarMoEny-main


PYTHON=/home/lcf/.conda/envs/harmony-clean/bin/python


PROJECT_ROOT=/home/lcf/MoE_Syn/HarMoEny-main


MODEL=/home/lcf/MoE_Syn/models/google_switch_base_64


OUT_ROOT=/home/lcf/MoE_Syn/HarMoEny-main/experiment_outputs/strong_scaling


mkdir -p ${OUT_ROOT}



SYSTEMS=(

deepspeed
harmony
qosmoe

)



# ============================
# workload
# ============================

NUM_SAMPLES=4096

BATCH_SIZE=64

SEQ_LEN=256

WARMUP=10



# ============================
# MoE
# ============================

NUM_EXPERTS=64

CACHE_SIZE=8

EQ_TOKENS=150

CAPACITY=8.0



# ============================
# GPU
# ============================


for GPU_CFG in \
"0 1" \
"0,1 2" \
"0,1,2,3 4"


do


CUDA=$(echo ${GPU_CFG}|awk '{print $1}')

WORLD=$(echo ${GPU_CFG}|awk '{print $2}')



for SYS in "${SYSTEMS[@]}"

do


TAG=strong_${SYS}_${WORLD}gpu



echo "=============================="

echo ${TAG}

echo "=============================="



${PYTHON} src/run_all_moe_benchmarks.py \

--project_root ${PROJECT_ROOT} \

--outputs_root ${OUT_ROOT}/${TAG} \

--systems ${SYS} \

--tag ${TAG} \

--model_name ${MODEL} \

--dataset wikitext \

--num_samples ${NUM_SAMPLES} \

--batch_size ${BATCH_SIZE} \

--seq_len ${SEQ_LEN} \

--world_size ${WORLD} \

--warmup_rounds ${WARMUP} \

--num_experts ${NUM_EXPERTS} \

--expert_cache_size ${CACHE_SIZE} \

--eq_tokens ${EQ_TOKENS} \

--capacity_factor ${CAPACITY} \

--enable_router_skew 1 \

--router_skew 0.9 \

--router_num_experts_skewed 8 \

--cuda_visible_devices ${CUDA} \

--clean



done

done



echo "Strong scaling finished"
