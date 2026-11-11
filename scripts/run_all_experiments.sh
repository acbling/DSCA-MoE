#!/usr/bin/env bash

set -euo pipefail


cd /home/lcf/MoE_Syn/HarMoEny-main


PYTHON=/home/lcf/.conda/envs/harmony-clean/bin/python


PROJECT_ROOT=/home/lcf/MoE_Syn/HarMoEny-main


MODEL_NAME=/home/lcf/MoE_Syn/models/google_switch_base_64


OUTPUT_ROOT=/home/lcf/MoE_Syn/HarMoEny-main/experiment_outputs



mkdir -p ${OUTPUT_ROOT}



SYSTEMS=(
    deepspeed
    harmony
    qosmoe
)



COMMON_ARGS="
--project_root ${PROJECT_ROOT}
--model_name ${MODEL_NAME}
--dataset wikitext
--num_samples 4096
--seq_len 256
--warmup_rounds 10
--num_experts 64
--expert_cache_size 8
--eq_tokens 150
--capacity_factor 8.0
--enable_router_skew 1
--router_skew 0.9
--router_num_experts_skewed 8
"



run_exp(){

    NAME=$1
    SYS=$2
    GPU=$3
    WORLD=$4
    BATCH=$5
    EXTRA=$6


    OUT=${OUTPUT_ROOT}/${NAME}/${SYS}_${WORLD}gpu


    mkdir -p ${OUT}


    echo "=============================================="
    echo "Experiment: ${NAME}"
    echo "System: ${SYS}"
    echo "GPU: ${GPU}"
    echo "World: ${WORLD}"
    echo "Batch: ${BATCH}"
    echo "=============================================="


    CUDA_VISIBLE_DEVICES=${GPU} \
    ${PYTHON} src/run_all_moe_benchmarks.py \
        --outputs_root ${OUT} \
        --systems ${SYS} \
        --tag ${NAME}_${SYS}_${WORLD}gpu \
        --batch_size ${BATCH} \
        --world_size ${WORLD} \
        --cuda_visible_devices ${GPU} \
        ${COMMON_ARGS} \
        ${EXTRA}



}



###################################################
# 1. Strong Scaling
###################################################

echo "======== Strong Scaling ========"


for cfg in \
"0 1 64" \
"0,1 2 64" \
"0,1,2,3 4 64"

do

GPU=$(echo ${cfg}|awk '{print $1}')
WORLD=$(echo ${cfg}|awk '{print $2}')
BS=$(echo ${cfg}|awk '{print $3}')


for SYS in "${SYSTEMS[@]}"
do

run_exp \
strong_scaling \
${SYS} \
${GPU} \
${WORLD} \
${BS} \
""

done

done





###################################################
# 2. Weak Scaling
###################################################

echo "======== Weak Scaling ========"


for cfg in \
"0 1 32" \
"0,1 2 64" \
"0,1,2,3 4 128"

do


GPU=$(echo ${cfg}|awk '{print $1}')
WORLD=$(echo ${cfg}|awk '{print $2}')
BS=$(echo ${cfg}|awk '{print $3}')


for SYS in "${SYSTEMS[@]}"
do

run_exp \
weak_scaling \
${SYS} \
${GPU} \
${WORLD} \
${BS} \
""

done

done





###################################################
# 3. Router skew sensitivity
###################################################

echo "======== Router Skew ========"


for SKEW in 0.0 0.7 0.9

do


for SYS in "${SYSTEMS[@]}"
do


run_exp \
router_skew_${SKEW} \
${SYS} \
0,1,2,3 \
4 \
64 \
"--router_skew ${SKEW}"



done

done






###################################################
# 4. SLO sensitivity
###################################################

echo "======== SLO ========"


for DEADLINE in 0.2 0.3 0.4 0.5

do


run_exp \
slo_${DEADLINE} \
qosmoe \
0,1,2,3 \
4 \
64 \
"--qos_default_deadline_s ${DEADLINE}"



done





echo ""
echo "======================================"
echo "All experiments finished"
echo "Results:"
echo ${OUTPUT_ROOT}
echo "======================================"
