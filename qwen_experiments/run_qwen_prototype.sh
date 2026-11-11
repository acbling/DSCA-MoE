#!/usr/bin/env bash
set -euo pipefail
cd /home/lcf/MoE_Syn/HarMoEny-main/qwen_experiments
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
mkdir -p results
python qwen_validate.py
for MODE in native priority_prototype; do
  torchrun --standalone --nproc_per_node=4 qwen_benchmark.py --mode "$MODE" --out "results/qwen_${MODE}.json" --batch-size-per-gpu 4 --seq-len 256 2>&1 | tee "results/${MODE}.log"
done
