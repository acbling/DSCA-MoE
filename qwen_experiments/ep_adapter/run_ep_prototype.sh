#!/usr/bin/env bash
source /home/lcf/.conda/etc/profile.d/conda.sh
source /home/lcf/.conda/etc/profile.d/conda.sh
set -euo pipefail
cd /home/lcf/MoE_Syn/HarMoEny-main
source /home/lcf/.conda/etc/profile.d/conda.sh 2>/dev/null || true
mkdir -p qwen_experiments/ep_results
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export QWEN_MODEL=/home/lcf/MoE_Syn/models/Qwen1.5-MoE-A2.7B
DIR=qwen_experiments/ep_adapter
if ! torchrun --standalone --nproc_per_node=4 "$DIR/validate_ep.py" 2>&1 | tee qwen_experiments/ep_results/validation.log; then
  echo 'Validation failed: stopping without benchmark numbers' >&2
  exit 1
fi
for policy in fifo priority; do
  torchrun --standalone --nproc_per_node=4 "$DIR/qwen_ep.py" \
    --policy "$policy" --batch-per-rank 4 --seq-len 256 \
    --iterations 8 --warmup 2 \
    --result "qwen_experiments/ep_results/${policy}.json" \
    2>&1 | tee "qwen_experiments/ep_results/${policy}.log"
done
python "$DIR/report_ep.py" --folder qwen_experiments/ep_results
