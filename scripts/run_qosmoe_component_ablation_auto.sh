#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_qosmoe_component_ablation_auto.sh "python src/start_qosmoe.py YOUR_ORIGINAL_ARGS"
#
# Important:
#   Do not include --path in YOUR_ORIGINAL_ARGS. This script appends --path automatically.

CMD_BASE=${1:-}
if [ -z "$CMD_BASE" ]; then
  echo "Usage: bash scripts/run_qosmoe_component_ablation_auto.sh \"python src/start_qosmoe.py YOUR_ORIGINAL_ARGS\""
  exit 1
fi

OUT_ROOT=${OUT_ROOT:-ablation_results_qosmoe}
mkdir -p "$OUT_ROOT"

run_one () {
  local name=$1
  local async=$2
  local lb=$3
  local prefetch=$4
  local out_dir="$OUT_ROOT/$name"
  mkdir -p "$out_dir"
  echo "============================================================"
  echo "Running $name: async=$async load_balance=$lb prefetch=$prefetch"
  echo "Output: $out_dir"
  echo "============================================================"

  QOSMOE_ENABLE_ASYNC=$async \
  QOSMOE_ENABLE_LOAD_BALANCE=$lb \
  QOSMOE_ENABLE_PREFETCH=$prefetch \
  bash -lc "$CMD_BASE --path $out_dir" | tee "$out_dir/run.log"
}

run_one "QoSMoE-Full"        1 1 1
run_one "QoSMoE-wo-Async"    0 1 1
run_one "QoSMoE-wo-LB"       1 0 1
run_one "QoSMoE-wo-Prefetch" 1 1 0

python scripts/summarize_ablation_e2e.py --root "$OUT_ROOT"
