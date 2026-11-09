#!/usr/bin/env bash
# Reagent ablation on the DGX: builds the reagent-free data, then runs the two arms in parallel,
# one arm per GPU (control on GPU 0, no_reagents on GPU 1), seeds 42-46 in sequence on each.
# With one GPU both arms run one after the other on it. Logs go to experiments/reagent_ablation/logs/.
#
#   bash experiments/reagent_ablation/run_dgx.sh
#   NGPU=1 bash experiments/reagent_ablation/run_dgx.sh
#
# Safe to re-run: finished (arm, seed) runs are skipped. Collects the results at the end.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=${PY:-python}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
LOGS=experiments/reagent_ablation/logs
mkdir -p "$LOGS"

"$PY" experiments/reagent_ablation/reagents.py

pids=()
if (( NGPU >= 2 )); then
  for pair in "0 control" "1 no_reagents"; do
    set -- $pair
    CUDA_VISIBLE_DEVICES=$1 nohup "$PY" experiments/reagent_ablation/run.py --device cuda --arms "$2" \
        > "$LOGS/$2.log" 2>&1 &
    pids+=($!)
    echo "GPU $1: $2 (log $LOGS/$2.log)"
  done
else
  CUDA_VISIBLE_DEVICES=0 nohup "$PY" experiments/reagent_ablation/run.py --device cuda \
      > "$LOGS/both.log" 2>&1 &
  pids+=($!)
  echo "GPU 0: control then no_reagents (log $LOGS/both.log)"
fi

echo "running (pids ${pids[*]}); follow with: tail -f $LOGS/*.log"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
"$PY" experiments/reagent_ablation/run.py --list | tail -1
"$PY" experiments/reagent_ablation/collect.py || true
exit $fail
