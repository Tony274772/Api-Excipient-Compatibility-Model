#!/usr/bin/env bash
# Salt-context ablation on the DGX: salt_norm on GPU 0, salt_norm_nowater on GPU 1, seeds 42-46 each.
# With one GPU both arms run one after the other. The control arm is not retrained: it is
# experiments/results/reagent_ablation/control/, which must be present.
#
#   bash experiments/salt_ablation/run_dgx.sh
#   NGPU=1 bash experiments/salt_ablation/run_dgx.sh
#
# Safe to re-run: finished (arm, seed) runs are skipped. Collects the results at the end.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=${PY:-python}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
LOGS=experiments/salt_ablation/logs
mkdir -p "$LOGS"

if [ ! -d experiments/results/reagent_ablation/control ]; then
  echo "experiments/results/reagent_ablation/control/ is missing (the raw-salt control runs)"; exit 1
fi

pids=()
if (( NGPU >= 2 )); then
  for pair in "0 salt_norm" "1 salt_norm_nowater"; do
    set -- $pair
    CUDA_VISIBLE_DEVICES=$1 nohup "$PY" experiments/salt_ablation/run.py --device cuda --arms "$2" \
        > "$LOGS/$2.log" 2>&1 &
    pids+=($!)
    echo "GPU $1: $2 (log $LOGS/$2.log)"
  done
else
  CUDA_VISIBLE_DEVICES=0 nohup "$PY" experiments/salt_ablation/run.py --device cuda > "$LOGS/both.log" 2>&1 &
  pids+=($!)
  echo "GPU 0: salt_norm then salt_norm_nowater (log $LOGS/both.log)"
fi

echo "running (pids ${pids[*]}); follow with: tail -f $LOGS/*.log"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
"$PY" experiments/salt_ablation/run.py --list | tail -1
"$PY" experiments/salt_ablation/collect.py || true
exit $fail
