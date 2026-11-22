#!/usr/bin/env bash
# Ablations of dmpnn__x_maccs on the DGX: 23 variants x 5 seeds = 115 runs.
# Plan: paper/dmpnn_maccs_ablation_plan.md. Notes: experiments/ablations_dmpnn_maccs/README.md
#
#   bash experiments/ablations_dmpnn_maccs/run_dgx.sh                 # everything, all GPUs
#   NGPU=1 bash experiments/ablations_dmpnn_maccs/run_dgx.sh          # one GPU
#   SKIP_REPRO_CHECK=1 bash experiments/ablations_dmpnn_maccs/run_dgx.sh   # do not stop if step 0 drifts
#
# Safe to re-run and to Ctrl-C: a run is finished only when its COMPLETE sentinel exists (written
# last), and every re-run does exactly the runs that are missing.
#
# Order:
#   step 0   `full` x 5 seeds, then a check that it reproduces the grid's dmpnn__x_maccs
#            (val PR-AUC within 0.005 per seed). Stops here if not, because every comparison is
#            against this baseline.
#   step 1   the other 22 variants x 5 seeds, sharded over the GPUs
#   step 2   collect (validation only) + chemical subset diagnostics. Held-out stays sealed.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=${PY:-python}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
EPOCHS=${EPOCHS:-40}
LOGS=experiments/results/ablations_dmpnn_maccs/logs
mkdir -p "$LOGS"

echo "=============================================================================="
echo "dmpnn__x_maccs ablations | $NGPU GPU(s) | max $EPOCHS epochs"
echo "=============================================================================="
"$PY" -m experiments.ablations_dmpnn_maccs.run_seeds --dry-run | head -1

sweep () {   # sweep <tag> <run_seeds args...>
  local tag=$1; shift
  local pids=()
  if (( NGPU >= 2 )); then
    for (( i=0; i<NGPU; i++ )); do
      CUDA_VISIBLE_DEVICES=$i nohup "$PY" -m experiments.ablations_dmpnn_maccs.run_seeds "$@" \
          --device cuda --epochs "$EPOCHS" --shard "$i/$NGPU" > "$LOGS/_sweep_${tag}_gpu${i}.log" 2>&1 &
      pids+=($!)
    done
  else
    CUDA_VISIBLE_DEVICES=0 nohup "$PY" -m experiments.ablations_dmpnn_maccs.run_seeds "$@" \
        --device cuda --epochs "$EPOCHS" > "$LOGS/_sweep_${tag}.log" 2>&1 &
    pids+=($!)
  fi
  echo "  running (pids ${pids[*]}); follow with: tail -f $LOGS/_sweep_${tag}*.log"
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  "$PY" -m experiments.ablations_dmpnn_maccs.run_seeds "$@" --dry-run | head -1
  if (( fail )); then
    echo "  NOTE: at least one run failed (logs in $LOGS/). Re-run this script to retry them."
  fi
}

# ---- step 0
echo
echo "--- step 0: full x 5 seeds"
sweep full --ablations full
"$PY" -m experiments.ablations_dmpnn_maccs.collect --ablations full | sed -n '/Step 0/,/^## Abl/p'
if [[ -z "${SKIP_REPRO_CHECK:-}" ]]; then
  if ! "$PY" -c "
import pandas as pd
df = pd.read_csv('experiments/results/ablations_dmpnn_maccs/reproduction.csv')
assert df['ok'].all(), f'reproduction failed:\n{df}'
"; then
    echo "STEP 0 FAILED: harness baseline does not reproduce the grid (see above)."
    echo "Investigate before running the ablations, or re-run with SKIP_REPRO_CHECK=1 to bypass."
    exit 1
  fi
fi

# ---- step 1
echo
echo "--- step 1: remaining ablations x 5 seeds"
sweep rest

# ---- step 2
echo
echo "--- step 2: collect + chemical subgroup diagnostics"
"$PY" -m experiments.ablations_dmpnn_maccs.collect
"$PY" -m experiments.ablations_dmpnn_maccs.subsets
echo
echo "Done. Summary in experiments/results/ablations_dmpnn_maccs/summary.md"
echo "Held-out is STILL SEALED. Pass --reveal-heldout to collect.py only after the decision is locked."

