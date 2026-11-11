#!/usr/bin/env bash
# Ablations of chemberta__x_pubchemfp on the DGX: 23 variants x 5 seeds = 115 runs.
# Plan: paper/chemberta_pubchemfp_ablation_plan.md. Notes: experiments/ablations_chemberta/README.md
#
#   bash experiments/ablations_chemberta/run_dgx.sh                 # everything, all GPUs
#   NGPU=1 bash experiments/ablations_chemberta/run_dgx.sh          # one GPU
#   SKIP_REPRO_CHECK=1 bash experiments/ablations_chemberta/run_dgx.sh   # do not stop if step 0 drifts
#
# Safe to re-run and to Ctrl-C: a run is finished only when its COMPLETE sentinel exists (written
# last), and every re-run does exactly the runs that are missing.
#
# Order:
#   step 0   `full` x 5 seeds, then a check that it reproduces the grid's chemberta__x_pubchemfp
#            (val PR-AUC within 0.005 per seed). Stops here if not, because every comparison is
#            against this baseline.
#   step 1   the other 22 variants x 5 seeds, sharded over the GPUs
#   step 2   collect (validation only) + untrained-token subsets. Held-out stays sealed.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=${PY:-python}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
EPOCHS=${EPOCHS:-40}
LOGS=experiments/results/ablations_chemberta/logs
mkdir -p "$LOGS"

echo "=============================================================================="
echo "chemberta__x_pubchemfp ablations | $NGPU GPU(s) | max $EPOCHS epochs"
echo "=============================================================================="
"$PY" -m experiments.ablations_chemberta.run_seeds --dry-run | head -1

sweep () {   # sweep <tag> <run_seeds args...>
  local tag=$1; shift
  local pids=()
  if (( NGPU >= 2 )); then
    for (( i=0; i<NGPU; i++ )); do
      CUDA_VISIBLE_DEVICES=$i nohup "$PY" -m experiments.ablations_chemberta.run_seeds "$@" \
          --device cuda --epochs "$EPOCHS" --shard "$i/$NGPU" > "$LOGS/_sweep_${tag}_gpu${i}.log" 2>&1 &
      pids+=($!)
    done
  else
    CUDA_VISIBLE_DEVICES=0 nohup "$PY" -m experiments.ablations_chemberta.run_seeds "$@" \
        --device cuda --epochs "$EPOCHS" > "$LOGS/_sweep_${tag}.log" 2>&1 &
    pids+=($!)
  fi
  echo "  running (pids ${pids[*]}); follow with: tail -f $LOGS/_sweep_${tag}*.log"
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  "$PY" -m experiments.ablations_chemberta.run_seeds "$@" --dry-run | head -1
  if (( fail )); then
    echo "  NOTE: at least one run failed (logs in $LOGS/). Re-run this script to retry them."
  fi
}

# ---- step 0
echo
echo "--- step 0: full x 5 seeds"
sweep full --ablations full
"$PY" -m experiments.ablations_chemberta.collect --ablations full | sed -n '/Step 0/,/^## Abl/p'
if [[ -z "${SKIP_REPRO_CHECK:-}" ]]; then
  if "$PY" - <<'EOF'
import sys, pandas as pd
r = pd.read_csv("experiments/results/ablations_chemberta/reproduction.csv")
bad = r[~r["ok"].astype(bool)]
if len(bad):
    print("step 0 FAILED: harness full does not reproduce the grid for seed(s)",
          ", ".join(map(str, bad["seed"])), "\n", bad.to_string(index=False))
    sys.exit(1)
print("step 0 ok: harness full reproduces the grid on every seed")
EOF
  then :; else
    echo "Stopping. Check the device / library versions against the grid runs, or set"
    echo "SKIP_REPRO_CHECK=1 to continue and compare everything with this harness's full."
    exit 1
  fi
fi

# ---- step 1
echo
echo "--- step 1: the 22 ablations x 5 seeds"
sweep ablations

# ---- step 2
echo
echo "--- step 2: collect"
"$PY" -m experiments.ablations_chemberta.collect 2>&1 | tee "$LOGS/_collect.log"
"$PY" -m experiments.ablations_chemberta.token_subsets 2>&1 | tee "$LOGS/_token_subsets.log"

echo
echo "done. Tables: experiments/results/ablations_chemberta/summary.md, summary.csv,"
echo "             experiments/results/ablations_chemberta/token_subsets/"
echo "Held-out is still sealed. After the decision is locked, and only then:"
echo "  $PY -m experiments.ablations_chemberta.collect --reveal-heldout"
