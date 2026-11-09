#!/usr/bin/env bash
# The 9x9 encoder grid on the DGX: 101 cells x 5 seeds = 505 runs.
# Plan: paper/encoder_grid_plan.md. Operating notes: experiments/encoder_comparison/README_GRID.md
#
#   bash experiments/encoder_comparison/run_grid_dgx.sh              # everything, all GPUs
#   NGPU=1 bash experiments/encoder_comparison/run_grid_dgx.sh       # force one GPU
#   BLOCKS="floor pooling" bash experiments/encoder_comparison/run_grid_dgx.sh
#   STOP_AFTER=fp_secondary bash experiments/encoder_comparison/run_grid_dgx.sh
#
# Safe to re-run and safe to Ctrl-C: a run counts as finished only when its COMPLETE sentinel
# exists, and that is written after every output file is on disk. Anything killed part-way is
# redone on the next invocation.
#
# Blocks run IN ORDER and the script waits for each to finish before starting the next. That is
# deliberate: pooling has to be settled before any deep row is believed, and the shuffled and
# no-cross-attention controls decide how every later number is read. Within a block the work is
# split across the GPUs by shard.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=${PY:-python}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
EPOCHS=${EPOCHS:-40}
LOGS=experiments/results/encoder_grid/_logs
mkdir -p "$LOGS"

ALL_BLOCKS="floor pooling shuffled no_cross_attn fp_secondary emb_secondary"
BLOCKS=${BLOCKS:-$ALL_BLOCKS}
STOP_AFTER=${STOP_AFTER:-}

echo "=============================================================================="
echo "encoder grid | $NGPU GPU(s) | max $EPOCHS epochs | blocks: $BLOCKS"
echo "=============================================================================="

# ---- step 0a: the 5 encoders' mean-pooled vectors, once. Skips tables that already exist.
# A single encoder that cannot even be built (gin needs dgl and dgllife) must not abandon the
# sweep: precompute refuses to WRITE a bad table, so a failure here means "no table", and
# run_all_grid skips exactly the cells that needed it and names them. The remaining cells run.
echo
echo "--- step 0a: precompute the mean-pooled encoder embeddings"
set +e
CUDA_VISIBLE_DEVICES=0 "$PY" -m experiments.encoder_comparison.precompute_embeddings \
    --device cuda 2>&1 | tee "$LOGS/_precompute.log"
precompute_rc=${PIPESTATUS[0]}
set -e
if (( precompute_rc != 0 )); then
  echo
  echo "  NOTE: at least one encoder has no embedding table (see $LOGS/_precompute.log)."
  echo "        Its cells will be skipped and listed by name. Install what it needs, re-run"
  echo "        precompute_embeddings, then re-run this script to fill them in."
fi

# ---- step 0b: audit every representation before anything long starts.
# Exits non-zero on a real data defect, so `set -e` stops the sweep here rather than after
# 25 GPU-hours of runs reading a bad table. A missing table is reported as BLOCKED, not a defect.
echo
echo "--- step 0b: audit the representations"
"$PY" -m experiments.encoder_comparison.audit_reps 2>&1 | tee "$LOGS/_audit.log"

# ---- steps 1-4: the blocks, in order, sharded across the GPUs
for block in $BLOCKS; do
  echo
  echo "=============================================================================="
  echo "block: $block"
  echo "=============================================================================="
  pids=()
  if (( NGPU >= 2 )); then
    for (( i=0; i<NGPU; i++ )); do
      CUDA_VISIBLE_DEVICES=$i nohup "$PY" -m experiments.encoder_comparison.run_all_grid \
          --blocks "$block" --device cuda --epochs "$EPOCHS" --shard "$i/$NGPU" \
          > "$LOGS/_sweep_${block}_gpu${i}.log" 2>&1 &
      pids+=($!)
      echo "  GPU $i: shard $i/$NGPU (log $LOGS/_sweep_${block}_gpu${i}.log)"
    done
  else
    CUDA_VISIBLE_DEVICES=0 nohup "$PY" -m experiments.encoder_comparison.run_all_grid \
        --blocks "$block" --device cuda --epochs "$EPOCHS" \
        > "$LOGS/_sweep_${block}.log" 2>&1 &
    pids+=($!)
    echo "  GPU 0: the whole block (log $LOGS/_sweep_${block}.log)"
  fi

  echo "  running (pids ${pids[*]}); follow with: tail -f $LOGS/_sweep_${block}*.log"
  block_fail=0
  for p in "${pids[@]}"; do wait "$p" || block_fail=1; done

  # Report what is left rather than stopping: one failed run should not abandon the sweep, and
  # re-running this script picks the stragglers up.
  "$PY" -m experiments.encoder_comparison.run_all_grid --blocks "$block" --dry-run | tail -3
  if (( block_fail )); then
    echo "  NOTE: at least one run in '$block' failed. Its log is in $LOGS/. Re-running this"
    echo "        script retries only the runs without a COMPLETE sentinel."
  fi

  if [[ -n "$STOP_AFTER" && "$block" == "$STOP_AFTER" ]]; then
    echo
    echo "STOP_AFTER=$STOP_AFTER reached; not starting the remaining blocks."
    break
  fi
done

# ---- step 5: collect. Held-out stays sealed; --reveal-heldout is a separate, deliberate call.
echo
echo "=============================================================================="
echo "collecting"
echo "=============================================================================="
"$PY" -m experiments.encoder_comparison.collect_grid \
    --csv-out paper/encoder_grid.csv \
    --matrix-out paper/encoder_grid_matrix.csv 2>&1 | tee "$LOGS/_collect.log"

echo
echo "done. Results: experiments/results/encoder_grid/"
echo "      tables : paper/encoder_grid.csv, paper/encoder_grid_matrix.csv"
echo "held-out is still sealed. After the decision is locked, and only then:"
echo "  $PY -m experiments.encoder_comparison.collect_grid --reveal-heldout"
