#!/usr/bin/env bash
# Spread the top-5 x 5-seed runs over the DGX GPUs: one model per GPU (round-robin), all five
# seeds of that model run in sequence on it. Logs go to experiments/top5_seeds/logs/.
#
#   bash experiments/top5_seeds/run_dgx.sh            # use every visible GPU
#   NGPU=2 bash experiments/top5_seeds/run_dgx.sh     # use only GPUs 0 and 1
#
# Safe to re-run: finished (model, seed) pairs are skipped. Afterwards:
#   python experiments/top5_seeds/collect.py
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=${PY:-python}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
LOGS=experiments/top5_seeds/logs
mkdir -p "$LOGS"

MODELS=(
  chemberta_cross_attn_cls_bce
  pgb_dmpnn_chemprop_cross_attn_cls_asl
  chemberta_cross_attn_global_gated_bce
  pgb_molformer_cross_attn_global_gated_asl
  dmpnn_chemprop_cross_attn_cls_asl
)

echo "${#MODELS[@]} models on $NGPU GPU(s)"
pids=()
for g in $(seq 0 $((NGPU - 1))); do
  mine=()
  for i in "${!MODELS[@]}"; do
    if (( i % NGPU == g )); then mine+=("${MODELS[$i]}"); fi
  done
  (( ${#mine[@]} )) || continue
  echo "GPU $g: ${mine[*]}"
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" experiments/top5_seeds/run_seeds.py --device cuda \
      --models "${mine[@]}" > "$LOGS/gpu$g.log" 2>&1 &
  pids+=($!)
done

echo "running (pids ${pids[*]}); follow with: tail -f $LOGS/gpu*.log"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
"$PY" experiments/top5_seeds/run_seeds.py --list | tail -1
"$PY" experiments/top5_seeds/collect.py || true
exit $fail
