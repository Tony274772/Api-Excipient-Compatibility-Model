# Top 5 grid models x 5 seeds

The five best configurations of the 80/20 full grid, ranked by **validation PR-AUC** in
`experiments/results/full_grid/summary.csv` (seed 42), retrained with seeds 42-46.

| # | Run | Head | Val PR-AUC (seed 42) |
|---|---|---|---|
| 1 | `chemberta_cross_attn_cls_bce` | baseline | 0.7627 |
| 2 | `pgb_dmpnn_chemprop_cross_attn_cls_asl` | PGB | 0.7545 |
| 3 | `chemberta_cross_attn_global_gated_bce` | baseline | 0.7398 |
| 4 | `pgb_molformer_cross_attn_global_gated_asl` | PGB | 0.7324 |
| 5 | `dmpnn_chemprop_cross_attn_cls_asl` | baseline | 0.7317 |

Everything except the seed is the grid recipe: `data/grid_8020/` (train 2770 / val 774), the same
early stopping (patience 6 baseline / 7 PGB, min-delta 0.002), threshold tuned on validation, and
the 22 external held-out pairs reported as `test`. Training goes through
`experiments/full_grid/run_grid.py:run_one`, so the code path is identical to the grid.

## On the DGX

Needs `data/grid_8020/` (from `python experiments/full_grid/prepare_data.py`) and the same
environment the grid ran in.

```bash
bash experiments/top5_seeds/run_dgx.sh            # one model per GPU, 5 seeds each
```

or on a single GPU:

```bash
python experiments/top5_seeds/run_seeds.py --device cuda
```

Both are resumable: a (model, seed) with a `run_info.json` is skipped. `--list` shows progress,
`--models` / `--seeds` narrow the run, `--overwrite` retrains.

## Output

```
experiments/results/top5_seeds/seed_<s>/<run>/   val_metrics.json, test_metrics.json, run_info.json, ...
checkpoints/top5_seeds/seed_<s>/<run>/           best_model.pt
```

```bash
python experiments/top5_seeds/collect.py
```

writes `experiments/results/top5_seeds/per_seed.csv` and `summary.csv` (mean and std over seeds,
ranked by mean val PR-AUC). To bring results back, copy `experiments/results/top5_seeds/` only;
the checkpoints are gitignored.
