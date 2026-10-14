# Excipient-Prior Ablation (PGB, Morgan family)

Tests how much the PGB model gains from its 5-dimensional, 4-level excipient prior.
Nothing outside the prior is changed: data split, towers, bilinear rank, losses, learning
rates, epochs, early stopping and threshold tuning are those of `experiments/pgb/single_run.py`.
No file in `src/` or `experiments/pgb/` is modified.

## Variants

| Variant | Prior cascade | Prior vector | Gate input | Pair representation | Prior-trust input |
|---|---|---|---|---|---|
| `baseline_4level_5d` | exact → family → NN → global | `[p_exact, p_delta, p_family, log_n, unseen]` | 5+18+18+5 = **46** | 4·96+16+5+5 = **410** | 5 |
| `ablation_3level_5d` | exact → NN → global | same 5 slots (`p_family` = `p_exact`) | **46** | **410** | 5 |
| `ablation_3level_1d` | exact → NN → global | `[p_exact]` | 1+18+18+5 = **42** | 4·96+16+1+5 = **406** | 1 |

- **exact**: Laplace-smoothed rate `(pos + 4·0.094) / (n + 4)` for the canonical, stereo-blind excipient.
- **NN**: rate of the single most Tanimoto-similar training excipient (RDKit Morgan, radius 2, 512 bits) when similarity ≥ 0.6.
- **global**: 0.094.
- Training rows use leave-own-API-cluster-out priors (Butina clusters), exactly as in the baseline.

Comparisons:
- **A** = baseline vs `3level_5d`, which isolates removing the family level.
- **B** = `3level_5d` vs `3level_1d`, which isolates the 5-d → 1-d prior vector.
- **A+B** = baseline vs `3level_1d`, both changes together.

### What stays identical in the head

All prior-dependent widths come from `config.pgb_prior_vec_dim`, so all variants run the same `PGBHead` code.

- Gate: `Linear(in → 32) → GELU → Linear(32 → 16) → Sigmoid`, only `in` changes.
- Residual MLP: `Linear(in → 128) → GELU → Dropout 0.3 → Linear(128 → 48) → GELU → Dropout 0.2 → Linear(48 → 1)`, only `in` changes.
- Adaptive prior-trust branch is kept: `Linear(in → 8) → GELU → Linear(8 → 1) → Sigmoid`, only `in` changes.
  In the 1-d model `w` can depend only on `p_exact`. It can no longer use `log_n` or `unseen`
  to judge how reliable the prior is. Losing that is part of what change B measures.
- Final logit: `residual + w · logit(p_exact) + b`, with the same `prior_bias` initialisation.

The 1-d model has 672 fewer parameters than the 5-d models.

### How often the family level matters (train table fitted on `data/train.csv`)

| Split | Rows | Family level fires (4-level) | 3-level: those rows go to |
|---|---|---|---|
| train (leave-cluster-out) | 2030 | 24 | NN +21, global +3 |
| val | 842 | 13 | NN +10, global +3 |
| test | 672 | 8 | NN +5, global +3 |
| held-out | 24 | 1 | NN +1 |

Each run also writes these counts to `prior_level_counts.json`.

## Running (from the project root)

```bash
# all 3 variants x 5 seeds (finished seeds are skipped, so the sweep can be resumed)
python -m experiments.ablation_studies.run_ablation --variant all --seeds 42 43 44 45 46

# summary (mean ± std) + paired bootstrap
python -m experiments.ablation_studies.compare
```

Options: `--variant {baseline_4level_5d, ablation_3level_5d, ablation_3level_1d, all}`,
`--family` (default `morgan`), `--overwrite`, `--epochs` (smoke tests only),
`--results-root`, `--checkpoint-root`.

## Outputs

```
experiments/results/ablation_studies/<variant>/morgan/seed_<s>/
    val_metrics.json  test_metrics.json  heldout_metrics.json  training_history.json
    predictions_{val,test,heldout}.csv   # per-row prob, label, prior p_exact, prior level
    prior_level_counts.json  run_info.json
experiments/results/ablation_studies/comparison_morgan/
    summary.csv  summary.md  bootstrap.json  bootstrap.md
checkpoints/ablation_studies/<variant>/morgan/seed_<s>/   # best_model.pt, prior_table.json, norm stats
```

## Statistics

- **Seeds:** 42, 43, 44, 45, 46, the same for every variant. Tables report mean ± sample std.
- **Paired bootstrap** (`compare.py`): resample the evaluation rows with replacement, stratified so
  each resample has the same number of positives (77 on test). On every resample, score all models on the same rows and compute
  `mean_seeds(metric X) − mean_seeds(metric Y)`. Report the 2.5–97.5 percentile interval and a
  two-sided bootstrap p-value. 10,000 resamples by default, for PR-AUC and ROC-AUC on test and val.
  The held-out set (24 rows) is only summarised. It is too small for a bootstrap.

## Reproduction check

`baseline_4level_5d` with seed 42 reproduces the existing result in
`experiments/results/metrics_pgb/group_split/morgan/` exactly: val PR-AUC 0.6925, test PR-AUC 0.7253, test MCC 0.5836.
So the ablation runner follows the original pipeline faithfully.
