# Reagent ablation

Does the best grid model, `pgb_dmpnn_chemprop_cross_attn_cls_asl`, depend on the laboratory reagents
in the training data? 14 "excipients" (KMnO4, ozone, H2SO4, NaOH, KOH, H2O2, formaldehyde, KNO3, Al
and Zn metal, glyceraldehyde, free phosphate / sulfate / carbonate ions) are never used in real
formulations, and most of them are labelled incompatible with every API they appear with. They
make up 2% of the rows but 15% of the incompatible rows in train and 19% in val. None of the 22
held-out pairs contains one. The full list is `REAGENTS` in `reagents.py`.

| Arm | Data | Train | Val | Held-out |
|---|---|---|---|---|
| `control` | `data/grid_8020/` | 2770 (276 incompatible) | 774 (68) | 22 |
| `no_reagents` | `data/grid_8020_no_reagents/` | 2714 (234) | 758 (55) | 22, the same pairs |

The only difference between the arms is the data. Training goes through
`experiments/full_grid/run_grid.py:run_one`, the same recipe as `top5_seeds_depth6`, with seeds 42-46.
The PGB excipient prior and the normalisation stats are refit from each arm's own train split.
The control arm repeats the top-5 runs because those runs saved no per-row predictions. With
predictions from both arms, both can be scored on the same val rows.

## Commands (DGX, from `project/`)

```bash
bash experiments/reagent_ablation/run_dgx.sh
```

This builds the data, puts control on GPU 0 and no_reagents on GPU 1, and collects the results at the
end. With a single GPU, run `NGPU=1 bash ...`. Step by step instead:

```bash
python experiments/reagent_ablation/reagents.py
python experiments/reagent_ablation/run.py --device cuda
python experiments/reagent_ablation/collect.py
```

`run.py --list` shows progress. A finished (arm, seed) run is skipped, so after an interruption run
the same command again. `--arms`, `--seeds` and `--models` narrow the run, and `--overwrite` retrains.

## Reading the results

`collect.py` writes `experiments/results/reagent_ablation/{per_seed,summary,delta}.csv`:

- `heldout_*`: the 22 external pairs, identical in both arms. **This is the main comparison.**
- `valclean_*`: val without reagent rows (the same 758 rows in both arms). This is the fair val comparison.
- `val_*`: each arm's own val set, as a normal run reports it. Do not compare these across arms.
- `reagent_recall` (control only): the share of incompatible reagent rows in val that the model flags.
- `delta.csv`: per-seed paired difference, no_reagents minus control.

Bring back `experiments/results/reagent_ablation/` (checkpoints stay on the DGX).

## Results (seeds 42-46, DGX)

The control arm reproduced `top5_seeds_depth6` exactly: val PR-AUC 0.7269 vs 0.7273, held-out
18.2/22 in both. So the only difference between the arms is the data.

| Mean ± std over 5 seeds | control | no_reagents | Paired Δ (no_reagents − control) | Paired t-test p |
|---|---|---|---|---|
| Held-out PR-AUC (22 pairs) | 0.917 ± 0.020 | 0.888 ± 0.031 | −0.029 ± 0.031 | 0.10 |
| Held-out ROC-AUC | 0.915 ± 0.016 | 0.887 ± 0.022 | −0.028 ± 0.021 | 0.04 |
| Held-out MCC | 0.680 ± 0.068 | 0.552 ± 0.145 | −0.128 ± 0.147 | 0.12 |
| Held-out correct | 18.2 ± 0.8 | 16.8 ± 1.5 | −1.4 ± 1.5 | 0.11 |
| Val PR-AUC, 758 reagent-free rows | 0.639 ± 0.028 | 0.619 ± 0.024 | −0.020 ± 0.036 | 0.29 |
| Val ROC-AUC, 758 reagent-free rows | 0.928 ± 0.006 | 0.923 ± 0.008 | −0.005 ± 0.010 | 0.29 |
| Val MCC, 758 reagent-free rows | 0.613 ± 0.024 | 0.594 ± 0.039 | −0.019 ± 0.024 | 0.15 |
| Val PR-AUC, own val set | 0.727 ± 0.014 (774 rows) | 0.619 ± 0.024 (758 rows) | not comparable | – |
| Recall on incompatible reagent rows in val | 0.85 ± 0.05 | – | – | – |

Without reagents, 4 of 5 seeds were worse on every held-out metric. The p-values come from a
paired t-test across the 5 seeds (scipy `ttest_1samp` on `delta.csv`).

### Findings

1. **Removing the reagents does not improve the model.** It is slightly worse: 1.4 fewer correct
   held-out pairs and 0.028 lower held-out ROC-AUC. On the same reagent-free val rows, every metric
   drops a little, and none of those drops is significant. The evidence is limited (5 seeds and 22
   held-out pairs, where one swap of two pairs in the ranking moves ROC-AUC by 0.008), so read this
   as "no benefit, possibly a small loss". Removing the rows also removes 42 of the 276 incompatible
   training examples (15%), and some of them (strong bases, oxidisers) carry real reactivity
   chemistry.

2. **The model does not use the reagents as a shortcut for the held-out inorganic salts.** The mean
   probability over the 5 seeds hardly changes for the salt pairs: Trichlormethiazide + MgO
   0.941 → 0.938, Phenytoin sodium + CaSO4 0.865 → 0.861, Quercetin + MgCO3 0.908 → 0.902,
   Warfarin + Mg-Al silicate 0.820 → 0.842. The held-out pairs that changed mostly flipped because
   the decision threshold moved. With only 55 incompatible pairs left in val, the threshold tuned
   on val (best MCC with TNR ≥ 0.97) ranges from 0.73 to 0.87 across seeds, against 0.74 to 0.83
   in control. The largest single loss was Tetracycline + calcium phosphate: 0.83 → 0.75, and
   correct in 2/5 seeds instead of 5/5.

3. **The reagent rows inflate the val PR-AUC.** The same control models score 0.727 on the full
   val set but 0.639 on its 758 reagent-free rows. They flag 85% of the incompatible reagent rows,
   so those 16 rows are easy points worth about 0.09 PR-AUC. ROC-AUC is barely affected
   (0.928 on reagent-free rows). No held-out pair contains a reagent, so the held-out numbers are
   unaffected.

### Decision

- **Keep the reagents in training.** Removing them does not help, and the held-out results are
  slightly worse.
- **Report val on the reagent-free rows as well.** For this model: val PR-AUC 0.639 ± 0.028 on
  reagent-free rows, against 0.727 ± 0.014 on the full val set. In the paper, say that 16 of the
  774 val rows are laboratory reagents labelled almost always incompatible and that they inflate
  the full-val PR-AUC. Compare models on the reagent-free rows.
- Held-out results (18.2 ± 0.8 / 22, PR-AUC 0.917 ± 0.020) stand as reported.

Files: `experiments/results/reagent_ablation/per_seed.csv`, `summary.csv`, and `delta.csv` (per-seed
paired differences). Per-row predictions for every run are under
`<arm>/seed_<s>/<run>/predictions_{val,heldout}.csv`.
