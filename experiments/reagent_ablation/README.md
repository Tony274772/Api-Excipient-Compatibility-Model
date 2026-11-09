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
