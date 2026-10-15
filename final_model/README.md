# FINAL MODEL: Morgan-PGB on the 60/20/20 split (19/24 held-out)

This folder holds the final API–excipient compatibility model and everything it needs: code, data
splits, held-out set, trained checkpoint and the original results. It does not read anything outside
this folder, so you can move or copy it anywhere and it still runs.

| | |
|---|---|
| Model | Prior-Gated Bilinear (PGB) head on Morgan-1024 + MACCS fingerprints, chemistry flags, PGB descriptors, salt context, excipient prior, mechanism flags |
| Split | train 2075 / val 774 / test 695 rows (58.5 / 21.8 / 19.6 %), Butina API clusters, no cluster in two splits |
| Seed | 42 (best epoch 2 of 9, early stopping on val PR-AUC) |
| Threshold | 0.690 (best val MCC with true-negative rate ≥ 0.97) |
| Held-out (24 pairs) | **19/24 correct**: TP 8, TN 11, FP 1, FN 4, PR-AUC 0.914, ROC-AUC 0.903, MCC 0.602 |
| Test (695) | PR-AUC 0.584, ROC-AUC 0.892, MCC 0.450, F1 0.496 |
| Val (774) | PR-AUC 0.664, ROC-AUC 0.915, MCC 0.609, F1 0.626 |

In the original project this was run `oct6_resplit / morgan / seed_42` (made by
`experiments/oct6_resplit/run.py`, commit `f5ee9f1`).

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.12.7, torch 2.5.1 (CUDA and CPU), rdkit 2026.03.6, scikit-learn 1.5.1,
pandas 2.2.2 and numpy 1.26.4. The model runs on CPU (about 30 s for a full evaluation); a GPU is
optional.

## Run

You can run these from any directory:

```bash
python run.py
```
Evaluates the shipped model. Prints val, test and held-out metrics and the 24 held-out predictions,
then checks them against `reference_results/`. Writes everything to `outputs/morgan_pgb_seed42/`.

```bash
python run.py predict --csv my_pairs.csv
```
Scores new pairs. The CSV needs `API_Smiles` and `Excipient_Smiles` columns; `Api_name`,
`Excipient_name` and a label column (`Outcome1` or `ground_truth`) are optional. If a label column is
present, metrics are printed too.

```bash
python run.py train --seed 42
```
Retrains from scratch with the same recipe and writes the result to `checkpoints/retrain_seed_42/`,
then evaluates it. It never overwrites the shipped model. On the original GPU (RTX 4050) seed 42
reproduces the shipped model exactly: same best epoch, threshold and 19/24. On another GPU or on CPU
the result can differ slightly. Other seeds give different models.

```bash
python run.py evaluate --checkpoint checkpoints/retrain_seed_42
```
Evaluates a retrained checkpoint.

```bash
python run.py split
```
Rebuilds train / val / test from `data/start_dataset.csv`, checks for leakage, writes
`outputs/rebuilt_split/split_log.txt` and confirms that the result is byte-identical to
`data/train.csv`, `val.csv` and `test.csv`.

Use `--device cpu` or `--device cuda` with `evaluate`, `predict` or `train` to choose the device.

## Contents

```
run.py                              entry point: evaluate / predict / train / split
src/                                model, dataset, features, prior table, training loop, metrics
                                    (copied unchanged from the project's src/; encoders trimmed to the one used)
checkpoints/morgan_pgb_seed42/      THE FINAL MODEL
    best_model.pt                   weights + training config
    prior_table.json                excipient prior table (fit on train only)
    pgb_norm_stats.npz              PGB descriptor mean / std (train only)
    threshold.json                  decision threshold 0.690 (from val)
data/
    start_dataset.csv               full dataset (3544 pairs) that the split is made from
    train.csv  val.csv  test.csv    the split the model was trained / tuned / tested on
    split_log.txt                   step-by-step log of the split (clustering, leakage checks, summary)
    held_out_test_set.csv           24 external pairs (label column: ground_truth, 1 = incompatible)
reference_results/                  original metrics, predictions, training history and run info
outputs/                            created by run.py (metrics JSON + prediction CSVs)
```

## How the split was made

The split is made from `data/start_dataset.csv` (3544 pairs, 9.7 % incompatible). The full log is in
[`data/split_log.txt`](data/split_log.txt).

1. **Cluster near-duplicate APIs:** Butina clustering of the 530 unique API SMILES on Morgan
   fingerprints (radius 2, 2048 bits) at Tanimoto ≥ 0.85 gives 512 clusters. 18 clusters hold more
   than one API. `cluster_id` is the grouping key.
2. **val:** `GroupShuffleSplit(test_size=0.2, random_state=42)` takes 20 % of the API clusters
   (103 clusters, 774 rows).
3. **test / train:** the remaining clusters go into `StratifiedGroupKFold(n_splits=4, shuffle=True,
   random_state=42)`, stratified on the label. Fold 1 is **test** (105 clusters, 695 rows) and
   folds 2–4 are **train** (304 clusters, 2075 rows).

| split | rows | share | incompatible | API clusters | APIs | excipients |
|---|---|---|---|---|---|---|
| train | 2075 | 58.5 % | 193 (9.3 %) | 304 | 318 | 259 |
| val | 774 | 21.8 % | 68 (8.8 %) | 103 | 108 | 158 |
| test | 695 | 19.6 % | 83 (11.9 %) | 105 | 107 | 164 |

- Every row of start_dataset is used exactly once. No API cluster, API CID or API SMILES appears
  in two splits. Excipients are shared across splits by design.
- Three held-out pairs also occur in start_dataset and were kept in their split:
  Hydrocortisone + Mg stearate and Hydrocortisone + β-cyclodextrin (train), and
  Tetracycline + calcium phosphate (val). These three pairs are not a fully independent test.

## Training recipe

- The PGB descriptor norm stats and the excipient prior table are fit on train only. Train rows use
  leave-own-cluster-out priors.
- Loss: asymmetric focal (γ+ 1.2, γ− 2.5) with a positive-class weight and no balanced sampler.
- Optimiser: AdamW, LR 1.2e-3, weight decay 2e-3, batch size 64.
- Early stopping and the LR schedule use val PR-AUC: max 40 epochs, patience 7, LR patience 4.
- The threshold is chosen on val. Test and held-out are only scored.
