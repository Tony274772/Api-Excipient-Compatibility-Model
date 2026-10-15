# FINAL PROJECT: Morgan-PGB on the 80/20 split (20/24 held-out)

This folder holds the API–excipient compatibility model trained on the 80/20 split, and everything
it needs: code, data, held-out set, the original trained checkpoint and its metrics. It does not read
anything outside this folder, so you can copy it anywhere and it still trains, evaluates and predicts.

| | |
|---|---|
| Model | Prior-Gated Bilinear (PGB) head on Morgan-1024 + MACCS fingerprints, chemistry flags, PGB descriptors, salt context, excipient prior, mechanism flags |
| Split | train 2770 / val 774 rows (80 / 20 by Butina API cluster, no cluster in both) + the 24 held-out pairs as external test |
| Seed | 42 (best epoch 1, early stopping on val PR-AUC) |
| Threshold | 0.671 (best val MCC with true-negative rate ≥ 0.97) |
| Held-out (24 pairs) | **20/24 correct**: TP 8, TN 12, FP 0, FN 4, PR-AUC 0.959, ROC-AUC 0.958, MCC 0.707 |
| Held-out, 22 pairs not in train | 18/22 correct, PR-AUC 0.942, MCC 0.671 |
| Val (774) | PR-AUC 0.686, ROC-AUC 0.924, MCC 0.639, F1 0.655, 734/774 correct |

In the original project this was `run_pgb_single.py --family morgan` (commit `f534ef7`, trained
2026-10-06); its checkpoint was `checkpoints/pgb_fixed_vector_morgan_concat_asl/single_run_80_20split/`.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.12.7, torch 2.5.1, rdkit 2026.03.6, scikit-learn 1.5.1, pandas 2.2.2 and
numpy 1.26.4. Training and evaluation run on CPU; a GPU is optional and faster.

## Train

From inside the folder:

```bash
python run.py train
```

Or from any other directory, by giving the path to `run.py`:

```bash
python path/to/final_project/run.py train
```

This trains from scratch with seed 42 and the original recipe. It:

1. prints the loss and val PR-AUC for each epoch (max 40, early stopping with patience 7),
2. saves the checkpoint to `checkpoints/morgan_pgb_8020_retrain_seed42/`
   (`best_model.pt`, `prior_table.json`, `pgb_norm_stats.npz`, `threshold.json`),
3. evaluates the new model and prints the val and held-out metrics plus the 24 held-out predictions,
4. writes metrics, predictions, `training_history.json` and `run_info.json` to
   `metrics/morgan_pgb_8020_retrain_seed42/`.

The shipped model in `checkpoints/morgan_pgb_8020_seed42/` is never overwritten.

Options:

```bash
python run.py train --seed 43
```
```bash
python run.py train --device cpu
```
```bash
python run.py train --overwrite
```

`--device cuda` uses the GPU, `--epochs N` changes the epoch cap, and `--overwrite` retrains into an
existing `retrain_seed<N>` folder.

The original model was trained on a GPU. A retrain with seed 42 uses the identical config (checked
against the config stored in the original checkpoint), but GPU and CPU arithmetic differ slightly, so
the held-out count can move by a pair or so. Other seeds give different models.

## Other commands

```bash
python run.py evaluate
```
Scores the shipped original model and checks it against `reference_results/` (prints 20/24).

```bash
python run.py evaluate --checkpoint checkpoints/morgan_pgb_8020_retrain_seed42
```
Scores a retrained checkpoint.

```bash
python run.py predict --csv my_pairs.csv
```
Scores new pairs. The CSV needs `API_Smiles` and `Excipient_Smiles` columns; `Api_name`,
`Excipient_name` and a label column (`Outcome1` or `ground_truth`) are optional. Add
`--checkpoint ...` to use a retrained model.

```bash
python run.py split
```
Rebuilds train / val from `data/start_dataset.csv`, writes `metrics/rebuilt_split/split_log.txt` and
confirms the result is identical to `data/train.csv` and `data/val.csv`.

## Contents

```
run.py                                  entry point: train / evaluate / predict / split
src/                                    model, dataset, features, prior table, training loop, metrics
                                        (same code as the project's src/; encoders trimmed to the one used)
checkpoints/morgan_pgb_8020_seed42/     THE ORIGINAL TRAINED MODEL (20/24)
    best_model.pt                       weights + training config
    prior_table.json                    excipient prior table (fit on train only)
    pgb_norm_stats.npz                  PGB descriptor mean / std (train only)
    threshold.json                      decision threshold 0.671 (from val)
data/
    start_dataset.csv                   full dataset (3544 pairs) the split is made from
    train.csv  val.csv                  the 80/20 split the model was trained / tuned on
    held_out_test_set.csv               24 external pairs (label column ground_truth, 1 = incompatible)
    split_log.txt                       log of the split reproduction
reference_results/                      original metrics, held-out and val predictions, run info
metrics/                                created by run.py (metrics JSON, prediction CSVs, training history)
```

## How the split was made

1. Butina-cluster the 530 unique API SMILES on Morgan fingerprints (radius 2, 2048 bits) with
   Tanimoto distance cutoff 0.15 (similarity ≥ 0.85). This gives 512 clusters, stored as `cluster_id`.
2. `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` grouped by `cluster_id` gives
   train (2770 rows, 10.0 % incompatible) and val (774 rows, 8.8 % incompatible). The split is not
   stratified on the label.
3. The test set is the 24 held-out pairs (12 incompatible, 12 compatible).

No API cluster or API CID is in both train and val. Excipients are shared by design.

## Training recipe

- The PGB descriptor norm stats and the excipient prior table are fit on train only. Train rows use
  leave-own-cluster-out priors.
- Loss: asymmetric focal (γ+ 1.2, γ− 2.5) with a positive-class weight and no balanced sampler.
- Optimiser: AdamW, LR 1.2e-3, weight decay 2e-3, batch size 64, gradient clip 1.0.
- Early stopping and the LR schedule (ReduceLROnPlateau, factor 0.5, patience 4) use val PR-AUC:
  max 40 epochs, patience 7.
- The threshold is chosen on val. The held-out pairs are only scored.

## Known issues to disclose

- **Held-out pairs in train.** Two held-out pairs are also in `train.csv` with the same label (1):
  Hydrocortisone + Magnesium stearate and Hydrocortisone + β-cyclodextrin. On the 22 held-out pairs
  the model never trained on, it scores 18/22 (`evaluate` prints this as `heldout_unseen`).
- **Near-duplicate in val.** `val.csv` contains Tetracycline + tricalcium phosphate, the same
  chemistry as held-out pair #4 (Tetracycline + calcium phosphate).
- **No internal test set.** Val is used for early stopping and the threshold, so the val metrics are
  optimistic; the held-out set is the only untouched test.
