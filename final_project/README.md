# FINAL PROJECT: Morgan-PGB on the 80/20 split (18/22 held-out)

This folder holds the API–excipient compatibility model trained on the 80/20 split, and everything
it needs: code, data, held-out set, the original trained checkpoint and its metrics. It does not read
anything outside this folder, so you can copy it anywhere and it still trains, evaluates and predicts.

| | |
|---|---|
| Model | Prior-Gated Bilinear (PGB) head on Morgan-1024 + MACCS fingerprints, chemistry flags, PGB descriptors, salt context, excipient prior, mechanism flags |
| Split | train 2770 / val 774 rows (80 / 20 by Butina API cluster, no cluster in both) + the 22 held-out pairs as external test |
| Seed | 42 (best epoch 1, early stopping on val PR-AUC) |
| Threshold | 0.671 (best val MCC with true-negative rate ≥ 0.97) |
| Held-out (22 pairs) | **18/22 correct**: TP 6, TN 12, FP 0, FN 4, PR-AUC 0.942, ROC-AUC 0.950, MCC 0.671 |
| Original held-out (24 pairs, 2 also in train) | 20/24 correct, PR-AUC 0.959, ROC-AUC 0.958, MCC 0.707 |
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
3. evaluates the new model and prints the val and held-out metrics plus the 22 held-out predictions,
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

### Training on another split

```bash
python run.py train --data-dir data/random
```

`--data-dir` takes any folder with `train.csv`, `val.csv` and `held_out_test_set.csv`. The default is
`data` (the API group split above). The other splits (see "Split variants" below) are `data/random`,
`data/excipient_new` and `data/api_excipient_new`. The recipe is the same. The checkpoint goes to
`checkpoints/morgan_pgb_<folder>_seed<N>/` (e.g. `morgan_pgb_random_seed42`) and stores the data folder in
`data_dir.txt`, so `python run.py evaluate --checkpoint checkpoints/morgan_pgb_random_seed42` scores
that model on its own val set. Pass `--data-dir` to `evaluate` to score on a different folder.

## Other commands

```bash
python run.py evaluate
```
Scores the shipped original model and checks it against `reference_results/` (prints 18/22).

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
checkpoints/morgan_pgb_8020_seed42/     THE ORIGINAL TRAINED MODEL (18/22)
    best_model.pt                       weights + training config
    prior_table.json                    excipient prior table (fit on train only)
    pgb_norm_stats.npz                  PGB descriptor mean / std (train only)
    threshold.json                      decision threshold 0.671 (from val)
data/
    start_dataset.csv                   full dataset (3544 pairs) the split is made from
    train.csv  val.csv                  the 80/20 split the model was trained / tuned on
    held_out_test_set.csv               22 external pairs (label column ground_truth, 1 = incompatible)
    split_log.txt                       log of the split reproduction
    make_split_variants.py              builds the three split variants below
    random/  excipient_new/  api_excipient_new/
                                        split variants: train.csv, val.csv, held_out_test_set.csv, split_log.txt
reference_results/                      original metrics, held-out and val predictions, run info
                                        (*_24pairs = the original 24-pair held-out results)
metrics/                                created by run.py (metrics JSON, prediction CSVs, training history)
```

## How the split was made

1. Butina-cluster the 530 unique API SMILES on Morgan fingerprints (radius 2, 2048 bits) with
   Tanimoto distance cutoff 0.15 (similarity ≥ 0.85). This gives 512 clusters, stored as `cluster_id`.
2. `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` grouped by `cluster_id` gives
   train (2770 rows, 10.0 % incompatible) and val (774 rows, 8.8 % incompatible). The split is not
   stratified on the label.
3. The test set is the 22 held-out pairs (10 incompatible, 12 compatible).

No API cluster or API CID is in both train and val. Excipients are shared by design.

### Split variants

`python data/make_split_variants.py` builds three more train / val splits from `start_dataset.csv` to
compare split strategies. All use seed 42 and the same 22 held-out pairs.

Excipient groups (`exc_group_id`, 234 groups) are what the excipient splits keep whole:

1. Butina on excipient SMILES with the API recipe: 284 clusters (`exc_cluster_id`).
2. Clusters merged when two excipients are forms of the same parent compound: same fragments once water,
   solvents and charges are removed (CaCl2 = CaCl2 dihydrate), or the same organic part with counter-ions
   dropped (sodium saccharin = saccharin, aluminium lake = dye), or one is a mixture containing the other.
3. Four PubChem duplicates no rule can see (`EXCIPIENT_SYNONYMS`): polysorbate 80, hypromellose (HPMC),
   anhydrous lactose, and ferric oxide yellow / hydrate.

The excipient split is stratified: `StratifiedGroupKFold` cannot balance it (its folds range 6–13 %
incompatible) because a few groups hold up to 11 % of rows. Instead the script keeps the best of 1000
random group shuffles: closest to 20 % of rows and to the overall 9.7 % rate, also scoring the rate of
the API + excipient val. Only row counts and labels are used.

| Folder | Split | Train (incompatible) | Val (incompatible) | Unseen in val |
|---|---|---|---|---|
| `data/` | API group (above) | 2770 (276, 10.0 %) | 774 (68, 8.8 %) | APIs |
| `data/random/` | random 80/20, stratified on the label | 2835 (275, 9.7 %) | 709 (69, 9.7 %) | nothing |
| `data/excipient_new/` | stratified group split on `exc_group_id` | 2828 (274, 9.7 %) | 716 (70, 9.8 %) | excipients |
| `data/api_excipient_new/` | val API clusters of `data/` × val excipient groups of `excipient_new/` | 2201 (220, 10.0 %) | 147 (14, 9.5 %) | APIs and excipients |

In `api_excipient_new`, the 1196 pairs that have one side in train groups and the other in val groups are
dropped (`dropped_mixed.csv`). Each folder's `split_log.txt` has the overlap checks and the merge list.

## Training recipe

- The PGB descriptor norm stats and the excipient prior table are fit on train only. Train rows use
  leave-own-cluster-out priors.
- Loss: asymmetric focal (γ+ 1.2, γ− 2.5) with a positive-class weight and no balanced sampler.
- Optimiser: AdamW, LR 1.2e-3, weight decay 2e-3, batch size 64, gradient clip 1.0.
- Early stopping and the LR schedule (ReduceLROnPlateau, factor 0.5, patience 4) use val PR-AUC:
  max 40 epochs, patience 7.
- The threshold is chosen on val. The held-out pairs are only scored.

## Known issues to disclose

- **Held-out pairs that were in train (removed).** The original held-out set had 24 pairs; two of
  them are also in `train.csv` (in every split, since they come from `start_dataset.csv`) with the same
  label (1): Hydrocortisone + Magnesium stearate and Hydrocortisone + β-cyclodextrin. They were removed
  from `held_out_test_set.csv` on 2026-11-04. The model predicted both correctly, so it scores 20/24 on
  the original set and 18/22 now.
- **Near-duplicate in val.** `val.csv` contains Tetracycline + tricalcium phosphate, the same
  chemistry as held-out pair #4 (Tetracycline + calcium phosphate).
- **No internal test set.** Val is used for early stopping and the threshold, so the val metrics are
  optimistic; the held-out set is the only untouched test.
