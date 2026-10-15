# data_main: data behind the 80/20 PGB Morgan model

This folder holds the exact dataset, split code and split log for the PGB Morgan model trained on
2026-10-06 with the 80/20 split. No project code reads from this folder.

| | |
|---|---|
| Model checkpoint | `checkpoints/pgb_fixed_vector_morgan_concat_asl/single_run_80_20split/best_model.pt` (best epoch 1, seed 42) |
| Original code | git commit `f534ef7`: `run_pgb_single.py` → `pgb_single_run()` in `src/single_run.py` |
| Decision threshold | 0.671, tuned on `val.csv` (best MCC with true-negative rate ≥ 0.97) |

## Files

| File | Rows | Incompatible | md5 | What it is |
|---|---|---|---|---|
| `start_dataset.csv` | 3544 | 344 | `9572ba67ca8be7af5fee7a3c9911ccfe` | Full labelled dataset that was split. Same content as at commit `f534ef7` (the md5 differs from git only by Windows line endings). |
| `train.csv` | 2770 | 276 (10.0%) | `cfa2e256c21a15abd42f1f852ef94357` | Training set the model was trained on |
| `val.csv` | 774 | 68 (8.8%) | `5ba1f79b6e5283a5350094c4b9b53edc` | Validation set (early stopping, LR schedule, threshold) |
| `test.csv` | 24 | 12 (50%) | `61aeb1f251ab0c5201310a3c8ed56103` | The 24 held-out pairs, as used in the run (`ground_truth` renamed to `Outcome1`) |
| `held_out_test_set.csv` | 24 | 12 | `acfd0c6ae9e19908f96dd7d1fdd3206a` | The original held-out file (`ground_truth` column) |
| `make_split.py` | | | | Standalone reproduction of the split (see below) |
| `split_log.txt` | | | | Log written by `make_split.py`. The original 2026-10-06 run did not save a log; this log comes from the reproduction, which matches the original files byte for byte. |
| `original_code_f534ef7/` | | | | The original source files from git, unchanged, for reference only (not imported anywhere) |
| `model_metrics/` | | | | Metrics the model saved: `val_metrics.json`, and `heldout_metrics.json` (saved by the run as `test_metrics.json`, because the held-out pairs were its test set) |

`train.csv`, `val.csv` and `test.csv` are byte-identical copies of the files the run wrote to
`metrics/pgb_fixed_vector_morgan_concat_asl/single_run_80_20split/`. All 9 PGB families trained on
2026-10-06 used this same split.

## How the split was made

1. Read `start_dataset.csv`.
2. Butina-cluster the 530 unique API SMILES: RDKit Morgan fingerprint (radius 2, 2048 bits), Tanimoto
   distance cutoff 0.15 (similarity ≥ 0.85). This gives 512 clusters, stored as the `cluster_id` column.
3. `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` grouped by `cluster_id` gives
   train (2770) and val (774). The split is not stratified on the label.
4. The test set is the 24 held-out pairs.

To reproduce it:

```bash
cd data_main
python make_split.py
```

The output goes to `reproduced/` and is compared with the files here. Checked on 2026-10-14 with
rdkit 2026.03.6, scikit-learn 1.5.1 and pandas 2.2.2: all three files are identical in rows, order
and bytes.

## Metrics of this model

Re-scored from the checkpoint on 2026-10-14; identical to `model_metrics/`.

| Metric | Val (774) | Held-out (24) |
|---|---|---|
| PR-AUC | 0.6859 | 0.9594 |
| ROC-AUC | 0.9239 | 0.9583 |
| F1 | 0.6552 | 0.8000 |
| MCC | 0.6393 | 0.7071 |
| Precision | 0.7917 | 1.0000 |
| Recall | 0.5588 | 0.6667 |
| Accuracy | 0.9483 | 0.8333 |
| Confusion matrix [[TN, FP], [FN, TP]] | [[696, 10], [30, 38]] | [[12, 0], [4, 8]] |
| Correct | 734/774 | 20/24 |

This model has no internal test set.

## Known issues to disclose

- **Held-out pairs in train.** Two held-out pairs are also in `train.csv` with the same label (1):
  - Hydrocortisone (CID 5754) + Magnesium stearate (CID 11177)
  - Hydrocortisone (CID 5754) + β-cyclodextrin (CID 444041)

  The SMILES differ only in stereo notation, which the model's features ignore. On the 22 held-out
  pairs the model never trained on, it scores 18/22.
- **Near-duplicate in val.** `val.csv` contains Tetracycline + tricalcium phosphate (CID 24456), the
  same chemistry as held-out pair #4 (Tetracycline + calcium phosphate, CID 21226083).
- **Not stratified.** The split is not stratified on the label: train is 10.0% incompatible, val 8.8%.
