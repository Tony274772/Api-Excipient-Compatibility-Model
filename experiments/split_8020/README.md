# 80/20 split runs

Trains the 54 baseline models in `metrics/` and the 9 PGB families on an 80/20 train/val split,
then evaluates every model on the 24 held-out pairs.

## 1. Build the split (once)

```bash
python data/data_split.py --mode 8020
```

`data/data_split.py` is the split strategy that produced `data/train.csv`, `val.csv` and `test.csv`:
APIs are grouped by Butina clusters (Morgan-2048, Tanimoto ≥ 0.85) and split with
`StratifiedGroupKFold`. `--mode 602020` reproduces those three files byte-for-byte, and it refuses to
overwrite them if anything differs (`--force` overrides).

`--mode 8020` uses the same first fold:

- **train** = the 60/20/20 train + val folds
- **val** = the 60/20/20 test fold

Rows that match a held-out pair are removed. A match means the same CID pair, or the same API and
excipient after salt and stereo stripping. These rows are:

- Hydrocortisone + Mg stearate and Hydrocortisone + β-cyclodextrin (from val)
- Tetracycline + tricalcium phosphate (from train)

Result: train 2871 rows, val 670 rows.

Outputs in `data/split_8020/`:

- `train.csv`, `val.csv`
- `descriptor_norm_stats.json` (from the 80/20 train rows only)
- `removed_heldout_overlap.csv`
- `splitlog_8020.log`

## 2. Train

```bash
python -m experiments.split_8020.run --list
python -m experiments.split_8020.run --models fixed_vector_morgan_concat_asl
python -m experiments.split_8020.run --models baseline --skip_duplicates
python -m experiments.split_8020.run --models pgb
python -m experiments.split_8020.run --models all --skip_duplicates --seeds 42 43 44
```

- **Baseline models** use the same recipe as `main.py`: the settings each model's checkpoint actually
  used (see `models.py`), with D-MPNN getting its separate encoder learning rate. The val threshold is
  chosen by best F1, as before.
- **PGB families** use the same recipe as `experiments/pgb`: leave-own-cluster-out excipient prior and
  a val threshold with true-negative rate ≥ 0.97.
- **Val** drives early stopping, the LR schedule and the threshold. The held-out pairs are never used
  for any choice.
- **Fixed-vector models** check that every train, val and held-out molecule has a vector, and stop with
  an error otherwise. Held-out molecules get the vector of their CID from the training files. Missing
  CIDs are computed with `data/compute_fixed_vectors.py` (Morgan, MACCS, Mol2vec) or read from
  `data/pubchem_fps_heldout.csv` (PubChem).
- **Finished runs are skipped**, so an interrupted sweep can be restarted with the same command.
  `--overwrite` retrains.
- **Failures are logged and skipped:** a failed model is written to `metrics/split_8020/failures.log`
  and the sweep continues.

## 3. Outputs

```
checkpoints/split_8020/<model>/seed_<s>/best_model.pt     (+ prior_table.json, pgb_norm_stats.npz for PGB)
metrics/split_8020/<model>/seed_<s>/
    val_metrics.json          PR-AUC, ROC-AUC, F1, MCC, precision, recall, threshold, confusion matrix
    heldout_metrics.json      same metrics + correct / tp / tn on the 24 pairs
    heldout_predictions.csv   per-pair probability, prediction, correct
    training_history.json
    run_info.json             config, split file md5s, git commit, epochs, threshold, held-out vector report
metrics/split_8020/summary.csv            all runs, sorted by held-out correct then val PR-AUC
metrics/split_8020/summary_by_model.csv   mean ± std over seeds (when more than one seed)
```

To rebuild the summary at any time:

```bash
python -m experiments.split_8020.run --summary
```
