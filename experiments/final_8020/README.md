# All models on the final_project 80/20 split

Trains every model of this repo on the split in `final_project/data/` (the split Morgan-PGB 20/24 was
trained on) and scores each one on the 24 held-out pairs.

| | |
|---|---|
| Train | `final_project/data/train.csv`: 2770 rows |
| Val | `final_project/data/val.csv`: 774 rows (early stopping, LR schedule, threshold) |
| Test | `final_project/data/held_out_test_set.csv`: 24 pairs (only scored, never used for any choice) |

## Models (74, all different)

| Group | n | Recipe | Threshold |
|---|---|---|---|
| `baseline` | 54 | `main.py` CompatibilityModel, with the settings each `metrics/<name>` checkpoint used | best val F1 |
| `pgb` | 9 | `experiments/pgb` (Prior-Gated Bilinear), names as `metrics/pgb_*` | best val MCC with TNR ≥ 0.97 |
| `pgb_global_gated` | 2 | `experiments/pgb/global_gated`: PGB ChemBERTa and D-MPNN with global gated attention pooling | best val MCC with TNR ≥ 0.97 |
| `gb` | 9 | `experiments/gated_bilinear` (Gated Bilinear) | best val MCC with TNR ≥ 0.97 |

**Concat cls vs concat global-gated.** `src/model.py` ignores `pooling` under concat fusion: it always
uses the encoder's readout. So in `metrics/`, each `*_concat_global_gated_*` run trained the same network as
its `*_concat_cls_*` twin. Here they are different models:

| Model | Class | Structural vector |
|---|---|---|
| `dmpnn_chemprop_concat_cls_{asl,bce}`, `pretrained_gat_concat_cls_{asl,bce}` | `CompatibilityModel` | proj(encoder readout = mean over atoms) |
| `dmpnn_chemprop_concat_global_gated_{asl,bce}`, `pretrained_gat_concat_global_gated_{asl,bce}` | `ConcatGatedPoolingModel` (`concat_pooling.py`) | fuse[proj(readout), gated attention pooling over proj(atom tokens)] |

Neither has cross-attention; both encode the API and the excipient separately. `python -m
experiments.final_8020.run --list` shows each model's class, fusion and pooling. The runner stops if two
baseline models would build the same network.

`pgb_fixed_vector_morgan_concat_asl` is the final_project model, so retraining it here doubles as a
reproduction check (20/24).

## Commands

Run all of these from the project root (`project/`).

```bash
python -m experiments.final_8020.run --list
```

Everything (74 runs):
```bash
python -m experiments.final_8020.run --models all
```

One group at a time:
```bash
python -m experiments.final_8020.run --models baseline
python -m experiments.final_8020.run --models pgb
python -m experiments.final_8020.run --models pgb_global_gated
python -m experiments.final_8020.run --models gb
```

One or more models by name:
```bash
python -m experiments.final_8020.run --models gb_morgan pretrained_gat_concat_global_gated_asl
```

Options:

- `--seeds 42 43 44`: several seeds; the summary then adds a mean ± std file.
- `--device cuda` or `--device cpu`: the default is auto.
- `--overwrite`: retrain finished runs; their folders are emptied first. Without it, finished runs are
  skipped, so an interrupted sweep restarts with the same command.
- `--data_dir <folder>` and `--out <name>`: another copy of the split, for example with the leaked rows
  removed. The folder needs `train.csv`, `val.csv` and `held_out_test_set.csv`. Pick a new `--out` so
  the results stay apart.
- `--gb_original_prior`: Gated Bilinear with the original `train_gb.py` priors (see below).

Rebuild the summary table:
```bash
python -m experiments.final_8020.run --summary
```

## Outputs

Every model and seed gets its own two folders. Nothing is shared between runs except the summary files.

```
checkpoints/final_8020/<model>/seed_<s>/
    best_model.pt                     best val PR-AUC epoch (weights + config)
    descriptor_norm_stats.json        baseline: descriptor mean / std of the train rows
    prior_table.json, pgb_norm_stats.npz   PGB: excipient prior and descriptor stats of the train rows
metrics/final_8020/<model>/seed_<s>/
    heldout_predictions.csv           24 pairs: label, probability, prediction, correct, in_unseen, in_clean
    training_history.json             loss, val PR-AUC and LR per epoch
    val_metrics.json                  PR-AUC, ROC-AUC, F1, MCC, precision, recall, confusion matrix, threshold
    run_info.json                     architecture, config, split md5s, git commit, best epoch, checks passed
    heldout_metrics.json              same metrics on the 24 pairs + "subsets" (unseen, clean); written last
metrics/final_8020/summary.csv        all finished runs, sorted by held-out correct, then val PR-AUC
metrics/final_8020/summary_by_model.csv   mean ± std over seeds (with more than one seed)
metrics/final_8020/failures.log       traceback of any failed run (the sweep carries on)
```

- **Clean start:** a run that starts (a new one, an unfinished one, or `--overwrite`) first empties its
  own two folders, so no file from an earlier attempt can be read back. In particular, the training
  loops reload `best_model.pt` from disk, so a stale one would otherwise get scored.
- **Finished marker:** a run counts as finished only when `best_model.pt` and all five metrics files
  exist. `heldout_metrics.json` is written last.
- **Restore check:** after training, the val PR-AUC of the restored model has to match the value stored
  in `best_model.pt`. This proves the scored model is this run's best checkpoint.

Held-out subsets:
- **unseen (22):** the held-out pairs whose CID pair is not in `train.csv`. This drops Hydrocortisone +
  Mg stearate and Hydrocortisone + β-cyclodextrin.
- **clean (20):** also leaves out pairs in `val.csv` and the two near-duplicates in val:
  - Tetracycline + calcium phosphate (val has CIDs 24441 / 24456)
  - Levetiracetam + talc (val has talc CID 26924)

## Checks before training (`checks.py`)

The repo code has silent fallbacks:
- a CID missing from a descriptor table gets an all-zero descriptor vector;
- an unknown molecule gets an all-zero fixed vector;
- a GAT molecule that fails featurisation gets all-zero embeddings;
- an all-zero PubChem / Mol2vec vector is marked unavailable.

So every run stops with an error, before training, if any of these holds:

- the split has NaN, empty SMILES, labels other than 0/1, a duplicate pair, a pair in both train and val,
  or a SMILES that RDKit cannot parse;
- a train/val/held-out CID has no descriptor row, or its row has NaN; or a descriptor norm std is zero;
- any of the 869 molecules gets an all-zero, non-finite or empty encoding from the model's encoder
  (fixed vector, MoLFormer, ChemBERTa, GIN, GAT, D-MPNN), or, for gated bilinear, has no PubChem /
  Mol2vec vector;
- any dataset row (train, val, held-out) has a non-finite feature, or a PGB fingerprint marked unavailable.

The checks save and restore every random-number generator, so they don't change training. Results are
identical with or without them: the Morgan-PGB retrain still gives 20/24.

## Recipe notes

- **Baseline:** descriptor normalisation stats come from the train rows and are saved with the
  checkpoint. Held-out molecules get the fixed vector of their CID, or one computed from SMILES
  (`experiments/split_8020/vectors.py`).
- **PGB:** train rows get leave-own-cluster-out excipient priors: the prior table is refit without the
  row's API cluster, using the split's `cluster_id` column, as in `final_project/run.py`.
- **Gated Bilinear priors (`gb_prior.py`):** in `train_gb.py`, a train row's family prior and its
  logit-offset prior `p_exact` (`logit = residual + s·logit(p_exact) + b`) always counted that row's
  own label. Only the exact count left out the row's cluster, and only when the CSV had an `API_Cluster`
  column; the old `data/train.csv` had none. Here every prior quantity of a train row comes from the
  train rows outside its API cluster, as in PGB. Val and held-out rows use all train rows, unchanged.
  `--gb_original_prior` restores the old behaviour, if you want to compare.
- **Gated Bilinear, PubChem / Mol2vec:** 16 held-out molecules are missing from `data/pubchem_fps.csv`
  and `data/mol2vec_embeddings.csv`; they are filled from `data/*_heldout.csv` instead of becoming zeros.

## Data facts (not errors, worth knowing)

- **Metformin HCl (CID 14219):** the held-out file writes it as a different tautomer
  (`CN(C)C(=N)NC(=N)N.Cl`) than train (`CN(C)C(=N)N=C(N)N.Cl`). Its held-out descriptors are computed
  from its own SMILES, so they differ from the train ones (TPSA 88.99 vs 91.49, H-bond donors 4 vs 3).
  Fixed-vector models look the vector up by CID, so they get the train tautomer's vector.
- **One train API (a peptide, 3 rows) has 526 tokens.** MoLFormer and ChemBERTa truncate it to 512,
  the same way in every run.
