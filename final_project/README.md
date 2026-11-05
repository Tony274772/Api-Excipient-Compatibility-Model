# Morgan-PGB: API–excipient compatibility prediction

Predicts whether an active pharmaceutical ingredient (API) and an excipient are **incompatible** (label 1)
or **compatible** (label 0) from their SMILES. The model is a Prior-Gated Bilinear (PGB) head on Morgan +
MACCS fingerprints, trained on an 80/20 API-cluster split and tested on 22 external held-out pairs.

This folder is self-contained: code, data, trained checkpoints and results. Nothing outside it is read, and
every command works from any directory, so the folder can be copied or moved anywhere.

| | |
|---|---|
| Model | PGB head on Morgan-1024 + MACCS fingerprints, chemistry flags, PGB descriptors, salt context, excipient prior, mechanism flags |
| Split | train 2770 / val 774 rows (80 / 20 by Butina API cluster, no cluster in both) |
| Test | 22 external held-out pairs (10 incompatible, 12 compatible), none of them in train or val |
| Seed | 42 (best epoch 1, early stopping on val PR-AUC) |
| Threshold | 0.671 (best val MCC with true-negative rate ≥ 0.97) |
| **Held-out (22 pairs)** | **18/22 correct**: TP 6, TN 12, FP 0, FN 4, PR-AUC 0.942, ROC-AUC 0.950, MCC 0.671 |
| Val (774) | PR-AUC 0.686, ROC-AUC 0.924, MCC 0.639, F1 0.655, 734/774 correct |

Full model description: [docs/model_summary.txt](docs/model_summary.txt).

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.12.7, torch 2.5.1, rdkit 2026.03.6, scikit-learn 1.5.1, pandas 2.2.2, numpy 1.26.4
(shap 0.45.1, matplotlib 3.9.2 and seaborn 0.13.2 for the SHAP analysis and figures; pytest for the tests). Training and evaluation run on CPU; a GPU is
optional and faster.

## Quick start

```bash
python run.py evaluate
```

Scores the shipped model on val and the 22 held-out pairs, checks it against `results/reference/` and
prints 18/22. Run it from inside the folder, or from anywhere as `python path/to/final_project/run.py evaluate`.

```bash
python run.py predict --csv my_pairs.csv
```

Scores new pairs. The CSV needs `API_Smiles` and `Excipient_Smiles` columns; `Api_name`, `Excipient_name`
and a label column (`Outcome1` or `ground_truth`) are optional. Output: `results/predictions/`.

```bash
python run.py train
```

Trains from scratch with seed 42 and the original recipe: prints loss and val PR-AUC per epoch (max 40,
early stopping with patience 7), saves the checkpoint to `checkpoints/morgan_pgb_8020_retrain_seed42/`,
then evaluates it. Metrics, predictions, `training_history.json` and `run_info.json` go to
`results/evaluation/morgan_pgb_8020_retrain_seed42/`. The shipped model is never overwritten.
Options: `--seed N`, `--device cpu|cuda`, `--epochs N`, `--overwrite`, `--split NAME`.

```bash
python run.py split
```

Rebuilds train / val from `data/raw/start_dataset.csv`, writes `results/rebuilt_split/split_log.txt` and
confirms the result is identical to `data/splits/api_cluster/`.

The original model was trained on a GPU. A retrain with seed 42 uses the identical config, but GPU and CPU
arithmetic differ slightly, so the held-out count can move by a pair. Other seeds give different models.

## Project layout

```
final_project/
├── run.py                      main entry point: train / evaluate / predict / split
├── requirements.txt
├── apiexc/                     the library (importable Python package)
│   ├── paths.py                every file / folder location, anchored on this folder
│   ├── config.py               Config dataclass (stored inside every checkpoint)
│   ├── pipeline.py             shared recipe: splits, training config, loaders, checkpoint loading, scoring
│   ├── splitting.py            how the default split is made (Butina API clusters + GroupShuffleSplit)
│   ├── dataset.py              PGBDataset: CSV of pairs -> model inputs
│   ├── utils.py                seeding, device selection
│   ├── features/               molecular features, 5 SMARTS mechanism flags, excipient prior table
│   ├── models/                 the PGB model and its (weight-free) encoder
│   ├── training/               training loop with early stopping, losses
│   ├── evaluation/             metrics, threshold tuning, prediction tables
│   ├── ablations/              model variants, mechanism flags v2, prior tables for the ablation study
│   └── explain/                SHAP wrappers, feature names, plots
├── scripts/                    command-line tools (all run from any directory)
│   ├── ablations/              run_ablation.py, run_seeds.py, collect_seeds.py, check_mech_v2.py
│   ├── shap/                   run_shap.py
│   ├── benchmark/              score_validation_pairs.py
│   ├── data/                   make_split_variants.py, make_api_cluster06_split.py
│   └── figures/                make_architecture_diagram.py, make_heldout_comparison.py, make_mechanism_flags_table.py
├── data/                       inputs (never written by training)
│   ├── raw/start_dataset.csv   all 3544 labelled pairs the splits are made from
│   ├── heldout/                held_out_test_set.csv: the 22 external test pairs (label column ground_truth)
│   ├── splits/<name>/          train.csv, val.csv, split_log.txt per split (api_cluster = default)
│   ├── benchmark/              extra validation pairs for scripts/benchmark/
│   └── baselines/              held-out predictions of DeInteract and Towards Safe (comparison figure)
├── checkpoints/
│   ├── morgan_pgb_8020_seed42/ THE SHIPPED MODEL (18/22): best_model.pt, prior_table.json,
│   │                           pgb_norm_stats.npz, threshold.json
│   ├── morgan_pgb_<...>_seed42/ retrain and the models trained on the other splits (split.txt = split name)
│   └── ablations/              ablation checkpoints (+ archive_global0094/)
├── results/                    everything the code writes
│   ├── reference/              original metrics and predictions of the shipped model
│   ├── evaluation/<checkpoint>/ run.py train / evaluate output
│   ├── ablations/              runs/, multi_seed/, logs/, summary.csv, archive_global0094/
│   ├── shap/<split>/           SHAP reports, values and plots
│   ├── benchmark/              benchmark predictions
│   └── figures/                paper figures
├── docs/                       model_summary.txt, ablations.md, shap_analysis.md
└── tests/                      pytest checks (python -m pytest tests)
```

Old checkpoints pickle their config as `src.config.Config` (the code lived in `src/` when they were trained);
`apiexc.pipeline.torch_load` maps that to `apiexc.config` when loading, so the checkpoint files are unchanged.

## Data and splits

1. Butina-cluster the 530 unique API SMILES on Morgan fingerprints (radius 2, 2048 bits) with Tanimoto
   distance cutoff 0.15 (similarity ≥ 0.85). This gives 512 clusters, stored as `cluster_id`.
2. `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` grouped by `cluster_id` gives
   train (2770 rows, 10.0 % incompatible) and val (774 rows, 8.8 % incompatible). The split is not
   stratified on the label.
3. The test set is the 22 held-out pairs (10 incompatible, 12 compatible).

No API cluster or API CID is in both train and val. Excipients are shared by design. No held-out pair
(API_CID + Excipient_CID) is in train or val; `run.py` checks this before every train and evaluate.

### Split variants

`python scripts/data/make_split_variants.py` and `python scripts/data/make_api_cluster06_split.py` build
more train / val splits from `start_dataset.csv` to compare split strategies. All use seed 42 and the same
22 held-out pairs. Train on one with `python run.py train --split <name>`; the checkpoint goes to
`checkpoints/morgan_pgb_<name>_seed<N>/` and `python run.py evaluate --checkpoint <that folder>` scores it on
its own split.

| `--split` | Split | Train (incompatible) | Val (incompatible) | Unseen in val | Held-out (seed 42) |
|---|---|---|---|---|---|
| `api_cluster` (default) | API group (above) | 2770 (276, 10.0 %) | 774 (68, 8.8 %) | APIs | 18/22 |
| `random` | random 80/20, stratified on the label | 2835 (275, 9.7 %) | 709 (69, 9.7 %) | nothing | 15/22 |
| `excipient_new` | stratified group split on `exc_group_id` | 2828 (274, 9.7 %) | 716 (70, 9.8 %) | excipients | 14/22 |
| `api_excipient_new` | val API clusters of `api_cluster` × val excipient groups of `excipient_new` | 2201 (220, 10.0 %) | 147 (14, 9.5 %) | APIs and excipients | 14/22 |
| `api_cluster06` | API group split at Tanimoto ≥ 0.6 | 2821 (256, 9.1 %) | 723 (88, 12.2 %) | APIs and close analogues | 17/22 |

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

In `api_excipient_new`, the 1196 pairs that have one side in train groups and the other in val groups are
dropped (`dropped_mixed.csv`). Each split's `split_log.txt` has the overlap checks and the merge list.

## Training recipe

- The PGB descriptor norm stats and the excipient prior table are fit on train only. Train rows use
  leave-own-cluster-out priors.
- Loss: asymmetric focal (γ+ 1.2, γ− 2.5) with a positive-class weight and no balanced sampler.
- Optimiser: AdamW, LR 1.2e-3, weight decay 2e-3, batch size 64, gradient clip 1.0.
- Early stopping and the LR schedule (ReduceLROnPlateau, factor 0.5, patience 4) use val PR-AUC:
  max 40 epochs, patience 7.
- The threshold is chosen on val. The held-out pairs are only scored.

## Other tools

| Task | Command | Docs / output |
|---|---|---|
| Ablation study | `python scripts/ablations/run_ablation.py list` | [docs/ablations.md](docs/ablations.md), `results/ablations/` |
| SHAP analysis | `python scripts/shap/run_shap.py --split heldout` | [docs/shap_analysis.md](docs/shap_analysis.md), `results/shap/` |
| Benchmark pairs | `python scripts/benchmark/score_validation_pairs.py` | `results/benchmark/<csv name>_predictions.csv` |
| Figures | `python scripts/figures/make_heldout_comparison.py` (and the other two) | `results/figures/` |
| Tests | `python -m pytest tests` | `tests/` |

## Known issues to disclose

- **Held-out pairs that were in train.** Two pairs of the external held-out list, Hydrocortisone +
  magnesium stearate and Hydrocortisone + β-cyclodextrin, are also in `start_dataset.csv` (and so in
  train) with the same label. They are not part of the held-out set, which therefore has 22 pairs.
- **Near-duplicate in val.** `val.csv` contains Tetracycline + tricalcium phosphate, the same
  chemistry as held-out pair #4 (Tetracycline + calcium phosphate).
- **No internal test set.** Val is used for early stopping and the threshold, so the val metrics are
  optimistic; the held-out set is the only untouched test, and with 22 pairs one pair moves accuracy by
  about 4.5 points.

In the original project this model was `run_pgb_single.py --family morgan` (commit `f534ef7`, trained
2026-10-06).
