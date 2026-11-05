# SHAP analysis of the Morgan-PGB model

SHAP attributions over all 2,468 input features of the shipped model
(`checkpoints/morgan_pgb_8020_seed42`, 18/22 held-out).

- Script: `scripts/shap/run_shap.py`
- Library code: `apiexc/explain/` (feature names, model wrappers, feature extraction, plots)
- Results: `results/shap/<split>/`

## Run (from any directory)

```bash
python scripts/shap/run_shap.py --split heldout
python scripts/shap/run_shap.py --split val
python scripts/shap/run_shap.py --split train
python scripts/shap/run_shap.py --split full
python scripts/shap/run_shap.py --csv my_pairs.csv
```

| `--split` | Data | Pairs | Time on GPU |
|---|---|---|---|
| `heldout` (default) | `data/heldout/held_out_test_set.csv` | 22 | about 1.5 min |
| `val` | `data/splits/api_cluster/val.csv` | 774 | about 15–25 min |
| `train` | `data/splits/api_cluster/train.csv` | 2770 | about 1 h |
| `full` | `data/raw/start_dataset.csv` | 3544 | about 1.5 h |

Options:

- `--max_samples N`: explain a random N rows.
- `--bg_samples 100`: number of background rows.
- `--nsamples 200`: expected-gradient samples per row. More means a better additivity check.
- `--checkpoint <dir>`: explain another Morgan-PGB checkpoint, e.g. `checkpoints/morgan_pgb_8020_retrain_seed42`.
- `--device cpu`.
- `--out <dir>`.

## Outputs (`results/shap/<split>/`)

```
summary_report.md         run checks, feature-group shares, top 20 features, decoded Morgan bits
feature_importance.csv    all 2,468 features: mean |SHAP|, share, mean SHAP (direction), std, max
predictions.csv           per pair: names, CIDs, label, probability, prediction (threshold 0.671), correct, top-3 features
shap_values.npz           shap_values [N, 2468], base_value, X, y, logits, tower_shap [N, 238], feature_names
plots/
  feature_group_importance.png      share of the 12 feature groups
  top_features_overall.png          top 25 features
  descriptor_attribution.png        12 physicochemical descriptors (API and excipient)
  chemistry_flags_attribution.png   18 functional-group flags
  mechanism_flags_attribution.png   5 reaction-mechanism flags
  prior_attribution.png             excipient prior vector
  morgan_top_bits.png               top Morgan bits with decoded substructures
  tower_level_attribution.png       API tower vs excipient tower vs pair features
  class_comparison.png              incompatible vs compatible signatures
  waterfall_<i>_<API>__<excipient>.png   per-pair explanation (every pair when <= 30, else 5 per class)
```

## Method and checks

- **Explainer:** `shap.GradientExplainer` (expected gradients) on the logit. Positive SHAP values push
  towards *incompatible*.
- **Background:** 100 random train rows, with the leave-own-cluster-out excipient priors they had in
  training.
- **Fidelity check, before any SHAP value:** the flat 2,468-feature wrapper must give the same logits
  as the model's own forward pass. The run stops if the maximum difference is above 1e-4; the held-out
  run gives exactly 0. Its probabilities match `results/reference/predictions_heldout.csv` to 1.5e-7,
  i.e. 18/22.
- **Feature names:** checked against `apiexc/features/` at startup (descriptor, chemistry-flag and
  mechanism order).
- **Additivity:** expected gradients is an approximation, so base + sum(SHAP) does not equal the logit
  exactly. The report gives the gap. On the held-out set, the mean gap is 0.10 against a logit standard
  deviation of 1.1. Raise `--nsamples` to tighten it.
- **Morgan-bit decoding:** done exactly as `apiexc.features.molecular.morgan_bits` computes the
  fingerprint (each `.`-fragment separately, `rdFingerprintGenerator`, radius 2, 1024 bits). A top bit
  that is set in none of the explained molecules gets its attribution from the substructure being
  *absent*. The report then shows which train substructure it stands for.
- **Binary bits:** expected gradients treats fingerprint bits as continuous inputs, so read bit
  attributions as relative importance.
