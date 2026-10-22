# SHAP analysis: final_project Morgan-PGB model

SHAP attributions over all 2,468 input features of the shipped model
(`checkpoints/morgan_pgb_8020_seed42`, 20/24 held-out). It uses only this folder's code (`src/`), data
(`data/`) and checkpoint. This is the same analysis set as `project/shap_analysis`, which explains
the older 60/20/20 model.

## Run (from inside `final_project/`)

```bash
python -m shap_analysis.run_shap --split heldout
python -m shap_analysis.run_shap --split val
python -m shap_analysis.run_shap --split train
python -m shap_analysis.run_shap --split full
python -m shap_analysis.run_shap --csv my_pairs.csv
```

| `--split` | Data | Pairs | Time on GPU |
|---|---|---|---|
| `heldout` (default) | `data/held_out_test_set.csv` | 24 | about 1.5 min |
| `val` | `data/val.csv` | 774 | about 15–25 min |
| `train` | `data/train.csv` | 2770 | about 1 h |
| `full` | `data/start_dataset.csv` | 3544 | about 1.5 h |

Options:

- `--max_samples N`: explain a random N rows.
- `--bg_samples 100`: number of background rows.
- `--nsamples 200`: expected-gradient samples per row. More means a better additivity check.
- `--checkpoint <dir>`: explain another Morgan-PGB checkpoint, e.g. `checkpoints/morgan_pgb_8020_retrain_seed42`.
- `--device cpu`.
- `--out <dir>`.

## Outputs (`shap_analysis/results/<split>/`)

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
  run gives exactly 0. Its probabilities match `reference_results/predictions_heldout.csv` to 1.5e-7,
  i.e. 20/24.
- **Feature names:** checked against `src/` at startup (descriptor, chemistry-flag and mechanism order).
- **Additivity:** expected gradients is an approximation, so base + sum(SHAP) does not equal the logit
  exactly. The report gives the gap. On the held-out set, the mean gap is 0.11 against a logit standard
  deviation of 1.1. Raise `--nsamples` to tighten it.
- **Morgan-bit decoding:** done exactly as `src.features.morgan_bits` computes the fingerprint (each
  `.`-fragment separately, `rdFingerprintGenerator`, radius 2, 1024 bits). A top bit that is set in
  none of the explained molecules gets its attribution from the substructure being *absent*. The
  report then shows which train substructure it stands for.
- **Binary bits:** expected gradients treats fingerprint bits as continuous inputs, so read bit
  attributions as relative importance.
