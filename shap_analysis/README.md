# SHAP Interpretability Suite — PGB Morgan Compatibility Model

This module provides comprehensive model interpretability and feature attribution for the **Prior-Gated Bilinear (PGB) Morgan** API–Excipient compatibility model.

---

## 🔬 What This Module Analyzes

The PGB Morgan model maps **2,468 input features** to an API–Excipient compatibility logit:

| Feature Group | Dimension | Description |
|---|---|---|
| `API Morgan Fingerprint` | 1024 | Circular substructures at radius 2 (primary feature) |
| `Excipient Morgan Fingerprint` | 1024 | Circular substructures at radius 2 (primary feature) |
| `API MACCS Keys` | 167 | Structural keys (secondary feature) |
| `Excipient MACCS Keys` | 167 | Structural keys (secondary feature) |
| `API PGB Descriptors` | 12 | Physicochemical descriptors (MolWt, LogP, TPSA, etc.) |
| `Excipient PGB Descriptors` | 12 | Physicochemical descriptors (MolWt, LogP, TPSA, etc.) |
| `API Salt Context` | 8 | Charges, inorganic heavy atoms, fragment counts |
| `Excipient Salt Context` | 8 | Charges, inorganic heavy atoms, fragment counts |
| `API Chemistry Flags` | 18 | Reactive functional group indicators |
| `Excipient Chemistry Flags` | 18 | Reactive functional group indicators |
| `Bayesian Prior Vector` | 5 | Empirical excipient prior (p_exact, p_delta, p_family, log_n, unseen) |
| `Mechanism Reaction Flags` | 5 | Known degradation reaction matches (Maillard, hydrolysis, etc.) |

---

## 🚀 Commands to Run SHAP

Each command automatically routes outputs to its own dedicated subfolder:

### 1. Run Full SHAP on Test Set (672 pairs)
```bash
python -m shap_analysis.run_shap --test_csv data/test.csv --test_samples 0 --device cuda
```
*Saved in:* `shap_analysis/results/test_set/`

### 2. Run Full SHAP on Validation Set (765 pairs)
```bash
python -m shap_analysis.run_shap --test_csv data/val.csv --test_samples 0 --device cuda
```
*Saved in:* `shap_analysis/results/val_set/`

### 3. Run Full SHAP on Held-Out Test Set (24 pairs)
```bash
python -m shap_analysis.run_shap --test_csv held_out_testset/held_out_test_set.csv --test_samples 0 --device cuda
```
*Saved in:* `shap_analysis/results/held_out_testset/`

### 4. Run Full SHAP on Complete Study Dataset (All 3,544 pairs)
```bash
python -m shap_analysis.run_shap --test_csv data/start_dataset.csv --test_samples 0 --device cuda
```
*Saved in:* `shap_analysis/results/full_dataset/`

---

## 📊 Isolated Output Folders

The results for each dataset are stored independently:

```
shap_analysis/results/
├── test_set/                       # Full SHAP on data/test.csv (672 pairs)
│   ├── shap_values.npz
│   ├── feature_importance.csv
│   ├── summary_report.md
│   └── plots/
├── val_set/                        # Full SHAP on data/val.csv (765 pairs)
│   ├── shap_values.npz
│   ├── feature_importance.csv
│   ├── summary_report.md
│   └── plots/
├── held_out_testset/               # Full SHAP on held_out_test_set.csv (24 pairs)
│   ├── shap_values.npz
│   ├── feature_importance.csv
│   ├── summary_report.md
│   └── plots/
└── full_dataset/                   # Full SHAP on data/start_dataset.csv (3,544 pairs)
    ├── shap_values.npz
    ├── feature_importance.csv
    ├── summary_report.md
    └── plots/
```
    ├── feature_group_importance.png    # Cumulative attribution across the 12 feature groups
    ├── top_features_overall.png        # Top 25 individual features driving predictions
    ├── descriptor_attribution.png      # API & Excipient physicochemical descriptor importance
    ├── chemistry_flags_attribution.png # Impact of 18 functional group flags
    ├── mechanism_flags_attribution.png # Chemical degradation reaction mechanism attribution
    ├── prior_attribution.png           # Bayesian prior integration & dependence
    ├── morgan_top_bits.png             # Decoded circular substructures with RDKit
    ├── tower_level_attribution.png     # API Tower vs. Excipient Tower contribution
    ├── class_comparison.png            # Feature signatures separating incompatible from compatible pairs
    └── waterfall_sample_*.png          # Per-sample waterfall explanations for individual drug pairs
```

---

## 🧬 Substructure Decoding

Individual Morgan fingerprint bits are decoded to actual chemical substructure fragments using RDKit atom environments via `decode_morgan_bit()` in `shap_analysis/utils.py`.
