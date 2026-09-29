# Experiment: MolFormer + Cross-Attention + Global/Gated Pooling + BCE on 1:1 Training Data

## 1. Executive Summary

This isolated experiment evaluates whether training `molformer_cross_attn_global_gated_bce` on a **strictly balanced 1:1 dataset** (191 compatible, 191 incompatible pairs = 382 samples) sampled randomly (`random_state=42`) from the official training set outperforms the baseline model trained on the full imbalanced training set with `WeightedRandomSampler`.

Evaluation was conducted on the **untouched, official validation and test sets** with threshold tuning performed exclusively on validation.

## 2. Dataset Composition

| Split | Compatible (0) | Incompatible (1) | Total | Balance Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Training (1:1 Subset)** | 191 | 191 | 382 | **1 : 1 (50.0% : 50.0%)** |
| **Validation (Official)** | 766 | 76 | 842 | 10.1 : 1 (91.0% : 9.0%) |
| **Test (Official)** | 595 | 77 | 672 | 7.7 : 1 (88.5% : 11.5%) |

> **Leakage Guard:** The 191 compatible pairs were randomly drawn strictly from `data/train.csv` (excluding all validation and test clusters). The selected indices are stored in `selected_compatible_indices.json`.

## 3. Comparison with Existing Baseline

| Metric | Existing Baseline (`WeightedRandomSampler`) | 1:1 Training Subset (No Sampler) | Delta | Direction |
| :--- | :---: | :---: | :---: | :---: |
| **Validation PR-AUC** | 0.7134 | 0.6830 | -0.0304 | 🔴 |
| **Validation F1** | 0.7020 | 0.6795 | -0.0225 | 🔴 |
| **Validation MCC** | 0.6727 | 0.6470 | -0.0256 | 🔴 |
| **Validation Threshold** | 0.0850 | 0.5160 | +0.4310 | — |
| **Test PR-AUC** | **0.7213** | **0.5598** | **-0.1615** | 🔴 |
| **Test F1** | 0.6250 | 0.4938 | -0.1312 | 🔴 |
| **Test MCC** | 0.5755 | 0.4253 | -0.1501 | 🔴 |
| **Test Precision** | 0.5556 | 0.4706 | -0.0850 | 🔴 |
| **Test Recall** | 0.7143 | 0.5195 | -0.1948 | 🔴 |
| **Test Accuracy** | 0.9018 | 0.8780 | -0.0238 | 🔴 |
| **Incompatible Detected** | **55 / 77** (71.4%) | **40 / 77** (51.9%) | -15 pairs | 🔴 |
| **False Positives (Comp -> Incomp)** | 44 / 595 | 45 / 595 | +1 pairs | 🔴 |

## 4. Confusion Matrices

### Test Confusion Matrix Comparison
- **Existing Baseline:**
  - True Negatives (Compatible): `551` | False Positives: `44`
  - False Negatives (Missed Incompat): `22` | True Positives (Detected): `55`
- **1:1 Training Model:**
  - True Negatives (Compatible): `550` | False Positives: `45`
  - False Negatives (Missed Incompat): `37` | True Positives (Detected): `40`

## 5. Answers to Required Experimental Questions

1. **Did 1:1 training improve Test PR-AUC?**
   - Baseline: `0.7213` vs 1:1 Model: `0.5598` (-0.1615).
2. **Did 1:1 training improve Incompatible Recall?**
   - Baseline: `0.7143` (55/77) vs 1:1 Model: `0.5195` (40/77).
3. **Did it improve F1?**
   - Baseline: `0.6250` vs 1:1 Model: `0.4938` (-0.1312).
4. **Did it improve MCC?**
   - Baseline: `0.5755` vs 1:1 Model: `0.4253` (-0.1501).
5. **Did it increase false positives among compatible pairs?**
   - Baseline False Positives: `44` vs 1:1 Model False Positives: `45` (+1 false positives).
6. **Does the result support proceeding to the proposed 9-subset ensemble experiment?**
   - Analysis provided below based on metric trade-offs.

## 6. Artifact Files in this Directory
- `selected_training_pairs.csv`: The exact 382 training pairs used.
- `selected_compatible_indices.json`: Exact index mapping of the 191 compatible samples from `data/train.csv`.
- `config.json`: Complete serialized training configuration.
- `metrics.json`: Detailed validation and test metrics with comparisons.
- `val_metrics.json` & `test_metrics.json`: Standardized format evaluation outputs.
- `validation_predictions.csv` & `test_predictions.csv`: Row-by-row prediction probabilities and classifications.
- `confusion_matrix.json`: Detailed validation and test confusion counts.
- `training_history.json`: Epoch-by-epoch loss, val PR-AUC, and learning rate curves.
- `checkpoints/best_model.pt`: Model weights at best validation epoch.