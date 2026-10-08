# PGB (Prior-Gated Bilinear) Experiment Package

This directory contains the **PGB (Prior-Gated Bilinear)** architecture implementation, maintained as an independent architecture separate from [`experiments/gated_bilinear/`](../gated_bilinear/).

---

## 1. Overview of the Two Architectures

| Feature | `experiments/gated_bilinear/` | `experiments/pgb/` |
| :--- | :--- | :--- |
| **Model Class** | `GatedBilinearModel` (`PriorGatedBilinearHead`) | `PGBCompatibilityModel` (`PGBHead`) |
| **Data Split** | Standard 60/20/20 in `data/` | Standard 60/20/20 in `data/` |
| **Bilinear Rank** | Rank-16 factored projection | Rank-16 factored projection |
| **Gate Inputs** | 28 features (prior 5 + excipient flags 18 + mechanism 5) | 46 features (prior 5 + API flags 18 + excipient flags 18 + mechanism 5) |
| **Concat Vector** | 426-d (includes both raw and gated bilinear) | 410-d (gated bilinear + representations + flags) |
| **Prior Weighting** | Fixed learned scale $s$ | Dynamic adaptive `prior_trust_mlp` |
| **Checkpoints** | `checkpoints/gated_bilinear/<family>/` | `checkpoints/pgb/<family>/` |
| **Metrics Folder** | `experiments/results/metrics_bilinear/<family>/` | `experiments/results/metrics_pgb/<family>/` |
| **Single Run Script** | `python -m experiments.gated_bilinear.train_gb` | `python -m experiments.pgb.single_run` |
| **CV Script** | N/A (single training loop) | `python -m experiments.pgb.cross_validate` |
| **Inference Script** | `python -m experiments.gated_bilinear.inference_gb` | `python -m experiments.pgb.inference` or `python Inference/inference_pgb_single.py` |
| **Predictions CSV** | `held_out_testset/held_out_predictions_gatedbilinear.csv` | `held_out_testset/held_out_predictions_pgb_single.csv` |

---

## 2. Directory Structure

```
experiments/pgb/
├── __init__.py           # Package exports
├── config.py             # PGBConfig & 9 family specifications
├── features.py           # Molecular descriptors, flags & fingerprints
├── prior.py              # ExcipientPriorTable & leave-cluster-out priors
├── dataset.py            # PGBDataset, pgb_collate_fn, PGBNormStats
├── model.py              # MoleculeTower, PGBHead, PGBCompatibilityModel
├── train.py              # PGB training loop with dual learning rate
├── single_run.py         # 60/20/20 data/ pipeline & CLI runner
├── cross_validate.py     # 5-fold Stratified Group K-Fold CV & CLI runner
├── inference.py          # Held-out 24-pair test set evaluation & predictions export
└── README.md             # Architecture documentation
```

---

## 3. Metrics Output Format

When training via `single_run.py`, metrics are saved in `experiments/results/metrics_pgb/<family>/`:
- `val_metrics.json`
- `test_metrics.json`
- `heldout_metrics.json`
- `training_history.json`

---

## 4. Usage

### Single-Run Training (Using 60/20/20 Data in `data/`)
```bash
# Run one family:
python -m experiments.pgb.single_run --family maccs
python -m experiments.pgb.single_run --family molformer

# Run all 9 families:
python -m experiments.pgb.single_run --family all
```

### 5-Fold Cross-Validation
```bash
python -m experiments.pgb.cross_validate --family maccs
python -m experiments.pgb.cross_validate --family all
```

### Inference on 24-Pairs Held-Out Test Set
```bash
python -m experiments.pgb.inference
# or:
python Inference/inference_pgb_single.py
```
Outputs predictions to: `held_out_testset/held_out_predictions_pgb_single.csv`
