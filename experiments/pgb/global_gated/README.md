# PGB Global Gated Attention Pooling Experiment

This package maintains a standalone variant of the **ChemBERTa** and **DMPNN** models using **`global_gated_attention`** pooling instead of the default **`cls`** pooling.

All existing code, checkpoints, and metrics for the standard 9 PGB models remain completely untouched and separate.

---

## 1. Overview of Configuration Differences

| Model | Encoder | Original Pooling | This Variant Pooling | Checkpoints Path | Metrics Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ChemBERTa** | `chemberta` | `cls` | `global_gated_attention` | `checkpoints/pgb/global_gated/chemberta/` | `experiments/results/metrics_pgb/global_gated/chemberta/` |
| **DMPNN** | `dmpnn_chemprop` | `cls` | `global_gated_attention` | `checkpoints/pgb/global_gated/dmpnn/` | `experiments/results/metrics_pgb/global_gated/dmpnn/` |

---

## 2. Dataset & Split Specification

This variant trains on the **exact same group-split dataset** as the original group-split single runs:
- **Train Dataset**: `data/train.csv` (2,030 rows)
- **Validation Dataset**: `data/val.csv` (842 rows)
- **Test Dataset**: `data/test.csv` (672 rows)
- **Held-Out Test Set**: `held_out_testset/held_out_test_set.csv` (24 pairs)

---

## 3. Training Commands

You can run training for ChemBERTa, DMPNN, or both:

```bash
# 1. Train ChemBERTa with global_gated_attention pooling:
python -m experiments.pgb.global_gated.single_run --family chemberta

# 2. Train DMPNN with global_gated_attention pooling:
python -m experiments.pgb.global_gated.single_run --family dmpnn

# 3. Train both sequentially:
python -m experiments.pgb.global_gated.single_run --family all
```

Optional arguments:
- `--epochs <N>`: Override default max epochs (default: 60)
- `--device <cuda/cpu>`: Specify compute device (default: auto)
- `--seed <int>`: Random seed (default: 42)

---

## 4. Inference & Predictions

After training, evaluate the models on the 24-pairs held-out test set:

```bash
python -m experiments.pgb.global_gated.inference
```

Outputs predictions to: `held_out_testset/held_out_predictions_pgb_global_gated.csv`
