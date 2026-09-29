# Loss & Sampler Ablation Study (4 × 2 = 8 Experiments)

This directory contains the ablation study examining the interaction between **Classification Loss Functions** and **Batch Balancing via `WeightedRandomSampler`** on the best performing model architecture:
**`molformer_cross_attn_global_gated`**.

---

## 1. Experimental Design Matrix

All 8 experiments use the exact same:
- **Base Architecture:** MoLFormer-XL encoder + Cross-Attention fusion + Global Gated Attention pooling + 21 physicochemical descriptors
- **Data Split:** Group-Safe API-level Butina cluster split (`data/train.csv`, `data/val.csv`, `data/test.csv`) with zero cluster leakage
- **Effective Batch Size:** 64
- **Learning Rate:** 1.5e-4 (AdamW + ReduceLROnPlateau)
- **Evaluation Protocol:** Validation threshold tuning sweeping [0.001, 1.0) to maximize F1, evaluated on the test set.

| Exp ID | Experiment Name | Sampler | Loss | Description |
| :---: | :--- | :---: | :---: | :--- |
| **A** | `exp_A_bce_sampler_on` | **ON** | **BCE** | Standard BCE with 1:1 balanced sampling |
| **B** | `exp_B_bce_sampler_off` | **OFF** | **BCE** | Standard BCE with natural class distribution (~90:10) |
| **C** | `exp_C_asl_sampler_on` | **ON** | **ASL** | Asymmetric Loss ($\gamma_-=4.0, \gamma_+=1.0$) + Balanced Sampler |
| **D** | `exp_D_asl_sampler_off` | **OFF** | **ASL** | Asymmetric Loss with natural class distribution |
| **E** | `exp_E_weighted_bce_sampler_off` | **OFF** | **Weighted BCE** | BCE with `pos_weight ≈ 9.0` alone (no sampler) |
| **F** | `exp_F_weighted_bce_sampler_on` | **ON** | **Weighted BCE** | BCE with `pos_weight ≈ 9.0` **AND** balanced sampler (dual weighting) |
| **G** | `exp_G_focal_sampler_on` | **ON** | **Focal** | Focal loss ($\gamma=2.0, \alpha=0.25$) + Balanced Sampler |
| **H** | `exp_H_focal_sampler_off` | **OFF** | **Focal** | Focal loss with natural class distribution |

---

## 2. Directory Structure

```
experiments/
├── run_ablation.py                 # Self-contained CLI runner for all or individual ablations
├── README.md                       # Experiment documentation and architecture overview
└── results/                        # Generated results directory
    ├── ablation_summary.csv        # Consolidated tabular comparison across all 8 runs
    ├── ablation_summary.json       # Consolidated JSON metrics
    ├── ablation_summary.md         # Auto-generated markdown comparison table
    ├── exp_A_bce_sampler_on/
    │   ├── checkpoints/best_model.pt
    │   ├── metrics/
    │   │   ├── val_metrics.json
    │   │   ├── test_metrics.json
    │   │   └── training_history.json
    │   └── config.json
    ├── exp_B_bce_sampler_off/
    └── ...
```

---

## 3. Running the Ablation

### Run All 8 Ablations Sequentially:
```bash
python experiments/run_ablation.py
# or explicitly:
python experiments/run_ablation.py --exp all
```

### Run an Individual Experiment (e.g. Experiment A or E):
```bash
python experiments/run_ablation.py --exp A
python experiments/run_ablation.py --exp E
```

### Run a Subset of Experiments:
```bash
python experiments/run_ablation.py --exp A,B,E,F
```

---

## 4. Key Questions Answered by This Ablation

1. **Does the sampler help or hurt standard BCE? (A vs B)**
   - In standard BCE without loss weighting, does forcing 50:50 batches via `WeightedRandomSampler` improve PR-AUC over training on the naturally imbalanced (~90:10) distribution?
2. **Double Weighting vs Pure Loss Weighting in Weighted BCE (E vs F)**
   - When using `weighted_bce` (`pos_weight ≈ 9.0`), does combining it with balanced sampling (dual weighting, giving incompatible samples ~9× gradient weight in already balanced batches) perform better or worse than applying `pos_weight` on the raw natural distribution?
3. **Does Focal Loss need a Balanced Sampler? (G vs H)**
   - Focal Loss down-weights easy negatives via $(1-p_t)^\gamma$. Does it perform better on raw batches or on balanced batches?
4. **Does ASL need a Balanced Sampler? (C vs D)**
   - Asymmetric Loss dynamically shifts probabilities ($p - m$) and suppresses easy negatives with $\gamma_-=4.0$. Does it replace the need for the sampler entirely?
