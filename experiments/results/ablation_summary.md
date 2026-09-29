# Loss & Sampler Ablation Study Results

**Base Model Architecture:** `molformer_cross_attn_global_gated`
**Data Split:** Group-safe API-level Butina cluster split (`data/train.csv`, `data/val.csv`, `data/test.csv`)

## Summary Comparison Table

| Exp | Sampler | Loss | Val PR-AUC | Val F1 | Val MCC | Test PR-AUC | Test F1 | Test MCC | Test Prec | Test Rec | Test Acc | Thresh |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | ON | BCE | 0.7134 | 0.7020 | 0.6727 | **0.7213** | 0.6250 | 0.5755 | 0.5556 | 0.7143 | 0.9018 | 0.0850 |
| **B** | OFF | BCE | 0.7724 | 0.6923 | 0.6788 | **0.6467** | 0.5217 | 0.4718 | 0.5902 | 0.4675 | 0.9018 | 0.1890 |
| **C** | ON | ASL | 0.7304 | 0.6772 | 0.6671 | **0.6246** | 0.5303 | 0.4891 | 0.6364 | 0.4545 | 0.9077 | 0.7850 |
| **D** | OFF | ASL | 0.7590 | 0.7075 | 0.6800 | **0.6386** | 0.5325 | 0.4683 | 0.4891 | 0.5844 | 0.8824 | 0.3870 |
| **E** | OFF | WEIGHTED_BCE | 0.6944 | 0.6410 | 0.6046 | **0.6656** | 0.5955 | 0.5416 | 0.5248 | 0.6883 | 0.8929 | 0.0720 |
| **F** | ON | WEIGHTED_BCE | 0.6837 | 0.6479 | 0.6175 | **0.5012** | 0.5031 | 0.4368 | 0.4878 | 0.5195 | 0.8824 | 0.9770 |
| **G** | ON | FOCAL | 0.7334 | 0.6968 | 0.6662 | **0.7012** | 0.6429 | 0.5949 | 0.5934 | 0.7013 | 0.9107 | 0.4160 |
| **H** | OFF | FOCAL | 0.7571 | 0.6826 | 0.6512 | **0.6828** | 0.6270 | 0.5804 | 0.5370 | 0.7532 | 0.8973 | 0.1860 |

## Key Insights & Discussion

1. **Sampler ON vs OFF across Losses:**
   - **BCE**: Comparing Exp A (ON) vs Exp B (OFF) tests whether equalizing positive and negative batch representation helps standard cross-entropy.
   - **Weighted BCE**: Comparing Exp E (OFF) vs Exp F (ON) isolates whether `pos_weight` alone is sufficient or whether combining it with balanced sampling produces better discrimination.
   - **Focal Loss**: Comparing Exp G (ON) vs Exp H (OFF) tests how focal modulating factors interact with batch class balance.
   - **ASL (Asymmetric Loss)**: Comparing Exp C (ON) vs Exp D (OFF) tests asymmetric margin clipping and negative suppression under balanced vs natural sampling.
