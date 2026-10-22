# Morgan-PGB (final_project) — SHAP report

- **Model**: Prior-Gated Bilinear head on Morgan-1024 + MACCS (`checkpoints/morgan_pgb_8020_seed42`), threshold 0.671
- **Data**: `data/train.csv` (train), 2770 pairs explained
- **Explainer**: shap.GradientExplainer (expected gradients) on the logit, background = train rows with their training priors
- **Baseline (mean background logit)**: -1.0001 (probability 26.89%)
- **Fidelity check**: flat-feature wrapper vs model forward, max |logit difference| = 7.15e-07
- **Additivity** (expected gradients is approximate): mean |base + sum(SHAP) − logit| = 0.0985, max = 1.0199 (logit std over these rows 1.247)
- **Predictions**: 2681/2770 correct at the model threshold (TP 227, TN 2454, FP 40, FN 49)
- Runtime 1054 s

## 1. Feature-group share of mean |SHAP|

| Rank | Group | Features | Total mean \|SHAP\| | Share |
|---|---|---|---|---|
| 1 | `api_primary` | 1024 | 0.8653 | 34.95% |
| 2 | `prior_vec` | 5 | 0.7522 | 30.39% |
| 3 | `api_secondary` | 167 | 0.2177 | 8.79% |
| 4 | `exc_primary` | 1024 | 0.1636 | 6.61% |
| 5 | `api_pgb_desc` | 12 | 0.1350 | 5.45% |
| 6 | `api_salt` | 8 | 0.1078 | 4.35% |
| 7 | `exc_secondary` | 167 | 0.0894 | 3.61% |
| 8 | `exc_salt` | 8 | 0.0741 | 2.99% |
| 9 | `exc_pgb_desc` | 12 | 0.0517 | 2.09% |
| 10 | `api_flags` | 18 | 0.0077 | 0.31% |
| 11 | `exc_flags` | 18 | 0.0059 | 0.24% |
| 12 | `mech_flags` | 5 | 0.0053 | 0.21% |

## 2. Top 20 features

| Rank | Feature | Group | Mean \|SHAP\| | Share | Mean SHAP (+ pushes towards incompatible) |
|---|---|---|---|---|---|
| 1 | `Prior_p_exact` | `prior_vec` | 0.6552 | 26.47% | -0.0001 |
| 2 | `API_salt_organic_heavy_atoms` | `api_salt` | 0.0961 | 3.88% | -0.0107 |
| 3 | `Prior_log_n` | `prior_vec` | 0.0829 | 3.35% | -0.0159 |
| 4 | `Exc_salt_organic_heavy_atoms` | `exc_salt` | 0.0338 | 1.37% | +0.0019 |
| 5 | `API_RingCount` | `api_pgb_desc` | 0.0300 | 1.21% | -0.0003 |
| 6 | `API_QED` | `api_pgb_desc` | 0.0218 | 0.88% | +0.0041 |
| 7 | `Exc_salt_inorganic_heavy_atoms` | `exc_salt` | 0.0171 | 0.69% | +0.0017 |
| 8 | `API_NumHeteroatoms` | `api_pgb_desc` | 0.0136 | 0.55% | +0.0001 |
| 9 | `API_NumAromaticRings` | `api_pgb_desc` | 0.0126 | 0.51% | +0.0013 |
| 10 | `Exc_FractionCSP3` | `exc_pgb_desc` | 0.0123 | 0.50% | +0.0006 |
| 11 | `API_MolLogP` | `api_pgb_desc` | 0.0115 | 0.47% | +0.0029 |
| 12 | `API_Morgan_Bit_807` | `api_primary` | 0.0104 | 0.42% | -0.0008 |
| 13 | `API_NumRotatableBonds` | `api_pgb_desc` | 0.0103 | 0.42% | +0.0014 |
| 14 | `Prior_p_delta` | `prior_vec` | 0.0102 | 0.41% | -0.0007 |
| 15 | `API_TPSA` | `api_pgb_desc` | 0.0094 | 0.38% | -0.0004 |
| 16 | `Exc_salt_max_abs_formal_charge` | `exc_salt` | 0.0092 | 0.37% | -0.0002 |
| 17 | `API_Morgan_Bit_514` | `api_primary` | 0.0081 | 0.33% | -0.0007 |
| 18 | `API_Morgan_Bit_80` | `api_primary` | 0.0075 | 0.30% | +0.0013 |
| 19 | `Exc_salt_num_metal_atoms` | `exc_salt` | 0.0074 | 0.30% | +0.0009 |
| 20 | `API_MolWt` | `api_pgb_desc` | 0.0073 | 0.30% | -0.0004 |

## 3. Top Morgan bits, decoded (salt-aware, same generator as the model)

| Feature | Most common substructure | Molecules with the bit | Occurrences |
|---|---|---|---|
| `API_Morgan_Bit_807` | `[O]` | 322 | 1047 |
| `API_Morgan_Bit_514` | `[Cl]` | 50 | 55 |
| `API_Morgan_Bit_80` | `[C]` | 288 | 932 |
| `API_Morgan_Bit_428` | `cnc` | 125 | 202 |
| `API_Morgan_Bit_396` | `ccc(F)c(c)F` | 12 | 16 |
| `API_Morgan_Bit_333` | `cc(c)c` | 74 | 79 |
| `API_Morgan_Bit_90` | `[N]` | 101 | 133 |
| `API_Morgan_Bit_200` | `cc(c)N` | 23 | 28 |
| `API_Morgan_Bit_887` | `CCC` | 61 | 173 |
| `API_Morgan_Bit_389` | `CO` | 142 | 229 |

SHAP values explain the model's logit (log-odds of incompatible). Positive = pushes towards incompatible.
Expected gradients treats binary fingerprint bits as continuous inputs, so read bit attributions as relative importance.
