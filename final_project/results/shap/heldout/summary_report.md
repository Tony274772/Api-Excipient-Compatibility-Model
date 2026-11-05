# Morgan-PGB — SHAP report

- **Model**: Prior-Gated Bilinear head on Morgan-1024 + MACCS (`checkpoints/morgan_pgb_8020_seed42`), threshold 0.671
- **Data**: `data/heldout/held_out_test_set.csv` (heldout), 22 pairs explained
- **Explainer**: shap.GradientExplainer (expected gradients) on the logit, background = train rows with their training priors
- **Baseline (mean background logit)**: -1.0001 (probability 26.89%)
- **Fidelity check**: flat-feature wrapper vs model forward, max |logit difference| = 0.00e+00
- **Additivity** (expected gradients is approximate): mean |base + sum(SHAP) − logit| = 0.1014, max = 0.3085 (logit std over these rows 1.106)
- **Predictions**: 18/22 correct at the model threshold (TP 6, TN 12, FP 0, FN 4)
- Runtime 78 s

## 1. Feature-group share of mean |SHAP|

| Rank | Group | Features | Total mean \|SHAP\| | Share |
|---|---|---|---|---|
| 1 | `api_primary` | 1024 | 1.3680 | 40.99% |
| 2 | `prior_vec` | 5 | 0.9527 | 28.54% |
| 3 | `api_secondary` | 167 | 0.2921 | 8.75% |
| 4 | `exc_primary` | 1024 | 0.1629 | 4.88% |
| 5 | `api_pgb_desc` | 12 | 0.1628 | 4.88% |
| 6 | `api_salt` | 8 | 0.1400 | 4.19% |
| 7 | `exc_salt` | 8 | 0.0964 | 2.89% |
| 8 | `exc_secondary` | 167 | 0.0911 | 2.73% |
| 9 | `exc_pgb_desc` | 12 | 0.0510 | 1.53% |
| 10 | `api_flags` | 18 | 0.0096 | 0.29% |
| 11 | `exc_flags` | 18 | 0.0065 | 0.20% |
| 12 | `mech_flags` | 5 | 0.0044 | 0.13% |

## 2. Top 20 features

| Rank | Feature | Group | Mean \|SHAP\| | Share | Mean SHAP (+ pushes towards incompatible) |
|---|---|---|---|---|---|
| 1 | `Prior_p_exact` | `prior_vec` | 0.8451 | 25.32% | +0.5165 |
| 2 | `API_salt_organic_heavy_atoms` | `api_salt` | 0.1278 | 3.83% | -0.1278 |
| 3 | `Prior_log_n` | `prior_vec` | 0.0785 | 2.35% | +0.0161 |
| 4 | `Exc_salt_organic_heavy_atoms` | `exc_salt` | 0.0557 | 1.67% | +0.0261 |
| 5 | `API_RingCount` | `api_pgb_desc` | 0.0324 | 0.97% | +0.0228 |
| 6 | `API_QED` | `api_pgb_desc` | 0.0292 | 0.87% | +0.0246 |
| 7 | `API_Morgan_Bit_807` | `api_primary` | 0.0244 | 0.73% | -0.0054 |
| 8 | `API_NumRotatableBonds` | `api_pgb_desc` | 0.0175 | 0.52% | +0.0155 |
| 9 | `API_Morgan_Bit_80` | `api_primary` | 0.0174 | 0.52% | -0.0083 |
| 10 | `API_Morgan_Bit_140` | `api_primary` | 0.0167 | 0.50% | +0.0133 |
| 11 | `Prior_p_delta` | `prior_vec` | 0.0148 | 0.44% | -0.0099 |
| 12 | `API_MolLogP` | `api_pgb_desc` | 0.0146 | 0.44% | +0.0115 |
| 13 | `API_NumHeteroatoms` | `api_pgb_desc` | 0.0143 | 0.43% | +0.0095 |
| 14 | `Exc_salt_inorganic_heavy_atoms` | `exc_salt` | 0.0142 | 0.43% | -0.0000 |
| 15 | `Exc_FractionCSP3` | `exc_pgb_desc` | 0.0135 | 0.40% | -0.0017 |
| 16 | `API_Morgan_Bit_514` | `api_primary` | 0.0135 | 0.40% | +0.0020 |
| 17 | `Prior_unseen` | `prior_vec` | 0.0128 | 0.38% | +0.0073 |
| 18 | `Exc_salt_max_abs_formal_charge` | `exc_salt` | 0.0127 | 0.38% | +0.0043 |
| 19 | `API_NumAromaticRings` | `api_pgb_desc` | 0.0127 | 0.38% | +0.0062 |
| 20 | `API_Morgan_Bit_147` | `api_primary` | 0.0121 | 0.36% | +0.0083 |

## 3. Top Morgan bits, decoded (salt-aware, same generator as the model)

| Feature | Most common substructure | Molecules with the bit | Occurrences |
|---|---|---|---|
| `API_Morgan_Bit_807` | `[O]` | 10 | 24 |
| `API_Morgan_Bit_80` | `[C]` | 8 | 16 |
| `API_Morgan_Bit_140` | `CN` | 6 | 8 |
| `API_Morgan_Bit_514` | `[Cl]` | 1 | 1 |
| `API_Morgan_Bit_147` | `[N]` | 12 | 13 |
| `API_Morgan_Bit_396` | `absent here; in train: ccc(F)c(c)F` | 0 | 0 |
| `API_Morgan_Bit_90` | `[N]` | 6 | 9 |
| `API_Morgan_Bit_333` | `cc(c)c` | 4 | 4 |
| `API_Morgan_Bit_659` | `[Cl]` | 4 | 7 |
| `API_Morgan_Bit_389` | `ccccc` | 4 | 7 |

SHAP values explain the model's logit (log-odds of incompatible). Positive = pushes towards incompatible.
Expected gradients treats binary fingerprint bits as continuous inputs, so read bit attributions as relative importance.
