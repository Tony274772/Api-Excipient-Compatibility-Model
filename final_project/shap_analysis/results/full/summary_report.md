# Morgan-PGB (final_project) — SHAP report

- **Model**: Prior-Gated Bilinear head on Morgan-1024 + MACCS (`checkpoints/morgan_pgb_8020_seed42`), threshold 0.671
- **Data**: `data/start_dataset.csv` (full), 3544 pairs explained
- **Explainer**: shap.GradientExplainer (expected gradients) on the logit, background = train rows with their training priors
- **Baseline (mean background logit)**: -1.0001 (probability 26.89%)
- **Fidelity check**: flat-feature wrapper vs model forward, max |logit difference| = 9.54e-07
- **Additivity** (expected gradients is approximate): mean |base + sum(SHAP) − logit| = 0.0976, max = 0.6040 (logit std over these rows 1.215)
- **Predictions**: 3415/3544 correct at the model threshold (TP 265, TN 3150, FP 50, FN 79)
- Runtime 690 s

## 1. Feature-group share of mean |SHAP|

| Rank | Group | Features | Total mean \|SHAP\| | Share |
|---|---|---|---|---|
| 1 | `api_primary` | 1024 | 0.9087 | 36.06% |
| 2 | `prior_vec` | 5 | 0.7500 | 29.76% |
| 3 | `api_secondary` | 167 | 0.2205 | 8.75% |
| 4 | `exc_primary` | 1024 | 0.1638 | 6.50% |
| 5 | `api_pgb_desc` | 12 | 0.1370 | 5.44% |
| 6 | `api_salt` | 8 | 0.1054 | 4.18% |
| 7 | `exc_secondary` | 167 | 0.0895 | 3.55% |
| 8 | `exc_salt` | 8 | 0.0741 | 2.94% |
| 9 | `exc_pgb_desc` | 12 | 0.0519 | 2.06% |
| 10 | `api_flags` | 18 | 0.0078 | 0.31% |
| 11 | `exc_flags` | 18 | 0.0059 | 0.23% |
| 12 | `mech_flags` | 5 | 0.0052 | 0.21% |

## 2. Top 20 features

| Rank | Feature | Group | Mean \|SHAP\| | Share | Mean SHAP (+ pushes towards incompatible) |
|---|---|---|---|---|---|
| 1 | `Prior_p_exact` | `prior_vec` | 0.6513 | 25.85% | +0.0035 |
| 2 | `API_salt_organic_heavy_atoms` | `api_salt` | 0.0930 | 3.69% | -0.0179 |
| 3 | `Prior_log_n` | `prior_vec` | 0.0841 | 3.34% | -0.0132 |
| 4 | `Exc_salt_organic_heavy_atoms` | `exc_salt` | 0.0338 | 1.34% | +0.0021 |
| 5 | `API_RingCount` | `api_pgb_desc` | 0.0300 | 1.19% | +0.0003 |
| 6 | `API_QED` | `api_pgb_desc` | 0.0224 | 0.89% | +0.0048 |
| 7 | `Exc_salt_inorganic_heavy_atoms` | `exc_salt` | 0.0171 | 0.68% | +0.0016 |
| 8 | `API_NumHeteroatoms` | `api_pgb_desc` | 0.0139 | 0.55% | +0.0006 |
| 9 | `API_NumAromaticRings` | `api_pgb_desc` | 0.0127 | 0.51% | +0.0014 |
| 10 | `Exc_FractionCSP3` | `exc_pgb_desc` | 0.0124 | 0.49% | +0.0006 |
| 11 | `API_MolLogP` | `api_pgb_desc` | 0.0121 | 0.48% | +0.0038 |
| 12 | `API_Morgan_Bit_807` | `api_primary` | 0.0107 | 0.42% | -0.0000 |
| 13 | `API_NumRotatableBonds` | `api_pgb_desc` | 0.0103 | 0.41% | +0.0017 |
| 14 | `Prior_p_delta` | `prior_vec` | 0.0101 | 0.40% | -0.0007 |
| 15 | `API_TPSA` | `api_pgb_desc` | 0.0097 | 0.38% | -0.0004 |
| 16 | `Exc_salt_max_abs_formal_charge` | `exc_salt` | 0.0091 | 0.36% | -0.0002 |
| 17 | `API_Morgan_Bit_514` | `api_primary` | 0.0086 | 0.34% | -0.0007 |
| 18 | `API_Morgan_Bit_80` | `api_primary` | 0.0082 | 0.33% | +0.0008 |
| 19 | `Exc_salt_num_metal_atoms` | `exc_salt` | 0.0075 | 0.30% | +0.0008 |
| 20 | `API_MolWt` | `api_pgb_desc` | 0.0073 | 0.29% | -0.0010 |

## 3. Top Morgan bits, decoded (salt-aware, same generator as the model)

| Feature | Most common substructure | Molecules with the bit | Occurrences |
|---|---|---|---|
| `API_Morgan_Bit_807` | `[O]` | 409 | 1275 |
| `API_Morgan_Bit_514` | `[Cl]` | 68 | 76 |
| `API_Morgan_Bit_80` | `[C]` | 359 | 1147 |
| `API_Morgan_Bit_428` | `cnc` | 153 | 242 |
| `API_Morgan_Bit_396` | `ccc(F)c(c)F` | 13 | 17 |
| `API_Morgan_Bit_200` | `cc(c)N` | 35 | 41 |
| `API_Morgan_Bit_333` | `cc(c)c` | 89 | 94 |
| `API_Morgan_Bit_656` | `[O]` | 138 | 207 |
| `API_Morgan_Bit_90` | `[N]` | 120 | 160 |
| `API_Morgan_Bit_389` | `CO` | 181 | 286 |

SHAP values explain the model's logit (log-odds of incompatible). Positive = pushes towards incompatible.
Expected gradients treats binary fingerprint bits as continuous inputs, so read bit attributions as relative importance.
