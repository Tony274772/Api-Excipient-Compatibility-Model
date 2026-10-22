# Morgan-PGB (final_project) — SHAP report

- **Model**: Prior-Gated Bilinear head on Morgan-1024 + MACCS (`checkpoints/morgan_pgb_8020_seed42`), threshold 0.671
- **Data**: `data/held_out_test_set.csv` (heldout), 24 pairs explained
- **Explainer**: shap.GradientExplainer (expected gradients) on the logit, background = train rows with their training priors
- **Baseline (mean background logit)**: -1.0001 (probability 26.89%)
- **Fidelity check**: flat-feature wrapper vs model forward, max |logit difference| = 0.00e+00
- **Additivity** (expected gradients is approximate): mean |base + sum(SHAP) − logit| = 0.1143, max = 0.3085 (logit std over these rows 1.107)
- **Predictions**: 20/24 correct at the model threshold (TP 8, TN 12, FP 0, FN 4)
- Runtime 81 s

## 1. Feature-group share of mean |SHAP|

| Rank | Group | Features | Total mean \|SHAP\| | Share |
|---|---|---|---|---|
| 1 | `api_primary` | 1024 | 1.4107 | 42.04% |
| 2 | `prior_vec` | 5 | 0.9490 | 28.28% |
| 3 | `api_secondary` | 167 | 0.2927 | 8.72% |
| 4 | `api_pgb_desc` | 12 | 0.1610 | 4.80% |
| 5 | `exc_primary` | 1024 | 0.1562 | 4.66% |
| 6 | `api_salt` | 8 | 0.1363 | 4.06% |
| 7 | `exc_salt` | 8 | 0.0933 | 2.78% |
| 8 | `exc_secondary` | 167 | 0.0871 | 2.60% |
| 9 | `exc_pgb_desc` | 12 | 0.0491 | 1.46% |
| 10 | `api_flags` | 18 | 0.0093 | 0.28% |
| 11 | `exc_flags` | 18 | 0.0062 | 0.19% |
| 12 | `mech_flags` | 5 | 0.0044 | 0.13% |

## 2. Top 20 features

| Rank | Feature | Group | Mean \|SHAP\| | Share | Mean SHAP (+ pushes towards incompatible) |
|---|---|---|---|---|---|
| 1 | `Prior_p_exact` | `prior_vec` | 0.8389 | 25.00% | +0.5588 |
| 2 | `API_salt_organic_heavy_atoms` | `api_salt` | 0.1244 | 3.71% | -0.1244 |
| 3 | `Prior_log_n` | `prior_vec` | 0.0824 | 2.46% | +0.0147 |
| 4 | `Exc_salt_organic_heavy_atoms` | `exc_salt` | 0.0551 | 1.64% | +0.0180 |
| 5 | `API_RingCount` | `api_pgb_desc` | 0.0313 | 0.93% | +0.0190 |
| 6 | `API_QED` | `api_pgb_desc` | 0.0294 | 0.88% | +0.0255 |
| 7 | `API_Morgan_Bit_807` | `api_primary` | 0.0225 | 0.67% | -0.0042 |
| 8 | `API_Morgan_Bit_80` | `api_primary` | 0.0168 | 0.50% | -0.0063 |
| 9 | `API_NumRotatableBonds` | `api_pgb_desc` | 0.0166 | 0.49% | +0.0148 |
| 10 | `API_Morgan_Bit_140` | `api_primary` | 0.0155 | 0.46% | +0.0120 |
| 11 | `API_NumHeteroatoms` | `api_pgb_desc` | 0.0150 | 0.45% | +0.0104 |
| 12 | `Prior_p_delta` | `prior_vec` | 0.0148 | 0.44% | -0.0104 |
| 13 | `API_MolLogP` | `api_pgb_desc` | 0.0142 | 0.42% | +0.0111 |
| 14 | `Exc_salt_inorganic_heavy_atoms` | `exc_salt` | 0.0129 | 0.38% | -0.0008 |
| 15 | `API_Morgan_Bit_514` | `api_primary` | 0.0127 | 0.38% | +0.0024 |
| 16 | `Exc_salt_max_abs_formal_charge` | `exc_salt` | 0.0125 | 0.37% | +0.0046 |
| 17 | `Exc_FractionCSP3` | `exc_pgb_desc` | 0.0124 | 0.37% | -0.0013 |
| 18 | `API_NumAromaticRings` | `api_pgb_desc` | 0.0123 | 0.37% | +0.0070 |
| 19 | `API_Morgan_Bit_147` | `api_primary` | 0.0120 | 0.36% | +0.0072 |
| 20 | `API_Morgan_Bit_396` | `api_primary` | 0.0116 | 0.35% | +0.0116 |

## 3. Top Morgan bits, decoded (salt-aware, same generator as the model)

| Feature | Most common substructure | Molecules with the bit | Occurrences |
|---|---|---|---|
| `API_Morgan_Bit_807` | `[O]` | 11 | 28 |
| `API_Morgan_Bit_80` | `[C]` | 9 | 17 |
| `API_Morgan_Bit_140` | `CN` | 6 | 8 |
| `API_Morgan_Bit_514` | `[Cl]` | 1 | 1 |
| `API_Morgan_Bit_147` | `[N]` | 12 | 13 |
| `API_Morgan_Bit_396` | `absent here; in train: ccc(F)c(c)F` | 0 | 0 |
| `API_Morgan_Bit_200` | `absent here; in train: cc(c)N` | 0 | 0 |
| `API_Morgan_Bit_90` | `[N]` | 6 | 9 |
| `API_Morgan_Bit_389` | `ccccc` | 4 | 7 |
| `API_Morgan_Bit_333` | `cc(c)c` | 4 | 4 |

SHAP values explain the model's logit (log-odds of incompatible). Positive = pushes towards incompatible.
Expected gradients treats binary fingerprint bits as continuous inputs, so read bit attributions as relative importance.
