# Morgan-PGB (final_project) — SHAP report

- **Model**: Prior-Gated Bilinear head on Morgan-1024 + MACCS (`checkpoints/morgan_pgb_8020_seed42`), threshold 0.671
- **Data**: `data/val.csv` (val), 774 pairs explained
- **Explainer**: shap.GradientExplainer (expected gradients) on the logit, background = train rows with their training priors
- **Baseline (mean background logit)**: -1.0001 (probability 26.89%)
- **Fidelity check**: flat-feature wrapper vs model forward, max |logit difference| = 9.54e-07
- **Additivity** (expected gradients is approximate): mean |base + sum(SHAP) − logit| = 0.0906, max = 0.5316 (logit std over these rows 1.091)
- **Predictions**: 734/774 correct at the model threshold (TP 38, TN 696, FP 10, FN 30)
- Runtime 350 s

## 1. Feature-group share of mean |SHAP|

| Rank | Group | Features | Total mean \|SHAP\| | Share |
|---|---|---|---|---|
| 1 | `api_primary` | 1024 | 1.0595 | 39.63% |
| 2 | `prior_vec` | 5 | 0.7430 | 27.79% |
| 3 | `api_secondary` | 167 | 0.2300 | 8.60% |
| 4 | `exc_primary` | 1024 | 0.1646 | 6.16% |
| 5 | `api_pgb_desc` | 12 | 0.1430 | 5.35% |
| 6 | `api_salt` | 8 | 0.0980 | 3.67% |
| 7 | `exc_secondary` | 167 | 0.0900 | 3.37% |
| 8 | `exc_salt` | 8 | 0.0738 | 2.76% |
| 9 | `exc_pgb_desc` | 12 | 0.0522 | 1.95% |
| 10 | `api_flags` | 18 | 0.0083 | 0.31% |
| 11 | `exc_flags` | 18 | 0.0059 | 0.22% |
| 12 | `mech_flags` | 5 | 0.0052 | 0.19% |

## 2. Top 20 features

| Rank | Feature | Group | Mean \|SHAP\| | Share | Mean SHAP (+ pushes towards incompatible) |
|---|---|---|---|---|---|
| 1 | `Prior_p_exact` | `prior_vec` | 0.6385 | 23.88% | +0.0277 |
| 2 | `Prior_log_n` | `prior_vec` | 0.0880 | 3.29% | -0.0036 |
| 3 | `API_salt_organic_heavy_atoms` | `api_salt` | 0.0833 | 3.12% | -0.0437 |
| 4 | `Exc_salt_organic_heavy_atoms` | `exc_salt` | 0.0333 | 1.25% | +0.0025 |
| 5 | `API_RingCount` | `api_pgb_desc` | 0.0293 | 1.10% | +0.0026 |
| 6 | `API_QED` | `api_pgb_desc` | 0.0246 | 0.92% | +0.0074 |
| 7 | `Exc_salt_inorganic_heavy_atoms` | `exc_salt` | 0.0172 | 0.64% | +0.0010 |
| 8 | `API_NumHeteroatoms` | `api_pgb_desc` | 0.0148 | 0.55% | +0.0020 |
| 9 | `API_MolLogP` | `api_pgb_desc` | 0.0144 | 0.54% | +0.0071 |
| 10 | `API_NumAromaticRings` | `api_pgb_desc` | 0.0130 | 0.48% | +0.0018 |
| 11 | `Exc_FractionCSP3` | `exc_pgb_desc` | 0.0126 | 0.47% | +0.0005 |
| 12 | `API_Morgan_Bit_807` | `api_primary` | 0.0115 | 0.43% | +0.0030 |
| 13 | `API_Morgan_Bit_514` | `api_primary` | 0.0107 | 0.40% | -0.0007 |
| 14 | `API_TPSA` | `api_pgb_desc` | 0.0107 | 0.40% | -0.0002 |
| 15 | `API_Morgan_Bit_80` | `api_primary` | 0.0107 | 0.40% | -0.0011 |
| 16 | `API_NumRotatableBonds` | `api_pgb_desc` | 0.0100 | 0.38% | +0.0030 |
| 17 | `Prior_p_delta` | `prior_vec` | 0.0099 | 0.37% | -0.0009 |
| 18 | `API_Morgan_Bit_656` | `api_primary` | 0.0092 | 0.35% | +0.0039 |
| 19 | `API_Morgan_Bit_117` | `api_primary` | 0.0092 | 0.35% | -0.0028 |
| 20 | `Exc_salt_max_abs_formal_charge` | `exc_salt` | 0.0089 | 0.33% | -0.0003 |

## 3. Top Morgan bits, decoded (salt-aware, same generator as the model)

| Feature | Most common substructure | Molecules with the bit | Occurrences |
|---|---|---|---|
| `API_Morgan_Bit_807` | `[C]` | 87 | 228 |
| `API_Morgan_Bit_514` | `[Cl]` | 18 | 21 |
| `API_Morgan_Bit_80` | `[C]` | 71 | 215 |
| `API_Morgan_Bit_656` | `[O]` | 29 | 37 |
| `API_Morgan_Bit_117` | `CNC` | 12 | 12 |
| `API_Morgan_Bit_200` | `cc(c)N` | 12 | 13 |
| `API_Morgan_Bit_389` | `CO` | 39 | 57 |
| `API_Morgan_Bit_396` | `CCC(CC)(CN)NC` | 1 | 1 |
| `API_Morgan_Bit_90` | `[N]` | 19 | 27 |
| `API_Morgan_Bit_428` | `cnc` | 28 | 40 |

SHAP values explain the model's logit (log-odds of incompatible). Positive = pushes towards incompatible.
Expected gradients treats binary fingerprint bits as continuous inputs, so read bit attributions as relative importance.
