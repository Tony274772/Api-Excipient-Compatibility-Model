# Ablation plan: `chemberta__x_pubchemfp`

Status: **implemented, not run** (code in `experiments/ablations_chemberta/`). Written 2026-11-11. This is the same kind of study as the ARES (Morgan-PGB)
ablations in `final_project/docs/ablations.md`, applied to the encoder-grid cell chosen as the
candidate final model. `final_project/` is not modified; the new harness lives in `experiments/`.

## 1. The model being ablated

`chemberta__x_pubchemfp` (encoder grid, `fp_secondary` block), 559,739 trainable parameters.

| Part | What it is |
|---|---|
| Primary (struct) | ChemBERTa-77M-MTR, frozen, projected to 128, cross-attention API <-> excipient, CLS pooling |
| Secondary | PubChem fingerprint, 881 bits -> 32 (secondary block) |
| Shared tower features | 12 descriptors -> 24, 18 chemistry flags -> 16, 8 salt-context values, mix -> 96 |
| Not present | availability flag / `missing_primary` embedding (ComparisonTower has neither) |
| Head | PGB head, unchanged from ARES: 5-D excipient prior, gate (46 -> 16), bilinear rank 16, MLP 410, prior trust `w`, 5 SMARTS mechanism flags (v1) |
| Prior | train positive rate (0.0996) already, family lookup on |

Baseline numbers (grid, seeds 42-46):

| | Val PR-AUC | Val MCC | Held-out correct |
|---|---|---|---|
| per seed, mean ± sd | 0.7359 ± 0.0192 | 0.679 | 17.2 (18, 17, 18, 15, 18) |
| 5-seed ensemble (threshold from val) | 0.7531 | — | 18 / 22 |

## 2. Protocol (held fixed in every run)

Identical to `experiments/encoder_comparison/run_grid.py`, so every ablation is paired with the
grid baseline seed by seed:

- data `data/splits/api_cluster/` train 2770 / val 774; held-out = the 22 pairs only
- seeds 42, 43, 44, 45, 46 (all five, every ablation)
- asymmetric focal loss, AdamW, PGB lr 1.2e-3, encoder-side lr 3e-4, batch 64, max 40 epochs,
  early stop on val PR-AUC (patience 7)
- threshold: best val MCC with TNR >= 0.97
- held-out written to a sealed folder; **not used for any decision** (22 pairs cannot rank models:
  seed noise 0.037 > between-model spread 0.031)

As in ARES, each variant builds the full model with the seed and then cuts its part out of the
existing weight matrices, so every kept layer starts from the same weights as the baseline.

**Step 0, before any ablation:** retrain `full` through the new harness for seeds 42-46 and check that
it reproduces the grid runs (val PR-AUC per seed within ±0.005 on the same device). Every comparison
uses this harness baseline, not the grid numbers.

## 3. Ablations

### A. Head (inherited from ARES, same switches)

| Name | Change | Question |
|---|---|---|
| `no_w` | prior trust MLP removed, `z = r + b` | Does the learned prior shortcut matter? (ARES: biggest loss, -0.012) |
| `prior1` | prior 5 -> 1 (`p_exact` only) | Are the 4 extra prior columns needed? |
| `prior1+no_w` | both | Smallest prior form |
| `no_prior` | excipient prior removed from gate, MLP and `w` | Is it the prior or the towers that predict? |
| `no_gate` | gate on the bilinear term removed | (ARES: removing it helped slightly) |
| `no_bilinear` | U, V and gate removed | Is the explicit pair interaction needed? |
| `no_mech` | 5 SMARTS mechanism flags removed from the head | (ARES: no effect) |
| `full --mech v2` | 13 literature mechanism rules in place of 5 | Do better rules help on this model? |

### B. Tower shared features (inherited)

| Name | Change |
|---|---|
| `no_salt` | 8-value salt context removed from both towers (mix -8) |
| `no_flags` | 18 chemistry flags removed from towers and gate |
| `no_flags_api` / `no_flags_exc` | one side only |
| `no_flags_metal` | the 10 metal / inorganic flags on both sides |
| `no_flags_metal+no_salt` | do the metal flags and the salt context cover for each other? |

`no_salt` and the metal flags matter more here than in ARES. About 45% of molecules give ChemBERTa
tokens it never learned (Cl, Br, [Na+], stereo), so salts may reach this model only through these
features.

### C. Data (inherited)

| Name | Change | Compare with |
|---|---|---|
| `prior1+no_family` | prior lookup exact -> nearest -> global (no family step) | `prior1` (identical weights) |

### D. New ablations for this model's description

| Name | Change | Question | Already in grid? |
|---|---|---|---|
| `no_secondary` | PubChem branch removed (mix -32) | Does PubChem add anything over ChemBERTa? | close: `chemberta` cell (0.7251), but a fresh init; rerun as a cut |
| `no_chemberta` | ChemBERTa struct branch, cross-attention and pooling removed; PubChem stays as secondary | Is the encoder needed at all? | no (`pubchemfp` alone uses a 64-wide primary block, so it is a different model) |
| `no_cross_attn` | the two cross-attention layers, their LayerNorms and the token placeholder cut (132,736 params); struct = the molecule's own projected ChemBERTa embedding, PubChem kept | Does cross-attention earn its parameters here? | close: `chemberta__noca` has no PubChem and uses a different (table + 64-wide) path |
| `pool_gated` | CLS -> global gated-attention pooling, PubChem kept | The best single cell was `chemberta__pool_gated` (0.7446) | no |
| `no_desc` | 12 descriptors removed from both towers (mix -24) | Not tested in ARES | no |
| `shuf_pubchemfp` | PubChem bits permuted across molecules (sha1 logged) | Is the gain from the fingerprint's chemistry or from the extra 28k parameters? | no (only shuffled MACCS / Morgan were run) |
| `chemberta_parent` | for molecules with `.`, `[`, `@`, `/` or `\` only: ChemBERTa reads the largest organic fragment (water dropped), neutralised, without stereo; every other molecule keeps its exact string; Cl / Br stay; salt context, flags and the rest read the original SMILES | Does dropping the tokens it cannot read help? | no |

### E. Evaluation only (no training)

- **Untrained-token subset.** Split val pairs by whether either molecule has a ChemBERTa token with
  untrained embeddings (`audit_pretrained.py` list). Report PR-AUC / MCC per subset for `full`,
  `no_chemberta` and `chemberta_parent`. If `full` is clearly worse on the affected subset, the
  "45% unreadable" objection holds and `chemberta_parent` or `no_chemberta` becomes the fix.
- **5-seed ensemble** for every ablation (val threshold), as `collect_seeds.py` did for ARES.

### Not carried over

| ARES ablation | Why not |
|---|---|
| `no_avail` | ComparisonTower has no availability flag or `missing_primary` embedding |
| `train_global`, `old_global` | the grid already uses the train rate (0.0996); nothing to switch |
| `design_a` / `design_b` | those combinations were built from ARES results; new designs come from section 5 |

## 4. Size

23 variants (incl. `full`) × 5 seeds = 115 runs. At about 1.2 min per run on the DGX GPU (grid timing)
this is roughly 2.3 GPU-hours. `chemberta_parent` needs no precompute: ChemBERTa encodes the parent
SMILES live, as it does the originals. Exact parameter counts: `run_ablation --list`.

## 5. Analysis and decision rule

Fixed before running:

1. Compare each ablation with the harness `full` using a paired test across the 5 seeds on val
   PR-AUC (same test as `collect_grid.py`), with BH correction over all ablations at q = 0.10.
   Report the effect size, the 95% CI and the smallest effect detectable at n = 5.
2. A part counts as **needed** if removing it lowers val PR-AUC significantly after correction.
3. A part counts as **removable** if removing it does not lower val PR-AUC by more than 0.01 and
   val MCC does not drop by more than 0.02 (mean over seeds).
4. Removable parts are combined into at most two candidate designs (as `design_a` / `design_b` in
   ARES). Each design is run on seeds 42-46 and must again pass rule 3 against `full`.
5. The final model is `full` unless a design passes rule 4 with fewer parameters.
6. Held-out (22 pairs) is revealed only after the choice, and is reported, never used to choose.

## 6. Implementation

Implemented in `experiments/ablations_chemberta/` (see its README.md). `src/` is unchanged: the
switches live in subclasses (`AblationPGBHead`, `AblationComparisonTower`, `AblationPGBModel`) that
take over the built model's weights, so `full` runs the original code paths. The only edit outside
the folder is one added line in `experiments/encoder_comparison/reps.py` registering `shuf_pubchemfp`
(no grid cell uses it). `final_project/` is not modified or imported.

- `variants.py`: cuts, `PLAN`, `BUILD_SETTINGS`, `REFERENCE`, `DESIGNS`
- `run_ablation.py`: one run, same step order as `run_grid.run_cell`; `--list`, `--smoke`
- `run_seeds.py`, `run_dgx.sh`: the sweep, resumable, sharded over GPUs, step 0 first
- `collect.py`: step-0 check, section-5 rule, ensembles; `token_subsets.py`: section E

Parameter counts from `--list` match the grid where the same model exists: `full` 559,739 (grid
cell), `no_secondary` 497,019 (grid `chemberta`), `pool_gated` 692,347 (grid `chemberta__pool_gated`
+ the 62,720 PubChem parameters).

## 7. Order of work

1. Harness + switches; `--list` param table; smoke run (1 epoch) for every variant.
2. Step 0: `full` × 5 seeds, check against the grid.
3. Sections A-C (inherited), then D (new), then E.
4. Collect, apply section 5, run designs, reveal held-out, write up.
