# Ablation plan: `dmpnn__x_maccs` (Final ARES Model)

Status: **implemented, ready to run** (code in `experiments/ablations_dmpnn_maccs/`).
Written for the final candidate model selected from the 110-cell encoder sweep: `dmpnn__x_maccs`.

---

## 1. The model being ablated

`dmpnn__x_maccs` (encoder grid, `fp_secondary` block), **1,366,011 trainable parameters** (10,080,251 total parameters including the frozen CheMeleon backbone).

| Part | What it is |
|---|---|
| Primary (struct) | D-MPNN (CheMeleon checkpoint, 2048-d graph representation, frozen), projected to 128, bidirectional cross-attention API <-> excipient, CLS pooling |
| Secondary | MACCS keys (167 bits) -> projected to 32 (secondary block, 5,376 parameters) |
| Shared tower features | 12 descriptors -> 24, 18 chemistry flags -> 16, 8 salt-context values, mix -> 96 |
| Not present | availability flag / `missing_primary` embedding (ComparisonTower has neither) |
| Head | PGB head, unchanged from ARES: 5-D excipient prior, gate (46 -> 16), bilinear rank 16, MLP 410, prior trust `w`, 5 SMARTS mechanism flags (v1) |
| Prior | train positive rate (0.0996), exact -> family -> nearest -> global lookup |

Baseline numbers (grid, seeds 42-46):

| Metric | Value |
|---|---|
| Val MCC | **0.6654 ± 0.0247** (seeds: 0.682, 0.680, 0.681, 0.661, 0.624) |
| Val PR-AUC | **0.7240 ± 0.0160** |
| Val ROC-AUC | 0.9383 |
| Val F1 | 0.6761 |
| Val Precision | 0.8332 |
| Val TNR | 0.9887 |
| Held-out correct /22 | **17.4** (seeds: 17, 17, 19, 18, 16) |

---

## 2. Protocol (held fixed in every run)

Identical to `experiments/encoder_comparison/run_grid.py`, so every ablation is paired with the grid baseline seed by seed:

- data `data/splits/api_cluster/` train 2,770 / val 774; held-out = 22 pairs (sealed)
- seeds 42, 43, 44, 45, 46 (all five, every ablation)
- asymmetric focal loss, AdamW, PGB lr 1.2e-3, encoder-side lr 3e-4, batch 64, max 40 epochs, early stop on val PR-AUC (patience 7)
- threshold: best val MCC with TNR >= 0.97
- held-out written to a sealed folder; **not used for any decision**

As in ARES, each variant builds the full model with the seed and cuts its part out of existing weight matrices in-place, so every kept layer starts from the exact same weights as the baseline.

**Step 0, before any ablation:** retrain `full` through the new harness for seeds 42-46 and check that it reproduces the grid runs (val PR-AUC per seed within ±0.005 on the same device).

---

## 3. Ablations

### A. Head (PGB decision mechanism)

| Name | Change | Question |
|---|---|---|
| `no_w` | prior trust MLP removed, `z = r + b` | Does the learned prior trust shortcut matter? |
| `prior1` | prior 5 -> 1 (`p_exact` only) | Are the 4 extra prior columns needed? |
| `prior1+no_w` | both | Minimal prior form |
| `no_prior` | excipient prior removed from gate, MLP and `w` | Is it the prior or the graph representations that predict? |
| `no_gate` | gate on the bilinear term removed | Does gating the interaction term improve generalization? |
| `no_bilinear` | U, V and gate removed | Is the explicit pair interaction needed? |
| `no_mech` | 5 SMARTS mechanism flags removed from the head | Do rule flags contribute above graph features? |
| `mech_v2` | 13 literature mechanism rules in place of 5 | Do expanded chemical incompatibility rules help? |

### B. Tower shared features

| Name | Change | Question |
|---|---|---|
| `no_salt` | 8-value salt context removed from both towers (mix -8) | Does salt context help D-MPNN handle ionic formulations? |
| `no_flags` | 18 chemistry flags removed from towers and gate | Are rule-based functional flags redundant? |
| `no_flags_api` / `no_flags_exc` | one side only | Asymmetric contribution |
| `no_flags_metal` | the 10 metal / inorganic flags on both sides | Do metal flags matter when D-MPNN and MACCS are both present? |
| `no_flags_metal+no_salt` | both removed | Do the metal flags and the salt context cover for each other? |
| `no_desc` | 12 RDKit descriptors removed from both towers (mix -24) | Do MW, logP, TPSA add value over the graph embedding? |

### C. Data

| Name | Change | Compare with |
|---|---|---|
| `prior1+no_family` | prior lookup exact -> nearest -> global (no family step) | `prior1` (identical weights) |

### D. Molecular Description (D-MPNN + MACCS specific)

| Name | Change | Question |
|---|---|---|
| `no_secondary` | MACCS branch removed (mix -32, saves 5,376 params) | Does MACCS add anything over D-MPNN alone? (Pure D-MPNN was 0.6245 MCC) |
| `no_dmpnn` | D-MPNN struct branch, cross-attention, projections and pooling removed; MACCS kept | Does the 1.36M-parameter deep graph network earn its place over pure MACCS? |
| `no_cross_attn` | cross-attention layers, LayerNorms and placeholders cut; struct = molecule's own projected D-MPNN embedding; MACCS kept | Does API <-> Excipient message attention help over independent embeddings? |
| `pool_gated` | CLS -> global gated-attention pooling over graph nodes | Can node-level gated attention improve upon CLS graph pooling? |
| `shuf_maccs` | MACCS bits permuted across molecules (control) | Is the gain from MACCS chemistry or from 5k arbitrary extra parameters? |
| `dmpnn_parent` | for multi-fragment salts, D-MPNN only encodes the largest organic fragment (covalent graph), while MACCS, salt context, and flags retain the intact salt | Does removing disconnected ions (deg=0) improve D-MPNN? |

### E. Evaluation only (no training)

- **Salt / Disconnected Ion Breakdown:** Split val pairs into multi-fragment salts vs single-fragment neutral molecules. Compare `full`, `no_secondary`, and `dmpnn_parent`.
- **Sugars and Polyol Excipient Breakdown:** Evaluate performance specifically on non-aromatic polyhydroxy excipients (Glucose, Lactose, Mannitol, Sorbitol).
- **5-seed ensemble** for every ablation (val threshold).

---

## 4. Size

23 variants (incl. `full`) × 5 seeds = 115 runs. At roughly 1.5–2.0 min per run on the DGX GPU, this is ~3.0–3.5 GPU-hours.

---

## 5. Analysis and decision rule

Fixed before running:

1. Compare each ablation with the harness `full` using a paired test across the 5 seeds on val PR-AUC and MCC, with BH correction over all ablations at q = 0.10.
2. A part counts as **NEEDED** if removing it lowers val PR-AUC significantly after correction.
3. A part counts as **REMOVABLE** if removing it does not lower val PR-AUC by more than 0.010 and val MCC does not drop by more than 0.020.
4. Removable parts are combined into candidate designs (`design_a`, `design_b`). Each design is run on seeds 42-46 and must pass rule 3 against `full`.
5. The final model is `full` unless a design passes rule 4 with fewer parameters.
6. Held-out (22 pairs) is revealed only after the choice, and is reported, never used to choose.

