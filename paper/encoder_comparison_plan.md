# Encoder comparison: 9 families on one fixed protocol

Goal: choose the encoder family for ARES on validation only, so the 22 held-out pairs stay an
untouched test set. One run per family, single fold, everything else identical.

## 1. What stays fixed (identical for all 9 runs)

| Item | Value |
|---|---|
| Data | `data/splits/api_cluster/` — train 2770, val 774 (the Morgan model's split) |
| Held-out | the 22 pairs; scored once per family, **after** the choice is made |
| Head | PGB head, unchanged (rank-16 bilinear, gate, prior-trust MLP) |
| Excipient prior | 5-value table, fit on train only, leave-own-cluster-out for train rows |
| Mechanism flags | the 5 SMARTS pair flags |
| Per-molecule extras | 18 chemistry flags, 12 PGB descriptors (train-only norm stats), 8 salt context, 1 availability |
| Loss | asymmetric focal, gamma+ 1.2, gamma- 2.5, pos_weight from train, no balanced sampler |
| Optimiser | AdamW, PGB lr 1.2e-3, encoder lr 3e-4, weight decay 2e-3, batch 64, grad clip 1.0 |
| Epoch budget | max 40, early stopping on val PR-AUC, patience 7, LR schedule patience 4 |
| Threshold | tuned on val only: best MCC with TNR >= 0.97 |
| Seed | 42 |
| Device | one device for all runs (CPU or GPU, not mixed) |

## 2. The confound to remove: the fingerprint mix

Earlier runs gave each family a different primary/secondary fingerprint, so "encoder" and
"fingerprint" were mixed together:

| Family | Old primary | Old secondary |
|---|---|---|
| morgan | Morgan 1024 | MACCS 167 |
| maccs | MACCS 167 | Morgan 512 |
| pubchemfp | PubChem 881 | Morgan 512 |
| mol2vec | Mol2vec 300 | Morgan 512 |
| chemberta | Morgan 512 | none |
| molformer / gin / gat / dmpnn | MACCS 167 | none |

**Rule for the new runs:** the family's own representation is the *only* molecular representation
it gets. No secondary fingerprint for anyone.

| Family | Representation into the tower |
|---|---|
| morgan | Morgan 1024 (r=2, salt-aware OR-fold) |
| maccs | MACCS 167 |
| pubchemfp | PubChem 881 (lookup file) |
| mol2vec | Mol2vec 300 (lookup file) |
| molformer | MoLFormer-XL embedding |
| chemberta | ChemBERTa embedding |
| gin | pretrained GIN embedding |
| gat | pretrained GAT embedding |
| dmpnn | D-MPNN (CheMeleon) embedding |

The 4 sequence/graph families keep cross-attention + their pooling, since that is the architecture
they are built for; the 4 fingerprint families and Mol2vec use concat. This difference is reported,
not hidden: it is a property of the family, not a free choice.

## 3. Selection rule (fixed before any run)

- **Primary metric: validation PR-AUC.** The positive class is ~9%, so PR-AUC is the right summary.
- Tie-break 1: validation MCC. Tie-break 2: fewer parameters.
- The held-out 22 pairs are **not** part of the rule.
- Report val PR-AUC, ROC-AUC, MCC, F1, precision, recall, confusion matrix, threshold, parameters,
  epochs trained, runtime.
- Held-out numbers are recorded for every family but only read after the winner is fixed.

## 4. Honest reporting of the noise

Single fold cannot separate close models. So:

- Report the full 9-family table, not just the winner.
- Mark families whose val PR-AUC is within +-0.03 of the best as "not separable at one fold".
- If Morgan is not the single best, say so plainly and state what the selection rule picked.
- Note in Methods that the original choice had been made on held-out counts, and that this
  protocol replaces it.

## 5. Steps

1. Add a comparison runner that takes `--family`, builds the config from the table above, and
   writes to `results/encoder_comparison/<family>/` (metrics JSON, predictions, run info, history).
   It reuses the existing PGB head, dataset, trainer and threshold code; no new model code.
2. Smoke-test one fingerprint family and one deep family for 1 epoch, confirm shapes and that the
   towers receive only the intended representation.
3. Run all 9, seed 42, same device, logging each run.
4. Collect into one table `results/encoder_comparison/summary.csv` ranked by val PR-AUC.
5. Apply the selection rule; record the winner.
6. Only then add the held-out column and look at it.

## 6. What this does and does not settle

- Settles: which representation the PGB head works best with, on a criterion that never touched
  the test set.
- Does not settle: whether the difference is real. One fold, one seed. If the top families are
  within noise, the follow-up is 5 seeds or grouped 5-fold CV on the same protocol, and the paper
  should say the single-fold ranking is indicative.
- Does not change: the shipped ARES model or any of its stored results.

## 7. Settled while writing the code (`experiments/encoder_comparison/`)

- **Deep families get no fingerprint at all.** Their one representation is the learned embedding,
  so the tower drops the primary fingerprint block entirely (`towers.py`). Previously they got
  MACCS or Morgan on top of the embedding.
- **The prior comes from ARES, not from this repository.** `src/pgb_prior.py` here is the old
  version with the global rate hard-coded to 0.094; ARES uses the train positive rate (0.0996).
  `ares_prior.py` loads the ARES module so the recipe matches the shipped model. Everything else
  (model, trainer, losses, features, metrics) is byte-identical between the two codebases.
- **PubChem / Mol2vec lookups are joined on CID, not SMILES.** 13 molecules are written
  differently in those tables (tautomer / Kekulé / stereo), and a SMILES lookup would hand the
  model all-zero vectors for them without any error. The runner now fails loudly instead.
- **Held-out metrics are written to `_heldout_sealed/`** and excluded from the summary table until
  `collect.py --reveal-heldout`, so the test set cannot inform the choice even by accident.
- **Same epoch budget for everyone:** 40 with early stopping. The old runs gave the deep families 60.
- **Three shared-code fixes, applied to all nine families** (each kept backward compatible, so every
  stored checkpoint still scores exactly as before):
  1. the 8 salt-context values are z-scored with train statistics — previously `organic_heavy_atoms`
     entered the mix layer raw at up to 291 while its neighbours were 0/1 bits or unit-scale;
  2. `PGBHead.init_bias`'s docstring corrected (the zero-residual logit is about 0.7x the global
     log-odds, not 1.2x, because the prior-trust weight starts near 0.5);
  3. `FixedVectorEncoder.output_dim` now reports 1024 for Morgan and 167 for MACCS instead of
     silently falling back to mol2vec's 300.
- **Two asymmetries stay and get reported:** fusion (concat vs cross-attention) and trainable
  parameter count (110k–220k for fingerprints, 500k–1.35M for deep families).

## 8. Input audit (run before the comparison)

`audit_inputs.py` pushes all 879 molecules through every representation. Clean on parsing,
all-zero fingerprints and embeddings, NaN/inf, descriptor variance, and encoder freezing.
Two findings to report in the paper rather than fix:

- **1 molecule of 879 is truncated** for MolFormer and ChemBERTa (526 -> 512 tokens). It is a
  524-character peptide occurring in 3 train pairs only, so it touches neither the validation
  ranking nor the held-out score. The graph encoders keep all 291 of its atoms.
- **20 of 774 val rows and 2 of 22 held-out rows** have an excipient the prior has never seen;
  the table falls back to the train global rate and flags `unseen = 1`. By design, and identical
  for all nine families.

## 9. Open question

If this protocol picks a family other than Morgan, decide in advance what happens: keep Morgan and
report the comparison as a limitation, or move ARES to the winner and retrain. Agreeing this now
avoids a second selection-on-results problem.
