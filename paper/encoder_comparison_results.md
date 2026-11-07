# Encoder comparison — results

Nine encoder families, one fixed protocol (`encoder_comparison_plan.md`): same split, head, prior,
loss, optimiser, 40-epoch budget with early stopping, val-tuned threshold, seed 42, one GPU, and
exactly one molecular representation per family. Run on 2026-11-05.

## Ranked by validation PR-AUC (the pre-registered selection rule)

| Rank | Family | val PR-AUC | val ROC-AUC | val MCC | val correct | Held-out | Held-out PR-AUC | Params | Epochs |
|---|---|---|---|---|---|---|---|---|---|
| 1 | chemberta | **0.7312** | 0.9483 | 0.6315 | 733/774 | 18/22 | 0.9353 | 497k | 12 |
| 2 | morgan | 0.7131 | 0.9403 | **0.6642** | **736/774** | 19/22 | 0.9295 | 219k | 9 |
| 3 | mol2vec | 0.7126 | 0.9443 | 0.6221 | 733/774 | 16/22 | **0.9667** | 126k | 12 |
| 4 | maccs | 0.6980 | 0.9328 | 0.6348 | 735/774 | 16/22 | 0.9464 | **109k** | 9 |
| 5 | pubchemfp | 0.6872 | 0.9325 | 0.5962 | 731/774 | 17/22 | 0.9263 | 201k | 9 |
| 6 | molformer | 0.6832 | 0.9405 | 0.5883 | 731/774 | 15/22 | 0.9081 | 826k | 9 |
| 7 | gat | 0.6785 | 0.9257 | 0.6350 | 733/774 | **20/22** | 0.9488 | 587k | 12 |
| 8 | dmpnn | 0.6668 | 0.9298 | 0.6134 | 733/774 | 16/22 | 0.9022 | 1349k | 11 |
| 9 | gin | 0.6472 | 0.9377 | 0.5288 | 726/774 | 16/22 | 0.9082 | 587k | 10 |

**Selection rule picks chemberta** (val PR-AUC 0.7312). But chemberta, morgan and mol2vec all sit
within 0.03 val PR-AUC of each other — **not separable at one fold with one seed**. Treat the
ranking as indicative until it is repeated with 5 seeds or grouped 5-fold CV.

## 1. Validation and held-out disagree

Spearman correlation between val PR-AUC and held-out correct count: **0.33**.

- **GAT is 7th of 9 on validation but 1st on held-out** (20/22).
- **MolFormer is 6th on validation and last on held-out** (15/22).
- **Mol2vec has the best held-out PR-AUC (0.9667) but only 16/22 correct**, because PR-AUC ranks
  while the correct count depends on the val-tuned threshold.

This is the clearest possible evidence for the methodological point: selecting on held-out
performance, as the original ARES choice did, picks a different model (GAT) than selecting on
validation (ChemBERTa). Neither is reliable from a single fold.

**Morgan is the one family that does well on both** — 2nd on validation, 2nd on held-out.

## 2. The held-out set is effectively a 10-item test

Every one of the nine families classifies **all 12 compatible pairs correctly. Zero false
positives, in every run.** The entire spread from 15/22 to 20/22 comes from how many of the 10
incompatible pairs each family catches (TP 3 to 8).

| Incompatible pair | Excipient p_exact (train rate) | Families correct (of 9) |
|---|---|---|
| Phenytoin sodium + calcium sulfate | 0.600 | 9 |
| Trichlormethiazide + magnesium oxide | 0.300 | 7 |
| Phenytoin + magnesium oxide | 0.300 | 6 |
| Quercetin + magnesium carbonate | 0.200 | 6 |
| Tetracycline + calcium phosphate | 0.400 | 5 |
| Warfarin + magnesium aluminum silicate | 0.100 | 4 |
| Fluvoxamine + lactose | 0.255 | 3 |
| Acetazolamide + lactose | 0.255 | 2 |
| Phenytoin sodium + lactose | 0.255 | 2 |
| Acetazolamide + mannitol | 0.030 | 1 |

Correlation between the excipient's training incompatibility rate and how many families get the
pair right: **0.72**.

All 12 compatible pairs have p_exact <= 0.109, and most are below 0.05 — so every representation
calls them compatible without needing to reason about the API at all.

## 3. What this means for the paper

- The held-out score is **driven by excipient base rates**, not by API–excipient interaction
  modelling. This matches the earlier finding that the shipped model's prior term alone reaches
  held-out PR-AUC 0.9525 against the full model's 0.9415.
- The pairs every family fails on (acetazolamide + mannitol, the lactose pairs) are exactly those
  whose excipient looks benign in training. These are the cases a genuine interaction model would
  need to get right, and no representation does.
- With 10 positives, one pair is 10 percentage points of recall. Differences of 1-2 pairs between
  families are noise.
- Report the full table, state that the encoder was chosen on validation, and present held-out as a
  single untouched estimate rather than a leaderboard.

## 4. Next steps

1. Repeat the top 3 (chemberta, morgan, mol2vec) with seeds 42-46 to see whether the validation gap
   survives; add GAT, since its held-out result is the outlier worth understanding.
2. Report the prior-only baseline alongside every model, as the honest floor.
3. A larger or more balanced external test set would do more for the paper than any encoder change.
