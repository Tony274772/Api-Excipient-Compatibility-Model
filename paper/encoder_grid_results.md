# Encoder grid — results

Design and pre-registered decision rule: `paper/encoder_grid_plan.md`.
Per-run numbers: `paper/encoder_grid.csv`. Matrix: `paper/encoder_grid_matrix.csv`.

Selection metric is **validation PR-AUC**, mean over seeds 42–46. The 22 held-out pairs are
**still sealed** and were not read for anything below.

---

## 0. The sweep is complete and clean

| Check | Result |
|---|---|
| Cells in the design | 101 new + 9 reused = **110** |
| Runs | **550** (505 new + 45 reused), every cell with all 5 seeds |
| Cells with a missing seed | **0** |
| Runs missing any of their 6 output files | **0** |
| Runs that crashed | **0** (no traceback in any log, no non-zero exit) |
| Epochs trained | min 8, median 10, max 21 — **none hit the 40 cap**, so early stopping bound every run |
| Held-out rows | **22 in all 550**, as required |

No run needs repeating.

---

## 1. The headline

**Nothing in the grid passes the pre-registered bar. Not one of the 72 pair cells.**

The second finding is larger than the first:

| | Validation PR-AUC |
|---|---|
| **Floor** — no molecular description at all | **0.6834** |
| Best single method (chemberta) | 0.7251 |
| Best cell in the whole grid (chemberta + gated pooling) | 0.7446 |

Everything the nine molecular representations contribute, over descriptors, chemistry flags, salt
context, the excipient prior and mechanism flags, is **0.061 of validation PR-AUC**. The spread
across all 110 cells is 0.6237 to 0.7446 — and the bottom of that range is a shuffled control, not
a real method.

With a detectable difference of 0.022 at 5 seeds, most of the distance between methods is noise.

---

## 2. Test 1 — is any molecular description worth anything?

Against the floor, BH-corrected across the 9 tests:

| Method | mean | Δ vs floor | p | p (BH) | Beats floor? |
|---|---|---|---|---|---|
| chemberta | 0.7251 | +0.0417 | 0.087 | 0.195 | no |
| maccs | 0.7203 | +0.0369 | 0.035 | 0.106 | no |
| mol2vec | 0.7076 | +0.0242 | 0.0005 | **0.005** | **yes** |
| pubchemfp | 0.7052 | +0.0219 | 0.203 | 0.305 | no |
| dmpnn | 0.7035 | +0.0201 | 0.011 | **0.049** | **yes** |
| morgan | 0.6989 | +0.0155 | 0.256 | 0.330 | no |
| gat | 0.6880 | +0.0046 | 0.727 | 0.727 | no |
| molformer | 0.6775 | **−0.0059** | 0.500 | 0.562 | no |
| gin | 0.6625 | **−0.0209** | 0.172 | 0.305 | no |

**Only mol2vec and D-MPNN survive correction**, and they do so on small, consistent margins rather
than large ones (mol2vec's sd of the paired difference is 0.0053 — the most reproducible effect in
the grid). ChemBERTa has the largest raw gain of any method and **does not** clear the bar, because
its seed-to-seed variation is eight times mol2vec's.

MoLFormer and GIN are **worse than using no molecular information at all**.

---

## 3. Test 2 — pooling

Settled on 5 seeds instead of the single seed the current assignment came from:

| Method | current | alternative | Δ | p |
|---|---|---|---|---|
| chemberta | cls 0.7251 | **gated 0.7446** | +0.0195 | 0.30 |
| gin | gated 0.6625 | cls 0.6741 | +0.0116 | 0.52 |
| molformer | gated 0.6775 | cls 0.6792 | +0.0017 | 0.83 |
| dmpnn | cls 0.7035 | gated 0.6865 | −0.0170 | 0.30 |
| gat | gated 0.6880 | cls 0.6701 | −0.0180 | 0.11 |

**No method's pooling choice is significant.** Per Part 5 of the plan, where the test separates
nothing the existing pooling stands. Two of the five original single-seed choices point the wrong
way at 5 seeds (chemberta and gin), which is exactly the fragility the grid was built to expose.

`chemberta__pool_gated` at **0.7446 is the highest number in the grid**, and it is not
significantly above `chemberta` at 0.7251 (p = 0.30). It is also not above `maccs` (p = 0.053).

---

## 4. Test 3 — is it the encoder, or the cross-attention?

| Method | with CA | without CA | Δ | p |
|---|---|---|---|---|
| dmpnn | 0.7035 | 0.6770 | +0.0265 | 0.065 |
| gin | 0.6625 | 0.6387 | +0.0238 | 0.30 |
| gat | 0.6880 | 0.6775 | +0.0105 | 0.15 |
| molformer | 0.6775 | 0.6689 | +0.0086 | 0.39 |
| chemberta | 0.7251 | **0.7311** | **−0.0060** | 0.83 |

**No encoder's cross-attention earns its place.** Cross-attention is roughly 380,000 trainable
parameters (two projection heads, two attention blocks, the pooling fusion) and it buys nothing
measurable. For ChemBERTa it is slightly negative: the simpler `chemberta__noca` scores higher.

---

## 5. Test 4 — the fair comparison, fusion held equal

All nine as a single fixed vector, no cross-attention anywhere:

| | Method | mean |
|---|---|---|
| 1 | chemberta (no CA) | **0.7311** |
| 2 | maccs | 0.7203 |
| 3 | mol2vec | 0.7076 |
| 4 | pubchemfp | 0.7052 |
| 5 | morgan | 0.6989 |
| 6 | gat (no CA) | 0.6775 |
| 7 | dmpnn (no CA) | 0.6770 |
| 8 | molformer (no CA) | 0.6689 |
| 9 | gin (no CA) | 0.6387 |

**MACCS — 167 bits, no pretraining, no GPU — comes second**, 0.011 behind ChemBERTa (p = 0.57,
i.e. indistinguishable) and ahead of all four pretrained networks. Three of the five pretrained
encoders rank below every fingerprint.

---

## 6. Test 5 — does a second description help? (the main question)

72 pair cells, each against its own `none`, Benjamini-Hochberg at q = 0.10:

| | |
|---|---|
| Cells with a positive Δ | 32 of 72 |
| Raw p < 0.05 | 11 |
| **Passing BH at q = 0.10** | **0** |

Largest gains, all rejected:

| Cell | mean | Δ vs none | p | p (BH) |
|---|---|---|---|---|
| gin + chemberta | 0.7208 | +0.0583 | 0.014 | 0.260 |
| molformer + chemberta | 0.7287 | +0.0512 | 0.003 | 0.209 |
| gat + chemberta | 0.7307 | +0.0427 | 0.017 | 0.260 |
| gin + mol2vec | 0.7019 | +0.0394 | 0.038 | 0.273 |

The pattern is coherent rather than random: the three biggest gains are all **a weak encoder plus
ChemBERTa's embedding**, and all three exceed the 0.022 power bound. But 11 raw hits out of 72
against an expected 3.6 is not enough separation for any individual cell to survive, and BH's
threshold for the smallest p-value is 0.10/72 = 0.0014 against an observed 0.0029.

Honest statement: **no second description helps by more than about 0.022 validation PR-AUC.**
Not "a second description does not help".

---

## 7. Test 6 — chemistry or parameters? (and why this gate turned out weak)

22 of 72 cells beat **both** shuffled controls at p < 0.05. That sounds like good news and is not,
because of what the controls themselves did:

| Primary | + shuf_maccs | + shuf_morgan |
|---|---|---|
| mol2vec | −0.0736 | −0.0668 |
| chemberta | −0.0197 | −0.0749 |
| maccs | −0.0424 | −0.0526 |
| gat | −0.0059 | −0.0643 |
| pubchemfp | −0.0281 | −0.0676 |
| gin | −0.0267 | −0.0289 |
| dmpnn | −0.0071 | −0.0460 |
| molformer | −0.0148 | −0.0224 |
| morgan | −0.0107 | −0.0040 |

**Every shuffled control hurts, all 18 of them.** The five worst cells in the entire grid are
shuffled controls.

The control was designed to catch a gain that came from extra parameters rather than chemistry.
That failure mode does not exist here: extra parameters do not buy a free gain, they cost. So
"beats the shuffled control" only proves *real information beats noise*, which is a much weaker
claim than *the secondary helps*. The binding gate is test 5, and nothing passes it.

This is a useful negative result about the method, worth stating in the paper: at this dataset size
the 32-wide secondary block is capacity the model cannot exploit, and filling it with noise is
actively harmful.

---

## 8. Tests 7–11

**7. Fingerprint or encoder as the secondary?** Neither. The mean gain is near zero and its sign
tracks how good the primary already was — negative for the four strong primaries (morgan, maccs,
pubchemfp, mol2vec: −0.012 to −0.025) and positive for the weak ones (gin +0.025, gat +0.012). A
second description partially rescues a poor primary and degrades a good one. Fingerprint and
encoder secondaries are indistinguishable from each other.

**8. Does order matter?** Barely. Mean |Δ| over the 36 reversible pairs is 0.0136, below the power
bound; only 2 of 36 reach p < 0.05. The largest is `mol2vec+dmpnn` 0.6776 against `dmpnn+mol2vec`
0.7144 (Δ −0.0368, p = 0.0029). The 64-wide versus 32-wide asymmetry mostly does not matter, which
retires a worry the design had to allow for.

**9. Two-encoder models.** Cross-architecture pairings (transformer + graph, 12 cells) gain +0.0078
on average; same-architecture pairings (8 cells) +0.0033. Both are well inside noise. Two
pretrained encoders in one model is not better than one.

**10. Marginals.** ChemBERTa is the best primary averaged over all columns (0.7133) *and* the best
secondary averaged over all rows (0.7176) — the only representation that is useful in both slots.
PubChem is the second-best secondary (0.7069), ahead of all four encoders. Morgan is the worst
secondary (0.6803). The two-way interaction model could not be fitted: with the diagonal skipped
the design matrix is singular, so only marginal means are reported.

**11. Reproduction.** `morgan + maccs` scores 0.6924 here against **0.6999** in `final_project`
`full_5seeds` (an earlier draft quoted 0.710, which was wrong) — a paired difference of −0.0075,
p = 0.62, **inside** the 0.022 detectable bound. The two code paths
agree. This grid's numbers are canonical for this comparison.

---

## 9. The decision rule, applied

| Rule | Outcome |
|---|---|
| 1. A pair wins only if it beats `none` (BH) **and** both shuffled controls | **0 of 72 pairs** |
| 2. Keep cross-attention only where it beats its own no-CA version | **no method** |
| 3. Best mean among singles, no-CA singles and passing pairs; ties within one seed-SD broken by fewest parameters | best is `chemberta__noca` 0.7311; **7 cells tie** (sd 0.0365); fewest parameters → **maccs** |
| 4. No pair passed → keep the single description and state the bound | **0.022** |
| 5. Unseal held-out only after the choice is locked | **not yet done** |

The tie set rule 3 produced:

| Cell | mean | trainable params |
|---|---|---|
| chemberta__noca | 0.7311 | 136,955 |
| chemberta | 0.7251 | 497,019 |
| **maccs** | **0.7203** | **109,179** |
| mol2vec | 0.7076 | 126,203 |
| pubchemfp | 0.7052 | 200,571 |
| dmpnn | 0.7035 | 1,348,987 |
| morgan | 0.6989 | 218,875 |

**Outcome: a single molecular description, and MACCS by parsimony.** It is statistically
indistinguishable from ChemBERTa (p = 0.57 against `chemberta__noca`) at a quarter of the
parameters, no pretrained weights, no GPU at inference, and no tokenizer.

**ARES's inherited Morgan + MACCS pairing is retired.** It scores 0.6924, below MACCS alone at
0.7203, and the pairing never passed any gate.

### One gap in rule 3, stated rather than quietly fixed

Rule 3 as pre-registered enumerates "the 9 singles, the no-cross-attention singles, and any pair
that passed rule 1". **Pooling variants are not in that list**, so `chemberta__pool_gated` — the
highest cell in the grid at 0.7446 — was never a candidate. The plan assumed pooling would be
settled in block 1 and propagated to every later block; running all four blocks in one sweep meant
that never happened.

It changes nothing, because that cell is not significantly above `chemberta` (p = 0.30) or `maccs`
(p = 0.053), and rule 3's own tie-breaking would still prefer MACCS at 109,179 parameters against
629,627. But the rule should say so explicitly next time, and the number is reported here rather
than omitted because it was ineligible.

---

## 10. What to say in the paper

1. **The excipient prior and the physicochemical features do most of the work.** No molecular
   representation at all reaches 0.6834; the best of 110 configurations reaches 0.7446. Any claim
   about encoder choice sits inside that 0.061.
2. **Only mol2vec and D-MPNN beat no-representation-at-all after correction.** MoLFormer and GIN
   are worse than it.
3. **A second molecular description does not help**, to within 0.022. Adding one to a strong
   primary slightly hurts.
4. **Cross-attention does not pay for itself** for any of the five pretrained encoders.
5. **MACCS keys, at 167 bits, match a pretrained transformer** and beat every graph network.
6. **Noise in the secondary slot is actively harmful** — all 18 shuffled controls hurt, and they
   occupy the bottom of the ranking.
7. Three limitations belong next to these numbers, all in the plan's appendices: the published
   ChemBERTa-77M-MTR checkpoint was pretrained with a character-level tokenizer, so it reads
   chlorine as carbon, bromine as boron and `[Na+]` as nitrogen, and drops charges and
   stereochemistry — 45% of this dataset's molecules are misread, and a tokenizer fix cannot help
   because the model never trained those tokens (Appendix B); GIN and GAT cannot
   distinguish an acid from its conjugate base (Appendix C); and 5 seeds cannot resolve anything
   below 0.022 (Part 7).

---

## 11. Still to do

- **Held-out is sealed.** `collect_grid --reveal-heldout` reads the 22 pairs. Per rule 5 that comes
  after the choice above is locked, and it should be run once.
- **Seeds 47–51** are owed to any cell being called a winner. Since no cell passed, the only
  candidates worth the extra 5 seeds are the three large-but-rejected gains
  (`gin+chemberta`, `molformer+chemberta`, `gat+chemberta`) and `chemberta__pool_gated`. At 10
  seeds the detectable difference falls to 0.013; all four would clear that if their effects are
  real. **20 runs**, under an hour on the DGX.

---

# 12. Held-out, unsealed

Read once, after the decision in section 9 was locked. The 22 pairs are **10 positive, 12
negative**.

## The decision does not change

| Cell | val | val rank | held-out | held-out rank | correct |
|---|---|---|---|---|---|
| **maccs** (chosen) | 0.7203 | 15 | **0.9354** | 20 | **17.6 / 22** |
| chemberta | 0.7251 | 8 | 0.9255 | 29 | 17.6 / 22 |
| chemberta__pool_gated | **0.7446** | **1** | 0.8904 | **80** | 16.4 / 22 |
| chemberta__noca | 0.7311 | 5 | 0.8757 | **98** | 16.2 / 22 |
| mol2vec | 0.7076 | 25 | 0.9605 | 3 | 16.2 / 22 |
| chemberta__x_mol2vec | 0.7229 | 11 | **0.9758** | **1** | 19.4 / 22 |
| floor | 0.6834 | 67 | 0.9101 | 60 | 16.6 / 22 |
| morgan + maccs (retired) | 0.6924 | 52 | 0.8925 | 77 | 16.2 / 22 |

MACCS is 20th of 110 on held-out having been 15th on validation — the most consistent placement of
any cell near the top. Nothing here argues for revising the choice.

## But the two evaluations do not agree, and that is the real result

**Spearman rank correlation between validation and held-out PR-AUC across all 110 cells: ρ = +0.165
(p = 0.085).** Validation does not rank-order held-out.

The cell that won on validation, `chemberta__pool_gated`, is **80th of 110** on held-out.
`chemberta__noca`, 5th on validation, is **98th**. The cell that wins on held-out,
`chemberta__x_mol2vec`, was 11th on validation and was rejected by the decision rule.

## Why the held-out set cannot arbitrate

| | |
|---|---|
| sd of held-out PR-AUC **between** cell means | 0.0312 |
| median sd of held-out PR-AUC **within** a cell, across its 5 seeds | **0.0372** |
| signal-to-noise | **0.84** |

**The seed-to-seed noise inside a single cell is larger than the entire spread between cells.** On
22 pairs one pair flipping is 4.5% of the set, and the chosen model's own 5 seeds span 0.9040 to
0.9809 — a range of 0.077, which is half the full 0.163 spread across all 110 cells.

The held-out ranking is, to a first approximation, noise. It confirms that these models work
(0.81–0.98 PR-AUC, 15–19 of 22 correct) and it cannot choose between them.

## This is what the sealing was for

Had held-out been visible during the grid, the obvious move would have been to pick
`chemberta__x_mol2vec` — first on held-out, 19.4/22 correct, a headline number. It is a cell the
pre-registered rule rejected, on an evaluation whose noise exceeds its signal, chosen after the
fact from 110 candidates. The sealed protocol is the only reason that did not happen.

## What this adds to the paper

8. **Validation and the 22-pair held-out set do not agree on model ranking** (ρ = +0.165). Any
   paper reporting a held-out number for a model selected on validation should say so.
9. **The held-out set is too small to select with.** Within-cell seed noise (0.0372) exceeds
   between-cell spread (0.0312). It is adequate for "does this work" and inadequate for "which of
   these is best".
10. On held-out the floor reaches **0.9101 with 16.6/22 correct** against the best cell's 0.9758
    and 19.4/22. On this test set, no molecular description at all is within about 3 pairs of the
    best of 110 configurations.
