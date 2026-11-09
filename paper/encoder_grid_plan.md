# The encoder grid — plain explanation and training plan

Replaces the earlier drafts of this file and `featurizer_pair_plan.md`. This is the only document
for the grid. Nothing has been trained from it yet.

---

# PART 1 — What we are trying to find out

Every model predicts: **will this API and this excipient be incompatible?**

To do that it must turn each molecule into numbers. There are many ways to do that, and we do not
know which is best. That is the only question this grid answers.

The head (the part that combines the two molecules and makes the prediction) is **already decided**:
the PGB head. It does not change anywhere in this grid.

Three questions, in plain words:

1. Which way of describing a molecule works best?
2. Does giving the model **two** descriptions of the same molecule beat one?
3. Are the differences we see real, or just luck?

---

# PART 2 — What one model looks like

Every model in the grid has the same three parts. Only **Part A** changes.

```
  API molecule                          Excipient molecule
       |                                        |
  [ PART A: molecule -> numbers ]      [ PART A: same method ]
       |                                        |
  [ PART B: tower -> 96 numbers ]      [ PART B: tower -> 96 numbers ]
       |                                        |
       +----------------+-----------------------+
                        |
            [ PART C: PGB head -> one score ]
```

## Part A — turning a molecule into numbers (THIS is what the grid varies)

Nine methods. Four are fingerprints (a fixed list of numbers computed by rules). Five are
pretrained neural networks (frozen — we never train them, we only read their output).

| Method | Type | What comes out |
|---|---|---|
| morgan | fingerprint | 1024 bits |
| maccs | fingerprint | 167 bits |
| pubchemfp | fingerprint | 881 bits |
| mol2vec | fingerprint (decimals) | 300 numbers |
| molformer | transformer | 768 numbers **per token** |
| chemberta | transformer | 384 numbers **per token** |
| gin | graph network | 300 numbers **per atom** |
| gat | graph network | 300 numbers **per atom** |
| dmpnn | graph network | 2048 numbers **per atom** |

The important difference: a fingerprint gives **one vector for the whole molecule**. The five
networks give **one vector per token/atom**, so there are many vectors per molecule.

Because they give many vectors, the five networks go through extra machinery the fingerprints do
not need:

```
tokens/atoms (768, 384, 300 or 2048 each)
   -> ProjectionHead: Linear(d->256) -> LayerNorm -> GELU -> Dropout(0.15)
                      -> Linear(256->128) -> LayerNorm -> GELU
   -> prepend a CLS vector (the encoder's own pooled output, same projection)
   -> CROSS-ATTENTION (2 x MultiheadAttention, 128 wide, 8 heads, dropout 0.15):
        excipient tokens look at API tokens, and API tokens look at excipient tokens
        each followed by residual + LayerNorm
   -> POOLING, squeeze many vectors into one:
        "cls"   = take the CLS position
        "gated" = learned attention over the tokens, then Linear(256->128)
   -> 128 numbers
```

So cross-attention is the step where the two molecules **look at each other before** the head.
Fingerprint methods have no tokens, so they cannot do this — their two molecules only meet in Part C.
**This is a difference between methods that has nothing to do with the molecule description itself,
and it is one of the things the grid has to separate out.**

## Part B — the tower (identical for every model)

Takes whatever Part A produced, plus features that never change, and makes 96 numbers:

| Input | Size | What happens to it |
|---|---|---|
| structural vector (deep methods only) | 128 | passed through |
| **primary** description | 1024/167/881/300/128 | `Linear(d,64) + LayerNorm + GELU + Dropout(0.15)` |
| **secondary** description (optional) | varies | `Linear(d,32) + LayerNorm + GELU + Dropout(0.075)` |
| chemistry descriptors | 12 | `Linear(12,24) + LayerNorm + GELU` |
| chemistry flags | 18 | `Linear(18,16) + GELU` |
| salt context (z-scored) | 8 | passed through |

All of those are glued together and squeezed: `Linear(total, 96) + LayerNorm + GELU`.

The total going into that squeeze depends on the row:

| Row type | Glued width |
|---|---|
| fingerprint, no secondary | 64 + 24 + 16 + 8 = 112 |
| fingerprint + secondary | 64 + 32 + 24 + 16 + 8 = 144 |
| deep with cross-attention, no secondary | 128 + 24 + 16 + 8 = 176 |
| deep with cross-attention + secondary | 128 + 32 + 24 + 16 + 8 = 208 |
| deep without cross-attention | 64 + 24 + 16 + 8 = 112 |
| no description at all (the floor) | 24 + 16 + 8 = 48 |

There are two towers, one for the API and one for the excipient, with separate weights.

**"Primary" and "secondary" are not equal**: primary gets a 64-wide block, secondary only 32. So
`morgan + maccs` and `maccs + morgan` are genuinely different models. That is why the grid runs
both directions.

## Part C — the PGB head (identical for every model, never changes)

Takes the API's 96 numbers, the excipient's 96 numbers, and some pair-level facts:

```
product  = h_api * h_exc                        (96)
diff     = |h_api - h_exc|                      (96)
bilinear = U(h_api) * V(h_exc)                  (16)   U,V = Linear(96->16), no bias
gate     = sigmoid MLP over [prior(5) + api_flags(18) + exc_flags(18) + mech(5)]
           Linear(46->32) -> GELU -> Linear(32->16) -> Sigmoid
gated    = bilinear * gate                      (16)

glue [h_api, h_exc, product, diff, gated, prior(5), mech(5)] = 410 numbers
  -> Linear(410->128) -> GELU -> Dropout
  -> Linear(128->48)  -> GELU -> Dropout
  -> Linear(48->1)    = residual

prior_weight = Linear(5->8) -> GELU -> Linear(8->1) -> Sigmoid
final score  = residual + prior_weight * log(p/(1-p)) + prior_bias
```

where `p` is how often that excipient was incompatible in the training data.

---

# PART 3 — The grid

**Rows** = the primary description (9 methods).
**Columns** = the secondary description, or none.

Every column goes through the same 32-wide block. A deep encoder used as a *secondary* is squeezed
to one vector per molecule first (average over its tokens/atoms) and gets **no** cross-attention —
it is treated exactly like a fingerprint. We say this in the paper rather than hide it.

## The 12 columns

| Column | What it is |
|---|---|
| none | no second description |
| morgan, maccs, pubchemfp, mol2vec | a second fingerprint |
| molformer, chemberta, gin, gat, dmpnn | a frozen encoder's averaged output, computed once up front |
| shuf_maccs | **control.** Real MACCS vectors, shuffled so each molecule gets someone else's |
| shuf_morgan | **control.** Same trick with Morgan |

### Why the two shuffled controls exist

Adding a secondary adds parameters. More parameters can improve a score on their own, with no new
chemistry. The shuffled control has the **same shape and the same parameter count** but carries no
information about the molecule it is attached to.

So: if `morgan + maccs` beats `morgan`, but `morgan + shuf_maccs` beats it by the same amount, the
gain was the extra parameters — not chemistry. That is the finding, and we would have fooled
ourselves without this control.

Two controls, not one, because the parameter count depends on the input size:

| Secondary size | `Linear(d,32)` parameters |
|---|---|
| maccs 167 | 5,376 |
| mol2vec / gin / gat 300 | 9,632 |
| chemberta 384 | 12,320 |
| molformer 768 | 24,608 |
| pubchemfp 881 | 28,224 |
| morgan 1024 | 32,800 |

A 6x spread. One MACCS-sized control cannot stand in for a Morgan-sized secondary. The two controls
bracket the whole range (167 and 1024), so if neither helps, extra parameters are ruled out
everywhere — without running seven separate shuffles.

The shuffle rules, fixed once: a **derangement** (nobody gets their own vector), the **same**
permutation for train / validation / held-out, and the **same** permutation in every epoch and every
seed. It must be stable, otherwise it is random noise each batch, which tests something else.

## The matrix

`DONE` = already trained. `new` = to train. `--` = same method twice, skipped.

| primary \ secondary | none | morgan | maccs | pubchem | mol2vec | molfmr | chembta | gin | gat | dmpnn | shuf_maccs | shuf_morgan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| morgan | DONE | -- | new | new | new | new | new | new | new | new | new | new |
| maccs | DONE | new | -- | new | new | new | new | new | new | new | new | new |
| pubchemfp | DONE | new | new | -- | new | new | new | new | new | new | new | new |
| mol2vec | DONE | new | new | new | -- | new | new | new | new | new | new | new |
| molformer | DONE | new | new | new | new | -- | new | new | new | new | new | new |
| chemberta | DONE | new | new | new | new | new | -- | new | new | new | new | new |
| gin | DONE | new | new | new | new | new | new | -- | new | new | new | new |
| gat | DONE | new | new | new | new | new | new | new | -- | new | new | new |
| dmpnn | DONE | new | new | new | new | new | new | new | new | -- | new | new |

Plus three extra blocks that are not columns:

- **The floor** (1 model): no molecule description at all. Only descriptors, flags, salt, prior and
  mechanism flags. This tells us what *any* description is worth. We already know the prior term
  alone reaches held-out PR-AUC 0.9525 against the full model's 0.9415, so this number matters a
  lot. Without it the grid only compares methods to each other and can never say whether any of
  them earns its keep.
- **Pooling control** (5 models): each deep method with the *other* pooling (see Part 5).
- **No-cross-attention control** (5 models): each deep method with cross-attention removed, so all
  nine methods can be compared with the fusion held equal.

---

# PART 4 — Exactly what is trained, and what is left

Every cell is trained with **5 seeds: 42, 43, 44, 45, 46**. One cell = 5 runs.

## Already done — 45 runs, nothing to redo

The `none` column, all 9 methods, 5 seeds each.
Location: `experiments/results/encoder_comparison/<method>_seed<N>/`.
These are reused exactly as they are. We do not retrain them.

## Left to train — 101 cells = 505 runs

| # | Block | Cells | Runs | What it answers |
|---|---|---|---|---|
| 1 | Floor (no description) | 1 | 5 | Is any molecule description worth anything? |
| 2 | Pooling control | 5 | 25 | Which pooling for each deep method |
| 3 | Shuffled controls | 18 | 90 | Is a gain chemistry or just parameters? |
| 4 | No-cross-attention control | 5 | 25 | Is a deep method good, or is cross-attention good? |
| 5 | Fingerprint secondaries | 32 | 160 | Does a second fingerprint help? |
| 6 | Embedding secondaries | 40 | 200 | Does a second encoder help? Two-encoder models |
| | **Total** | **101** | **505** | |

Of block 6's 40 cells, 20 are fingerprint-primary + encoder-secondary, and **20 are a cross-attended
encoder plus a different encoder's averaged output** — two pretrained encoders in one model.

## Time and disk

One run is 2.5-5 minutes (measured, from the 45 done runs). 505 runs is roughly **25-30 hours on one
GPU**, or 3-4 hours spread across the DGX.

Disk, if nothing is changed: the current checkpoints save the frozen encoder weights too, so
`molformer_seed42/best_model.pt` is **173 MB** and `dmpnn_seed42` is **39 MB**. At 505 runs that is
about **14 GB** of weights identical to the public ones. Fix this first (Part 6).

---

# PART 5 — What is held fixed, and the one thing that was not

## Held fixed in all 550 runs (45 done + 505 new)

| Item | Value |
|---|---|
| Data | `data/splits/api_cluster` — train 2770, validation 774 |
| Test | the 22 held-out pairs, scored but **sealed** until the decision is made |
| Head | PGB head, unchanged |
| Excipient prior | train positive rate 0.0996 as the global rate, fit on train only, leave-own-cluster-out for train rows |
| Extra features | 18 chemistry flags, 12 descriptors, 8 salt values (both z-scored on train), 5 mechanism flags |
| Loss | asymmetric focal, gamma+ 1.2, gamma- 2.5, pos_weight from train, no balanced sampler |
| Optimiser | AdamW; PGB layers 1.2e-3, encoder-side layers 3e-4; weight decay 2e-3; batch 64; clip 1.0 |
| Budget | max 40 epochs, early stop on validation PR-AUC, patience 7; LR patience 4 |
| Threshold | tuned on validation only: best MCC with TNR >= 0.97 (falls back to the highest-TNR threshold if none reaches the floor) |
| Encoders | all frozen, eval mode, gradients off |
| Seeds | 42, 43, 44, 45, 46 — the **same five** everywhere, which is what makes paired tests valid |

## The thing that was not fixed: pooling

The five deep methods do **not** currently share a pooling:

| Method | Pooling it uses now |
|---|---|
| molformer | gated attention |
| gin | gated attention |
| gat | gated attention |
| chemberta | **CLS** |
| dmpnn | **CLS** |

Where did that come from? The old single-seed `full_grid` runs:

| Method | CLS | gated |
|---|---|---|
| chemberta | **0.763** | 0.740 |
| dmpnn | **0.732** | 0.691 |
| gat | 0.634 | **0.686** |

Each method was given whichever pooling won **on one seed, under a different recipe**. That is
exactly the kind of choice this grid exists to replace. If we leave it alone, all 40
embedding-secondary cells silently inherit it.

So: **block 2 runs each deep method on both poolings, 5 seeds, before anything else.** Five of the
ten combinations already exist (each method's current one), so only 5 cells are new. Whatever wins
per method is then used for that method in every other block. If the test separates nothing, one
pooling is fixed for all five, choosing the one with fewer parameters.

---

# PART 6 — Three code problems to fix BEFORE any training starts

These are real, found by reading the code. Numbers 1 and 3 would corrupt results silently.

### 1. The resume marker is wrong (this one will cost runs)

`run_all_seeds.py:32` decides a run is finished if `val_metrics.json` exists.
But `run_comparison.py` writes **that file first** (line 237) and the held-out metrics and
prediction tables **last** (lines 248-249).

A run killed between those two points — time limit, out-of-memory, preemption, all of which happen
across 505 runs — is marked finished forever and skipped on every resume, **with no held-out
metrics**. We would only find out at collection time, with the GPUs already released, and could not
tell which cells were affected without auditing all 505 folders.

**Fix:** write a `COMPLETE` sentinel file as the very last action and key the skip on that.

### 2. Checkpoints save the frozen encoder

173 MB per MolFormer run, 39 MB per D-MPNN run, about 14 GB over the sweep, all of it weights
identical to the public checkpoints.

**Fix:** save trainable parameters only (plus the small norm-stats and prior-table files). Load with
`strict=False` so old checkpoints still work.

### 3. Continuous secondaries must be z-scored; bit vectors must not

Mol2vec values run about -0.35 to +0.50 and encoder outputs are unbounded, while MACCS and Morgan
are 0 or 1. Feeding an unscaled encoder vector into the 32-wide block is the **same bug** that was
already fixed once in this project, when raw `organic_heavy_atoms` reached 291 while everything
beside it was a 0/1 bit.

**Fix:** z-score continuous secondaries with train-only statistics; leave bit vectors raw; record
per secondary which treatment was used.

### Checked and found already correct

- Threshold tuner has a documented fallback when no threshold reaches TNR >= 0.97.
- The optimiser tolerates an empty encoder parameter group, which is why the fingerprint methods
  already train — so the no-cross-attention runs will not crash on this.
- Encoders are frozen and in eval mode, so the precomputed embeddings will be deterministic.
- A missing excipient gets a zero secondary vector, consistently with its zero primary.
- The held-out externality check is in place.

---

# PART 7 — How the results get judged

Main number: **validation PR-AUC**, mean over the 5 seeds. Held-out stays sealed throughout.

## The tests

| # | Test | Compares | Question in plain words |
|---|---|---|---|
| 1 | Floor | every method vs the no-description floor | Is any description worth anything at all? |
| 2 | Pooling | deep method, gated vs CLS | Which pooling, decided properly |
| 3 | Cross-attention | deep X vs X without it | Is the encoder good, or is cross-attention good? |
| 4 | Fair method comparison | X without cross-attention vs the fingerprints | Which description is best, fusion held equal |
| 5 | Does a secondary help? | X + S vs X + none | The main question |
| 6 | Chemistry or parameters? | X + S vs both shuffled controls | Is the gain real? |
| 7 | Fingerprint or encoder secondary? | mean gain of the 4 fingerprint columns vs the 5 encoder columns | Is a cheap fingerprint as good as a second encoder? |
| 8 | Does order matter? | A + B vs B + A | Because 64-wide is not 32-wide |
| 9 | Two-encoder models | the 20 deep+deep cells, split into cross-architecture (12: transformer+graph) and same-architecture (8) | If two encoders help, is it because they see different things? |
| 10 | Averages | a two-way model over primary, secondary, their interaction, with seed as a random effect | Best primary over all columns, best secondary over all rows, and whether the best secondary depends on the primary |
| 11 | Reproduction | morgan + maccs here vs `final_project` `full_5seeds` (0.710) | Do the two code paths agree? If not, this grid's numbers are canonical |

Test 7 note: the column counts are uneven because the diagonal is skipped. A fingerprint row has 3
fingerprint columns and 5 encoder columns; a deep row has 4 and 4. Report per-row means with their
counts rather than one pooled number.

## Two honesty requirements

**1. Five seeds cannot see small differences.** From the only existing pair-vs-single comparison on
shared seeds (`design_b` vs `ARES full`, `paper/designb_vs_nine.csv`), the per-seed differences are
`-0.0094, +0.0073, -0.0159, +0.0168, -0.0045`: mean -0.0011, standard deviation **0.0132**.

At 5 seeds and 80% power that only detects a difference of **0.022 or larger**. So if nothing
passes, the honest sentence is *"no second description helps by more than about 0.022 validation
PR-AUC"* — **not** *"a second description does not help"*. Any cell that looks like a winner gets
**5 more seeds (47-51)**, which brings the detectable difference down to about **0.013**.

These two numbers were 0.026 and 0.018 in the first draft of this plan, from a rougher
calculation. `collect_grid.detectable_delta` now solves the paired-t power equation exactly against
the noncentral t distribution and prints the bound with every report, so the number quoted in the
paper is the one the code computed rather than one copied from here. The normal approximation,
which gives 0.017 at 5 seeds, is optimistic by a third at 4 degrees of freedom and is not used.

**2. Seventy-two tests produce false winners by chance.** With 72 pair cells tested at p < 0.05,
about 3.6 will pass by luck alone. The "vs none" tests are therefore reported with
**Benjamini-Hochberg FDR at q = 0.10** across all 72, and the decision rule below uses the adjusted
p-value. Having to beat both shuffled controls as well is a second, independent filter, but it does
not replace this.

## The decision rule, fixed now, before any run

1. A cell X + S beats X only if it wins on validation PR-AUC against X + none **and** against
   **both** shuffled controls, all at adjusted p < 0.05 over the seeds.
2. Cross-attention is kept for a deep method only if it beats its own no-cross-attention version at
   p < 0.05. Otherwise the simpler version goes forward.
3. The final model is the best mean validation PR-AUC among: the 9 singles, the no-cross-attention
   singles, and any pair that passed rule 1. Cells within one seed-standard-deviation of the best
   count as tied, and among ties the one with fewest trainable parameters wins.
4. If no pair passes, we keep the single-description model and retire the Morgan+MACCS pairing ARES
   inherited — stating the 0.022 bound from above.
5. **Only after the choice is locked** do we unseal the 22 held-out pairs.

---

# PART 8 — Order of work

| Step | What | Runs | Why this order |
|---|---|---|---|
| 0 | Fix the three code problems in Part 6. Precompute and audit the 5 encoders' averaged vectors. Build the shuffle permutations. Smoke-test 1 epoch of each new kind of model. | 0 | Nothing long starts on a broken resume marker |
| 1 | Floor + pooling control | 30 | Pooling must be settled before any deep row runs |
| 2 | Shuffled + no-cross-attention controls | 115 | These decide how every later number is read |
| 3 | Fingerprint secondaries | 160 | |
| 4 | Encoder secondaries | 200 | |
| 5 | Collect, run the 11 tests, apply the decision rule, then unseal held-out | 0 | |

Stopping after step 3 still gives a complete, publishable answer with all controls in place. Step 4
widens it to the full 9x9.

## Files — as built

All under `experiments/encoder_comparison/`.

```
reps.py                    NEW. One vector per molecule for all 12 columns: the 2 computed
                           fingerprints, the 2 table fingerprints, the 5 mean-pooled encoders, and
                           the 2 shuffled controls. Owns the bits-raw / continuous-z-scored rule
                           and the fixed derangement.
precompute_embeddings.py   NEW. The 5 encoders' mean-pooled vectors, computed once with a built-in
                           audit that refuses to write a bad table.
towers.py                  CHANGED. Optional secondary (32-wide) and optional primary, so the floor
                           cell can have no description at all.
families.py                CHANGED. The 12 columns, the Cell type, the 6 block generators, the
                           101-cell grid.
data.py                    CHANGED. GridPGBDataset reads this cell's primary / secondary from
                           reps.py. OneRepresentationPGBDataset is kept FROZEN for the 45 done runs.
run_grid.py                NEW. One cell at one seed. COMPLETE sentinel, slim checkpoints, path
                           assertions, a first-batch shape check.
run_all_grid.py            NEW. The 101 cells x 5 seeds, resumable on the sentinel, --shard I/N
                           for several GPUs.
audit_reps.py              NEW. Every representation for every molecule: coverage, finiteness,
                           scaling, derangement, encoder collisions. Blocking problems are kept
                           separate from expected properties.
collect_grid.py            NEW. The matrix, the 11 tests, BH correction, the exact power bound and
                           the decision rule.
run_comparison.py          CHANGED only to keep working: ComparisonTower is now called with
                           keywords, so the 9 families keep getting secondary_dim=0.
run_all_seeds.py           CHANGED only in is_done(), which now requires the LAST files of a run
                           rather than the first.
src/train.py               CHANGED. config.pgb_save_trainable_only skips the frozen encoder in the
                           checkpoint and loads with strict=False, refusing to do so if any
                           encoder parameter requires grad.
```

Deviation from the draft above: the grid runs through **new** `run_grid.py` / `run_all_grid.py`
instead of rewriting `run_comparison.py` / `run_all_seeds.py`. Those two scripts are the record of
how the 45 completed runs were produced; overwriting them destroys that provenance and risks
silently changing the baseline the whole grid is measured against. Both were therefore changed only
where they would otherwise break or mislead. Verified after the change: the four fingerprint
families rebuild to **exactly** the recorded trainable parameter counts (218,875 / 109,179 /
200,571 / 126,203) and all 45 runs still register as complete.

## Smoke tests before step 1 (1 epoch each, results discarded)

| Run | Checks |
|---|---|
| `morgan+maccs` | fingerprint + fingerprint secondary |
| `chemberta+morgan` | deep primary + fingerprint secondary |
| `maccs+gat` | fingerprint primary + encoder secondary |
| `chemberta+gat` | two-encoder model |
| `gat+shuf_maccs` | shuffle table wired correctly |
| `dmpnn` no cross-attention | the fixed-vector path for a deep encoder |
| floor | the zero-description tower builds at all |

---

# APPENDIX A — Why every method's adapter ends at the same width

A fair question about the architecture: the five encoders output vectors of very different widths
(300, 300, 384, 768, 2048) and all of them are pushed through the same `Linear(d -> 256) -> 256 -> 128`
adapter. Should the adapter be sized per encoder instead?

**No. Sizing the adapter per encoder is the confound this grid exists to remove.**

## The rule, stated once

| Layer | Width | Varies? |
|---|---|---|
| encoder output `d` | 300 / 384 / 768 / 2048 | **yes — fixed by the pretrained checkpoint, not chosen by us** |
| adapter hidden | 256 | no |
| adapter output / structural vector | 128 | no |
| primary block output | 64 | no |
| secondary block output | 32 | no |
| tower output into PGB | 96 | no |
| PGB head | unchanged | no |

Only the **input** width varies, because `d` is a property of the released checkpoint. Every width
from the adapter output onwards is identical for all 9 methods. The PGB head never sees `d` at all:
it receives a 96-wide vector per molecule whatever the method was.

## Why per-encoder sizing would break the comparison

If D-MPNN's adapter were widened to 512 because its input is 2048, and GIN's left at 256, then the
D-MPNN row has a larger trainable tower than the GIN row. A win for D-MPNN could then mean either
"CheMeleon is a better representation" or "we gave D-MPNN a bigger network". The grid could not
separate the two. The whole point of the fixed 128 interface is that the only thing differing
between two rows is the molecular description.

## The residual asymmetry, stated honestly

The concern is not baseless. The adapter's first layer has `d x 256 + 256` parameters, so it does
scale with `d`:

| Method | `d` | First-layer params | Compression at that layer | Mean val PR-AUC (5 seeds) |
|---|---|---|---|---|
| gin | 300 | 77,056 | 1.17 : 1 | 0.6625 |
| gat | 300 | 77,056 | 1.17 : 1 | 0.6880 |
| chemberta | 384 | 98,560 | 1.50 : 1 | **0.7251** |
| molformer | 768 | 196,864 | 3.00 : 1 | 0.6775 |
| dmpnn | 2048 | 524,544 | 8.00 : 1 | 0.7188 |

That is a 6.8x spread in adapter parameters and a 1.17:1 to 8:1 spread in how much the layer must
discard. **There is no way to hold both the parameter count and the architecture equal when the
inputs have different widths — something has to vary.** We fix the architecture and let the adapter
width float, because the 128-wide interface is what every later layer sees, and because `d` is not
ours to change.

The same applies to the fingerprint blocks, `Linear(d -> 64)` for primary and `Linear(d -> 32)` for
secondary:

| Method | `d` | Primary params | Secondary params | Mean val PR-AUC |
|---|---|---|---|---|
| maccs | 167 | 10,752 | 5,376 | **0.7203** |
| mol2vec | 300 | 19,264 | 9,632 | 0.7076 |
| pubchemfp | 881 | 56,448 | 28,224 | 0.7052 |
| morgan | 1024 | 65,600 | 32,800 | 0.6989 |

Weak evidence that adapter capacity is not what drives the ranking: the best encoder is ChemBERTa,
with the second-smallest adapter, not D-MPNN with the largest; and among the fingerprints the trend
runs the **other** way, with the smallest input (MACCS, 167 bits) scoring highest and the largest
(Morgan, 1024) lowest. With four or five points this cannot establish anything, which is why the
parameter count is reported rather than argued away.

## What this changes in the plan

1. **No architecture change.** The 45 completed runs used this adapter; altering the 256 hidden layer
   would invalidate all of them and buy nothing — a plain `Linear(d -> 128)` adapter would still
   scale with `d`, so it does not solve the asymmetry, it only shrinks it for everyone.
2. **Report it.** `collect_grid.py` adds a trainable-parameter column per cell, so any reader can
   check a winner against its parameter count. This is the same reason the shuffled controls exist.
3. **Tie-breaking already accounts for it.** Decision rule 3 in Part 7 breaks ties by fewest
   trainable parameters, so a method cannot win on capacity alone.

---

# APPENDIX B — Found while building the code: ChemBERTa is not reading SMILES

`audit_reps.py` flagged ChemBERTa for giving two different molecules the same vector. Chasing it
down produced a finding that is **not** about the grid, and that is bigger than the grid.

## What the tokenizer does

The local snapshot at `models/pretrained/chemberta` loads as a `RobertaTokenizer`, which applies
byte-level BPE to the SMILES string. The vocabulary is correct and contains `[Na+]`, `[Cl-]`,
`[C@H]`, `[O-]` as single tokens. **The tokenizer never produces them.** It splits bracketed atoms
into characters and then drops whatever is not in the vocabulary on its own:

| SMILES | What ChemBERTa receives |
|---|---|
| `[Na+].[Cl-]` | `N . C -` — sodium becomes **nitrogen**, chloride becomes **carbon** |
| `CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]` | `... O - . N` — SDS's sodium becomes nitrogen |
| `C([C@H](O)CO)O` | `C ( C ( O ) C O ) O` — **stereochemistry silently gone** |
| `[Ag+]`, `[Ag]`, `[Al]`, `[Hg]`, `[K]`, `[Li]` | **nothing at all** — only `[CLS] [SEP]` |

Six molecules in this dataset are bare metal ions, and to ChemBERTa they are the empty string, so
all six get one identical embedding. Silver nitrate and potassium nitrate also collide. This is the
whole of the 5-collision warning.

Verified with `tokenizer.json` loaded directly through `PreTrainedTokenizerFast`, so it is the
saved snapshot that is wrong, not the class it is loaded with.

**MoLFormer is correct** — it returns `[Na+]`, `[Cl-]`, `[C@H]`, `[O-]` as proper single tokens.
This is specific to ChemBERTa.

## Why it matters beyond the grid

ChemBERTa is the current **leader** on validation PR-AUC (0.7251, mean of 5 seeds), and that number
was earned while reading sodium as nitrogen and ignoring every stereocentre. For an
API–excipient compatibility task where counterions and salt forms are a large part of the chemistry,
that is not a small defect.

Two readings, both worth stating:

1. The number is real but the label is wrong. Whatever ChemBERTa is contributing, it is not
   coming from the chemistry the SMILES encodes. It is closer to a coarse atom-count /
   heavy-atom-skeleton feature than to a pretrained chemical language model.
2. Fixed, it might do better. The vocabulary is right there; only the tokenization is broken.

## What is NOT being done about it here

Replacing the tokenizer changes ChemBERTa's input completely, which **invalidates the 45 completed
`none`-column runs** for that family and makes the old and new chemberta numbers incomparable. That
is a decision about the project's baseline, not a code fix to slip into a grid runner, so nothing
has been changed. The options, in order of cost:

| Option | Cost | What it buys |
|---|---|---|
| Report it as a limitation, change nothing | 0 runs | Honest, and the grid stays internally consistent |
| Fix the tokenizer, rerun chemberta's `none` column | 5 runs | A correct chemberta single-method number |
| Fix it and treat `chemberta_fixed` as a 10th method | 5 + its grid cells | Both numbers, and the difference is itself a result |

Until this is decided, every chemberta cell in the grid uses the tokenizer as it is, so chemberta
rows are at least consistent with each other and with the 45 completed runs.

---

# APPENDIX C — What the encoders cannot see (from the pre-training audit)

`audit_reps.py` classifies every case where two molecules receive a byte-identical vector. None of
these wastes a run. All of them bound what a column can possibly resolve, and two are squarely
about the chemistry this project is predicting.

## 1. The graph encoders cannot tell an acid from its salt

GIN and GAT give **identical** embeddings to each of these pairs:

| | |
|---|---|
| `B(O)(O)O` / `B([O-])([O-])[O-]` | boric acid / borate |
| `OS(=O)(=O)O` / `[O-]S(=O)(=O)[O-]` | sulfuric acid / sulfate |
| citric acid / citrate | 7 such pairs for GIN, 5 for GAT |
| tartaric acid / tartrate, maleic acid / maleate | |
| `[Fe+2]` / `[Fe+3]`, `[Cu+2]` / `[Cu]`, `[Ag+]` / `[Ag]` | oxidation state invisible |

Confirmed by construction: neutralising every formal charge makes each pair identical, so the
collision is *entirely* the charge state. `PretrainAtomFeaturizer` does not carry formal charge
into the representation these two encoders were pretrained with.

For API–excipient compatibility this is not a detail. Free acid versus salt form is one of the
distinctions the task is about. Any result for GIN or GAT is a result obtained while blind to it.

## 2. ChemBERTa additionally cannot see stereochemistry or metals

Already in Appendix B from the tokenizer side; the audit confirms it from the output side. Sorbitol
and its stereoisomer collide; levetiracetam and its racemate collide; `[Ag+]`, `[Ag]`, `[Al]`,
`[Hg]`, `[K]` and `[Li]` all share **one** vector; and silver nitrate collides with potassium
nitrate, because the counterion token is dropped and only the nitrate survives.

MoLFormer and D-MPNN have **no** unexplained collisions — MoLFormer separates all 861 identities.

## 3. A data-hygiene issue in the source CSVs

Six molecules are spelled twice, differing only by a leading or trailing space:
` C1=CC=C(C=C1)O` and `C1=CC=C(C=C1)O`, `...[Na]` and `...[Na] `. RDKit parses both the same way,
so no model is misled, but the dataset has 867 SMILES strings for 861 distinct chemical identities.
This is why the shuffled controls derange over canonical identity rather than over strings.

## 4. Resolution limits of the fingerprints

Expected, reported for completeness: MACCS collides 28 groups of genuinely different molecules
(167 bits for all of chemistry), PubChem 16, Morgan 7, Mol2vec 3. PubChem also has **235 of 881
bits** that are never set on this dataset, costing 7,520 secondary parameters that cannot carry
information; MACCS has 5 such bits. D-MPNN has 102 of 2048 channels that never vary.

## Why none of this blocks the sweep

The audit fails only on what would make a run's numbers **wrong**: a missing vector, a non-finite
value, an all-zero vector, a wrong width, scaling not applied, or a broken derangement. A collision
makes a column less informative, not incorrect, and every one above has a named cause. Calibrating
it the other way — which the first two versions of this script did — produced an audit that
reported 14 "problems" nobody could act on, immediately before a 25-hour sweep. That is worse than
no audit, because it trains you to pass `--force`.
