# Primary + secondary featurizer experiments — plan

The 9-family comparison gave each family **one** representation, to separate "which encoder" from
"which fingerprint". The original ARES instead used **two**: Morgan 1024 as primary and MACCS 167
as secondary. This plan tests whether a second representation earns its place, and which pairing is
best. Nothing here is run until the plan is agreed.

## The question

The tower concatenates, per molecule:

```
[ struct 128 (deep encoders only) ] + [ primary -> 64 ] + [ secondary -> 32 ]
+ [ 12 descriptors -> 24 ] + [ 18 chemistry flags -> 16 ] + [ 8 salt context ]   ->  mix -> 96
```

So primary and secondary are **not symmetric**: the primary gets a 64-wide block, the secondary 32.
`morgan+maccs` and `maccs+morgan` are therefore different models, and order is part of the question.

Four questions, in priority order:

1. **Does a second representation help at all**, against the same primary alone?
2. **Which pairing** is best?
3. **Does order matter** (64-wide vs 32-wide block for the same pair)?
4. **Do the deep encoders benefit from a fingerprint alongside** the learned embedding, as the old
   runs did (they all had MACCS or Morgan on top of the embedding)?

## Protocol: unchanged

Everything is identical to the single-representation comparison — same split, PGB head, prior
(train rate 0.0996), mechanism flags, chemistry flags, descriptors, **z-scored salt context**, loss,
optimiser, 40-epoch budget with early stopping, val-tuned threshold, one device. Only the tower's
representation inputs change.

Selection stays on **validation PR-AUC**; the 22 held-out pairs stay sealed in `_heldout_sealed/`
until a choice is made.

## Two stages, because single-seed screening cannot choose

We learned that one seed cannot separate models (seed sd ~0.018-0.04 on val PR-AUC). So stage 1 is a
**screen to discard**, not to choose, and stage 2 is the comparison that counts.

### Stage 1 — screen, seed 42 only (22 runs, about 1.5 h)

| Arm | Combinations | Runs |
|---|---|---|
| A. fingerprint + fingerprint, both orders | ordered pairs of {morgan, maccs, pubchemfp, mol2vec} | 12 |
| B. deep embedding + fingerprint | 5 deep encoders x {morgan, maccs} as the added fingerprint | 10 |

Baselines already exist: all 9 single-representation families on 5 seeds.

Stage 1 output: rank by val PR-AUC, discard anything clearly behind, keep the top candidates.
**No conclusion is drawn from stage 1 beyond "not worth 5 seeds".**

### Stage 2 — confirm, seeds 42-46 (about 25 runs, about 2 h)

The top 4-5 pairs from stage 1, each on 5 seeds. Every one is then compared **paired against its own
primary alone**, using the 5-seed single-representation results already on disk. That paired test is
the actual answer to question 1.

## The control that makes this interpretable

A secondary block adds parameters (`Linear(dim, 32)` plus a wider mix layer), so a gain could be
capacity rather than information. Stage 2 therefore includes, for the winning pair:

- **`<primary>_wide`**: the same single representation with its primary block widened from 64 to 96,
  so the parameter count roughly matches the pair.

If the pair beats the single but not `<primary>_wide`, the gain is capacity, not the second
featurizer — and that is the finding.

## What counts as a result

Given everything so far, the honest expectation is that no pairing separates from the others. The
decision rule, fixed now:

- A pair is preferred over its single only if the **paired t-test over 5 seeds gives p < 0.05** on
  val PR-AUC **and** it also beats the `_wide` control.
- Otherwise the conclusion is "a second representation does not help", which argues for the simpler
  single-representation model and retires the Morgan+MACCS pairing ARES inherited.

## Implementation (small)

1. `towers.py`: relax `ComparisonTower` to allow `struct + fingerprint` and `primary + secondary`,
   keeping the 64 / 32 block widths of the original tower. It currently rejects more than one
   representation on purpose, so this is a deliberate widening of that rule, with the block widths
   asserted to match `src/model.py`.
2. `families.py`: add a pair spec, `{"name": "morgan+maccs", "primary": "morgan", "secondary": "maccs"}`,
   and a `--pairs` list; keep the 9 singles as they are.
3. `data.py`: `OneRepresentationPGBDataset` -> emit a secondary vector when the spec names one
   (the machinery already exists in `PGBDataset._secondary`; it is currently stubbed to `None`).
4. `run_comparison.py`: accept a pair name, name runs `<primary>+<secondary>_seed<N>`.
5. `collect.py`: already groups by name and seed, so it needs no change.
6. Extend `audit_inputs.py` to check the secondary vector for all-zero / NaN, as it does the primary.

No change to the model, trainer, prior or metrics.

## Open questions before I build it

1. **Scope of stage 1.** 22 runs as above, or trim arm B to the two deep encoders that did best
   (chemberta, dmpnn) to save about 40 minutes?
2. **Three representations?** `struct + primary + secondary` is what the tower supports and what
   some old runs effectively had. Worth one or two runs in stage 1, or out of scope?
3. **Does the paper need this at all?** The single-representation result is already clean, and the
   likely outcome here is another null. It is worth running if a reviewer would otherwise ask "why
   did you drop the secondary fingerprint ARES used?" — which I think they would, so I lean yes.
