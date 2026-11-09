# Encoder comparison — 9 families, one fixed protocol

Chooses the encoder family for ARES on **validation only**, so the 22 held-out pairs stay an
untouched test set. Protocol: `paper/encoder_comparison_plan.md`.

Nothing here changes the shipped ARES model or any existing result. Output goes to
`experiments/results/encoder_comparison/<family>_seed<N>/` and
`checkpoints/encoder_comparison/<family>_seed<N>/`.

## Commands

Run from the project root (`versions_of_api-exc/project`).

**1. Check all 9 families build and run (no training, writes nothing):**

```bash
python -m experiments.encoder_comparison.smoke_check
```

**2. Run one family first, to see the time per run:**

```bash
python -m experiments.encoder_comparison.run_comparison --family morgan --device cuda
```

**3. Run all 9 (use the same device for every run):**

```bash
python -m experiments.encoder_comparison.run_comparison --family all --device cuda
```

**4. Run all 9 families on seeds 42-46 (45 runs; seed 42 is already done, so 36 remain):**

```bash
python -m experiments.encoder_comparison.run_all_seeds
```

Each run is a separate process, finished runs are skipped so it resumes after an interruption, and
logs go to `experiments/results/encoder_comparison/_logs/`. Add `--dry-run` to see what it would do,
or `--seeds 43 44` / `--families morgan gat` to narrow it.

**5. Build the selection table (held-out stays sealed):**

```bash
python -m experiments.encoder_comparison.collect
```

**6. Only after the family is chosen, reveal the held-out column:**

```bash
python -m experiments.encoder_comparison.collect --reveal-heldout
```

Useful flags: `--seed N` (default 42), `--epochs N` (default 40, same for all),
`--overwrite` to rerun a family, `--data-root <folder>` to read the split from somewhere other
than `final_project/data`, `--family <name>` for one of
`morgan, maccs, pubchemfp, mol2vec, molformer, chemberta, gin, gat, dmpnn`.

## What is held fixed

| Item | Value |
|---|---|
| Data | ARES `api_cluster` split: train 2770, val 774 |
| Test | the 22 held-out pairs, scored but sealed until the choice is made |
| Head | PGB head, unchanged (rank-16 bilinear, gate, prior-trust MLP) |
| Prior | excipient prior with the **train positive rate** (0.0996) as its global rate, fit on train only, leave-own-cluster-out for train rows |
| Shared features | 18 chemistry flags, 12 descriptors and 8 salt-context values (both z-scored with train-only statistics), 5 mechanism flags |
| Loss | asymmetric focal, γ+ 1.2 / γ− 2.5, pos_weight from train, no balanced sampler |
| Optimiser | AdamW, PGB lr 1.2e-3, encoder-side lr 3e-4, wd 2e-3, batch 64, clip 1.0 |
| Budget | max 40 epochs, early stopping on val PR-AUC (patience 7), LR patience 4 |
| Threshold | tuned on val only: best MCC with TNR ≥ 0.97 |
| Seed | 42 for a single sweep; 42-46 for the multi-seed comparison |

## One representation per family

This is what the earlier runs got wrong: every family also received a secondary fingerprint, and
which one varied by family, so "encoder" and "fingerprint" were confounded.

| Family | Representation | Earlier runs also added |
|---|---|---|
| morgan | Morgan 1024 | MACCS 167 |
| maccs | MACCS 167 | Morgan 512 |
| pubchemfp | PubChem 881 | Morgan 512 |
| mol2vec | Mol2vec 300 | Morgan 512 |
| molformer | MoLFormer-XL embedding | MACCS 167 |
| chemberta | ChemBERTa embedding | Morgan 512 |
| gin | GIN embedding | MACCS 167 |
| gat | GAT embedding | MACCS 167 |
| dmpnn | D-MPNN embedding | MACCS 167 |

All five pretrained encoders are **frozen** feature extractors; only the PGB layers (and, for the
deep families, the projection / cross-attention / pooling layers) are trained.

## Selection rule, fixed before any run

1. highest validation PR-AUC
2. tie-break: validation MCC
3. tie-break: fewer trainable parameters

Families within 0.03 val PR-AUC of the best are reported as **not separable at one fold** —
`collect.py` prints this. With one fold and one seed a small gap is noise, so if the top of the
table is crowded the honest follow-up is 5 seeds or grouped 5-fold CV on this same protocol.

## Two asymmetries to report, not hide

- **Fusion.** Fingerprint families use concat; sequence / graph families use cross-attention with
  their own pooling. That is a property of the family, not a free choice.
- **Trainable parameters.** The deep families carry 4–12× more trainable parameters (projection,
  cross-attention, pooling): roughly 110k–220k for fingerprints vs 500k–1.35M for the deep ones.
  `collect.py` prints the count for every family.

## Input audit

```bash
python -m experiments.encoder_comparison.audit_inputs
```

Pushes all 879 molecules through every representation and reports silent failures: unparseable
SMILES, all-zero fingerprints or embeddings, NaN/inf, tokenizer truncation, zero-variance
descriptor columns, and encoder weights that are not frozen. Current result:

| Check | Result |
|---|---|
| RDKit parses every molecule | pass (879/879) |
| Morgan / MACCS all-zero | none |
| Descriptors, salt context finite | pass |
| Descriptor zero variance on train | none |
| PubChem / Mol2vec vectors present and non-zero | pass, after the CID join (see below) |
| All-zero or NaN embedding, any deep encoder | none |
| GIN / GAT / D-MPNN keep every atom | pass |
| All five pretrained encoders frozen | pass |
| **MolFormer / ChemBERTa truncation** | **1 molecule of 879 cut from 526 to 512 tokens** |
| **Excipients unseen by the prior** | **20 of 774 val rows, 2 of 22 held-out rows** |

The two findings in bold are real but benign, and both are reportable rather than fixable here:

- The truncated molecule is a 524-character peptide that occurs in **3 train pairs only** — not in
  val, not in held-out. So it cannot affect the validation ranking or the test score; it only means
  the two transformer families saw ~97% of that one training molecule. The graph encoders see all
  291 of its atoms, so this is a small asymmetry between families, not a bug.
- The unseen excipients are by design: the prior falls back to the global (train) rate and sets its
  `unseen` flag to 1, which the model receives as an explicit input. It is identical for all nine
  families, so it does not bias the comparison.

## Files

```
families.py        the 9 specs (one representation each)
towers.py          molecule tower that takes exactly one representation, no secondary
data.py            split paths, held-out externality check, CID-joined vector lookup, dataset
run_comparison.py  train + evaluate one family or all
collect.py         summary table + selection rule
run_all_seeds.py   drives every family x seed, resumable, one subprocess per run
smoke_check.py     build-and-forward check, no training
audit_inputs.py    every molecule through every representation, hunting silent failures
```

Two shared-code fixes went in with this comparison and apply to all nine families:

- The excipient prior (`src/pgb_prior.py`) takes the **train positive rate** (0.0996) as its global
  rate instead of the hard-coded 0.094 it used before.
- The 8 salt-context values are **z-scored with train statistics** (`config.pgb_normalize_salt`).
  Before, `organic_heavy_atoms` went into the mix layer raw, reaching 291 while every other input to
  that layer was a 0/1 bit or already at about unit scale. Checkpoints trained before this store no
  salt statistics and keep the old behaviour when loaded, so earlier results are unchanged.

The PubChem and Mol2vec tables are joined on **CID** and re-keyed on our SMILES: 13 molecules are
written differently in those files (tautomer / Kekulé / stereo notation), and a plain SMILES
lookup would silently hand the model all-zero vectors for them.
