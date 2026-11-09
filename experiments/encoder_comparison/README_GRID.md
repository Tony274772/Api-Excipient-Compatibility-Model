# The 9x9 encoder grid — how to run it

Plan and reasoning: `paper/encoder_grid_plan.md`. This file is only the operating instructions.

One model = a **primary** molecular description (64-wide block) + an optional **secondary**
description (32-wide block) + the fixed tower + the fixed PGB head. Rows are the 9 primaries,
columns are the 12 secondaries. Nothing else varies.

**Already trained: 45 runs** — the `none` column, 9 methods x 5 seeds, in
`experiments/results/encoder_comparison/`. Reused as they are, never retrained.
**Left: 101 cells = 505 runs**, written to `experiments/results/encoder_grid/`.

---

## Order of work

### Step 0 — once, before anything long

```bash
python -m experiments.encoder_comparison.precompute_embeddings
```

Runs each of the 5 frozen encoders over all 867 molecules and writes its mean-pooled vector to
`experiments/results/encoder_grid/_embeddings/<name>_mean.csv`. Needed by the 40 embedding-secondary
cells and the 5 no-cross-attention cells. It audits its own output and refuses to write a table
with non-finite or all-zero rows. Deterministic — the encoders are frozen and in eval mode.

```bash
python -m experiments.encoder_comparison.audit_reps
```

Every representation for every molecule in every split: coverage, finiteness, all-zero rows,
scaling, the derangement, encoder collisions. Exits non-zero on a **blocking problem**. It also
prints **observations**, which are true properties and not defects — a 167-bit fingerprint
collides on 861 molecules by construction, and PubChem has 235 bits that are never set here.
Read those; do not try to fix them.

### Steps 1–4 — the sweep, in this order

```bash
python -m experiments.encoder_comparison.run_all_grid --dry-run
```

```bash
python -m experiments.encoder_comparison.run_all_grid --blocks floor pooling
```

```bash
python -m experiments.encoder_comparison.run_all_grid --blocks shuffled no_cross_attn
```

```bash
python -m experiments.encoder_comparison.run_all_grid --blocks fp_secondary
```

```bash
python -m experiments.encoder_comparison.run_all_grid --blocks emb_secondary
```

| Step | Blocks | Runs | Why this order |
|---|---|---|---|
| 1 | floor, pooling | 30 | Pooling must be settled before any deep row is believed |
| 2 | shuffled, no_cross_attn | 115 | These decide how every later number is read |
| 3 | fp_secondary | 160 | |
| 4 | emb_secondary | 200 | |

Stopping after step 3 still gives a complete, publishable answer with every control in place.
Step 4 widens it to the full 9x9 and adds the 20 two-encoder models.

### Across several GPUs

```bash
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python -m experiments.encoder_comparison.run_all_grid --shard $i/4 & done
```

Every process builds the identical queue and takes only the runs whose index mod N is its own, so
no two processes ever write the same folder.

### Step 5 — collect and decide

```bash
python -m experiments.encoder_comparison.collect_grid --csv-out paper/encoder_grid.csv --matrix-out paper/encoder_grid_matrix.csv
```

Prints coverage, the matrix, the exact power bound, the 11 tests, and the decision rule. Held-out
stays sealed. Only once the decision is locked:

```bash
python -m experiments.encoder_comparison.collect_grid --reveal-heldout
```

---

## Resume, and why the sentinel exists

A run counts as finished **only** when `experiments/results/encoder_grid/<cell>_seed<N>/COMPLETE`
exists. That file is written last, after every metric, prediction table and sealed held-out file is
on disk, and the runner verifies each of them first.

The old marker was `val_metrics.json`, which `run_comparison.py` writes **first**. A run killed
between that write and the sealed held-out files counted as done forever, with no held-out metrics,
and nothing would have revealed it until collection time with the GPUs already released. Over 505
runs, something gets killed.

So: `run_all_grid` is safe to stop with Ctrl-C and restart. It redoes anything without a sentinel,
and it treats "exit 0 but no sentinel" as a failure.

---

## Single runs, for debugging

```bash
python -m experiments.encoder_comparison.run_grid --list
```

```bash
python -m experiments.encoder_comparison.run_grid --cell morgan__x_maccs --seed 42 --epochs 1 --device cpu
```

Cell names: `<primary>`, `<primary>__x_<secondary>`, `<primary>__noca`,
`<primary>__pool_gated` / `__pool_cls`, and `floor`.

---

## The seven smoke tests, all passing

One epoch each, results discarded. Re-run these after touching `reps.py`, `towers.py` or `data.py`.

| Cell | Path it exercises | Result |
|---|---|---|
| `floor` | the zero-description tower builds at all | ok, 75,131 params |
| `morgan__x_maccs` | fingerprint + fingerprint secondary | ok, 235,899 params |
| `chemberta__x_maccs` | deep primary + fingerprint secondary | ok |
| `maccs__x_chemberta` | fingerprint primary + encoder secondary | ok, 140,091 params |
| `mol2vec__x_shuf_morgan` | shuffle table and continuous-primary scaling | ok |
| `chemberta__noca` | the fixed-vector path for a deep encoder | ok, 136,955 params |
| `chemberta__pool_gated` | the pooling control | ok |

The eighth, a two-encoder cell such as `chemberta__x_gat`, needs `precompute_embeddings` to have
run for both encoders.

---

## What is enforced in code, not just intended

- **Scaling.** Bit vectors are left raw; continuous vectors (mol2vec and all 5 encoder outputs) are
  z-scored with **train-only** statistics, saved per run to `primary_scaler.npz` /
  `secondary_scaler.npz`. Encoder outputs run to ±6, so feeding them raw into a 32-wide block next
  to 0/1 bits is the bug this project already fixed once for `organic_heavy_atoms`.
- **The shuffled controls.** A derangement over **chemical identity**, not over SMILES strings —
  the dataset spells 6 molecules two ways, and a string-level derangement could hand a molecule
  back its own vector through its other spelling, quietly putting real information into a control
  that must carry none. Seeded by a fixed constant, not the run seed, so all 5 seeds of a control
  see the identical mapping; the donor map's sha1 goes into every `run_info.json`.
- **No silent loss of an encoder.** `src/model.py:_encode_struct` returns `(None, None)` when
  `use_cross_attn` is false, discarding the encoder's output with no error. `run_grid.build_model`
  asserts that a deep cell is on the cross-attention path and that every other cell is not. The
  no-cross-attention control therefore runs its embedding through the **primary** slot, which is
  the only way to keep it.
- **Shape agreement.** The first validation batch is checked against the cell's declared widths
  before training starts, so a dataset/tower mismatch fails in seconds rather than after 40 epochs.
- **Slim checkpoints.** `config.pgb_save_trainable_only` omits everything under `encoder.`, and
  refuses to if any encoder parameter requires grad. ChemBERTa's checkpoint drops 16 MB to 2.5 MB;
  MolFormer's 173 MB to about the same. Roughly 1.2 GB over the sweep instead of 14 GB.
- **Backward compatibility.** The 4 fingerprint families rebuild to exactly their recorded
  trainable parameter counts (218,875 / 109,179 / 200,571 / 126,203), and all 45 completed runs
  still register as done.

---

## Known, deliberate, not bugs

- **ChemBERTa's tokenizer is broken** and reads `[Na+]` as nitrogen, drops stereochemistry, and
  maps bare metal ions to the empty string. See Appendix B of the plan. Nothing has been changed,
  because fixing it invalidates the 45 completed runs for the current leading method — that is a
  decision, not a patch.
- **PubChem has 235 dead bits** on this dataset, costing 7,520 secondary parameters that cannot
  carry information. Reported, not fixed: removing them would make the column non-standard.
- **Fingerprint collisions** (51 molecule pairs for MACCS, 29 for PubChem, 16 for Morgan) are what
  a fixed-width fingerprint does at 861 molecules. They cap what those columns can resolve and are
  reported for that reason.

---

## Verification status — what was actually tested, and what was not

Written down because "I checked it" is not a verification record.

### Verified by running it

| Check | Result |
|---|---|
| All 9 smoke-test paths train | floor, fp+fp, deep+fp, fp+encoder, encoder+encoder **both directions**, shuffle, transformer no-CA, graph no-CA, pooling control |
| The 45 completed runs are unaffected | 4 fingerprint families rebuild to exactly 218,875 / 109,179 / 200,571 / 126,203 trainable params; all 45 still register as done |
| Slim checkpoint restores exactly | 55 encoder entries dropped, 75 kept; predictions after restore differ from before by **0.0** |
| A killed run is retried | A folder with `val_metrics.json` and no sentinel is re-queued (under the old logic it was skipped forever) |
| Reruns are deterministic | Same cell twice: 0.6057 / 0.6057, identical threshold and MCC |
| Paired t-test | Matches `scipy.stats.ttest_rel` to 1e-12 |
| BH correction | Adjusted p-values **and** rejections match `statsmodels.multipletests(method='fdr_bh')` exactly |
| Power bound | Solves to exactly 0.80 achieved power; 0.0222 at 5 seeds, 0.0131 at 10 |
| The control gate works | A synthetic cell beating `none` at p≈0 but matching its shuffled controls is correctly **rejected** by rule 1 |
| Z-scoring is applied where it should be | Train mean 0.000 / sd 1.000; held-out at sd 1.069 and 0.875, i.e. scaled by train statistics and not its own |
| Scaling actually matters | Raw GAT output runs **−144 to +216**; mol2vec −0.35 to +0.50; MACCS 0/1 |

### Bugs found and fixed during this pass

- **A multi-GPU race.** Every run rebuilt `_lookup/<fp>_lookup.csv` at the same path, so sharded
  processes could write one file concurrently — a truncated CSV, meaning a crash at best and
  silently wrong vectors at worst, with no way afterwards to tell which runs were affected. Now
  reused if valid and written through a per-process temp file plus `os.replace`. Also removes 505
  needless rebuilds of a 6 MB table.
- **A hole in the shuffled control.** The derangement ran over SMILES strings, and this dataset
  spells 6 molecules two ways, so a molecule could receive its own vector through its other
  spelling — real information inside a control that must carry none. Now deranged over RDKit
  canonical identity (867 strings, 861 identities) with an assertion.
- **A signature break.** `ComparisonTower` gained `secondary_dim`, and `run_comparison.py` called
  it positionally, so `pgb_tower_dim` was landing in `secondary_dim`. Both call sites now use
  keywords.
- **Significance confused with size.** Rule 1 is a significance test and can pass a gain smaller
  than this study resolves. The rule is unchanged — moving it after seeing data is not allowed —
  but every winner now prints its effect size and is flagged when it falls under the power bound.

### NOT verified — know this before trusting a result

| Gap | Risk |
|---|---|
| **GPU never used.** Everything above ran on CPU. | Low. No device-specific code was added, but the first DGX block is the real test. |
| **molformer, gin, dmpnn never run in the new code.** Only chemberta and gat were. | Low-moderate. dmpnn is the one to watch: 2048-wide, by far the largest adapter. Run `precompute_embeddings` and one `dmpnn__noca` cell before launching the full sweep. |
| **Sharding never run with more than one process.** | Low. The race it depended on is fixed, and the queue split is pure arithmetic — but check that two shards do not claim the same run on the first block. |
| **Full 40-epoch runs never done.** All smoke tests were 1–2 epochs. | Low. Early stopping and LR scheduling are untouched code. |
| **`collect_grid` never run on a complete cell.** Only on the 45 `none` runs plus synthetic data. | Low. Every statistic is verified against scipy/statsmodels, but the first real collection should be eyeballed. |
| **ChemBERTa's tokenizer is still broken.** | High, and deliberate. See Appendix B of the plan. Not a bug in this code; a decision about the baseline. |

### The cheap insurance before committing 25 GPU-hours

```bash
bash experiments/encoder_comparison/run_grid_dgx.sh
```

Step 0 of that script precomputes all 5 embedding tables and runs `audit_reps`, which exits
non-zero on a blocking problem and stops the sweep. That is where a dmpnn or molformer surprise
will show up — in minutes, not after hours of runs. Then block 1 is only 30 runs; check its logs
before letting blocks 2-4 go.
