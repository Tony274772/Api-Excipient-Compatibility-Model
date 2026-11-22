# Ablations of `dmpnn__x_maccs` (Final ARES Model)

Ablation study for the candidate final model selected from the 110-cell encoder sweep:
**`dmpnn__x_maccs`** (CheMeleon D-MPNN Graph Neural Network + 167-bit MACCS Fingerprint).

The pre-registered plan, decision rules, and hypotheses are in [`paper/dmpnn_maccs_ablation_plan.md`](../../paper/dmpnn_maccs_ablation_plan.md).

---

## Quick Start on the DGX

From the project root:

```bash
bash experiments/ablations_dmpnn_maccs/run_dgx.sh
```

This runs:
1. **Step 0**: Trains `full` for 5 seeds (42–46) and checks that it reproduces the grid's `dmpnn__x_maccs` (Val PR-AUC within ±0.005 per seed).
2. **Step 1**: Runs the remaining 22 variants across 5 seeds, parallelized across all available GPUs.
3. **Step 2**: Runs `collect.py` (BH correction, verdicts) and `subsets.py` (salt/sugar breakdowns).

Total: 23 variants × 5 seeds = 115 runs (~3.0–3.5 GPU hours on DGX).
Resumable: only unfinished runs are executed.

---

## CLI Commands

```bash
# View all 23 variants and their parameter counts:
python -m experiments.ablations_dmpnn_maccs.run_ablation --list

# Smoke test a single variant (1 epoch, saves in _smoke/):
python -m experiments.ablations_dmpnn_maccs.run_ablation --ablation no_secondary --smoke

# Run a single ablation and seed:
python -m experiments.ablations_dmpnn_maccs.run_ablation --ablation no_secondary --seed 42

# Check remaining queue without running:
python -m experiments.ablations_dmpnn_maccs.run_seeds --dry-run

# Collect results and generate summary table (validation only, held-out sealed):
python -m experiments.ablations_dmpnn_maccs.collect

# Chemical subset diagnostics (salts, neutral organics, sugars):
python -m experiments.ablations_dmpnn_maccs.subsets

# Reveal held-out test set (ONLY AFTER the final architecture decision is locked):
python -m experiments.ablations_dmpnn_maccs.collect --reveal-heldout
```

---

## File Architecture

| File | Purpose |
|---|---|
| `variants.py` | In-place weight slicing functions, switchable head and towers, registry of all 23 variants, `PLAN`, and `BUILD_SETTINGS`. |
| `run_ablation.py` | One ablation × one seed runner. Implements Step 0 reproduction check, forward checks, training, and threshold tuning. |
| `run_seeds.py` | Sweep dispatcher supporting sharding (`--shard I/N`), resume capability, and logging. |
| `collect.py` | Step 0 check, paired $t$-tests against reference, Benjamini-Hochberg FDR ($q=0.10$), verdicts, and 5-seed ensembling. |
| `subsets.py` | Diagnostic breakdowns on salt vs. neutral pairs and sugars/polyols. |
| `parent_smiles.py` | Graph desalting / covalent fragment extraction for D-MPNN in `dmpnn_parent`. |
| `mech_v2.py` | 13 literature-curated reaction SMARTS rules. |
| `prior_lookup.py` | Excipient prior table without chemical family fallback. |
| `run_dgx.sh` | Bash script for multi-GPU parallel execution on the DGX server. |

