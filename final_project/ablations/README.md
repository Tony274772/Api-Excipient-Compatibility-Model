# Ablations of the Morgan-PGB 80/20 model

Each ablation is trained with the same data (`data/train.csv`, `data/val.csv`, `data/held_out_test_set.csv`),
recipe, seed (42) and threshold rule (best val MCC with TNR >= 0.97) as `python run.py train`. Only the
model changes. Nothing in `src/`, `run.py` or the final checkpoints is touched.

| Name | Change | Formula |
|---|---|---|
| `full` | none (baseline through this harness) | `z = r + w * logit(p_exact) + b` |
| `no_w` | A. adaptive prior trust removed (`prior_trust_mlp`, 57 params) | `z = r + b` |
| `prior1` | B. prior 5 -> 1 (`p_exact` only): gate 46 -> 42, MLP 410 -> 406, prior trust 1 -> 8 -> 1; `w` kept | `z = r + w * logit(p_exact) + b` |
| `no_gate` | C. gate on the bilinear term removed (2,032 params): `gated = bilinear`, MLP input still 410 | `z = r + w * logit(p_exact) + b` |
| `no_bilinear` | D. bilinear term `U*h_api (.) V*h_exc` and its gate removed (U, V, gate), MLP input 410 -> 394 | `z = r + w * logit(p_exact) + b` |
| `no_prior` | E. excipient prior removed: gate 46 -> 41, MLP 410 -> 405, prior trust `w` and its term removed | `z = r + b` |
| `no_salt` | F. 8-value salt context removed from both towers: mix layer 145 -> 137 | `z = r + w * logit(p_exact) + b` |
| `no_avail` | G. availability flag (mix input) and missing-fingerprint embedding removed from both towers: mix 145 -> 144 | `z = r + w * logit(p_exact) + b` |
| `no_mech` | H. 5 SMARTS mechanism flags removed from the head: gate 46 -> 41, MLP 410 -> 405 | `z = r + w * logit(p_exact) + b` |
| `no_flags` | I. all 18 chemistry flags removed from both towers (flag block, mix 145 -> 129) and the gate (46 -> 10) | `z = r + w * logit(p_exact) + b` |
| `no_flags_api` | I-a. the 18 API chemistry flags removed (API tower mix 145 -> 129, gate 46 -> 28) | `z = r + w * logit(p_exact) + b` |
| `no_flags_exc` | I-b. the 18 excipient chemistry flags removed (excipient tower mix 145 -> 129, gate 46 -> 28) | `z = r + w * logit(p_exact) + b` |
| `no_flags_metal` | I-c. the 10 metal / inorganic flags removed on both sides (flag blocks 18 -> 8, gate 46 -> 26) | `z = r + w * logit(p_exact) + b` |
| `no_family` | J. data only: excipient prior looked up as exact -> nearest neighbour -> global (family fallback removed); run as `prior1+no_family` | unchanged |
| `train_global` | K. data only: prior global rate = train positive rate (0.0996) in place of the hard-coded 0.094; run as `prior1+no_family+train_global` | unchanged |

Ablations combine with `+`, applied left to right (the order does not change the model):
`prior1+no_w` = 1-D prior and no prior trust, `z = r + b` (the "prior1" graphical-abstract variant).

| Name | Parameters |
|---|---|
| `full` | 236,219 |
| `no_w` | 236,162 |
| `prior1` | 235,547 |
| `prior1+no_w` | 235,522 |
| `no_gate` | 234,187 |
| `no_bilinear` | 229,067 |
| `no_prior` | 235,362 |
| `no_salt` | 234,683 |
| `no_avail` | 235,899 |
| `no_mech` | 235,419 |
| `no_flags` | 231,387 |
| `no_flags_api` / `no_flags_exc` | 233,803 |
| `no_flags_metal` | 235,259 |

In `no_w` the 5-value prior still goes into the gate and the MLP; only the `w * logit(p_exact)` shortcut
is gone. The bias `b` keeps its starting value (0.2 x train log-odds) and is still learned.

In `prior1` the dataset and prior table are unchanged (still 5 values per row); the head keeps only
column 0 (`p_exact`). Unseen excipients then get the global rate 0.094 with no `unseen` / `log_n` marker.

In `no_gate` the flags still reach the model through the towers, and the prior and mechanism flags
through the MLP input; only the per-pair gating of the 16 bilinear channels is gone. At init the gate
outputs about 0.5, so without it the bilinear term starts at about twice the scale.

In `no_bilinear` the other pair features stay: `h_api`, `h_exc`, `h_api * h_exc` and `|h_api - h_exc|`,
plus the prior, mechanism flags and `w`. The gate goes too, since it only scales the bilinear term
(`no_gate+no_bilinear` is the same model as `no_bilinear`).

In `no_prior` the model never reads the excipient prior (its output does not change when `prior_vec`
changes); it predicts from the towers, flags and mechanism SMARTS only. The dataset still builds the
prior and the prior table is still fitted and saved, because loading a checkpoint expects it.
`no_prior` already includes `no_w`, and `prior1+no_prior` is the same model as `no_prior`.

In `no_salt` the towers no longer get the salt context (fragments, metal atoms, inorganic fragments,
total / max formal charge, organic / inorganic heavy atoms, counterion). Fingerprints, descriptors and
the chemistry flags (incl. `is_salt`, `has_metal`, `has_Mg`, ...) are unchanged, as is the head.
The organic heavy-atom count is roughly molecule size, so a raw size signal goes too (MolWt is still
in the descriptors). `no_salt` only touches the towers, so it combines with every head ablation.

In `no_avail` the towers lose the primary-fingerprint availability flag and the `missing_primary`
embedding used when the flag is 0. With Morgan bits computed from SMILES the flag is 1 for every pair in
train, val and held-out, and `missing_primary` was never trained (all zeros), so the flag only acts as
an extra bias in the mix layer. Its weights are not folded into the mix bias; the constant input is gone.

In `no_mech` the head no longer sees the 5 pair-level mechanism flags (Maillard, ester/amide hydrolysis,
metal-ion oxidation, acid-base salt displacement, Schiff base) in the gate or the MLP. The 18 chemistry
flags per molecule are unchanged. `keep_mech(head, names)` keeps a subset, for leave-one-out runs.

The chemistry flags (18 per molecule, `src/features.py`) are read in two places: each tower's flag block
(18 -> 16, into the mix layer) and the gate (API and excipient flags, 36 of its 46 inputs). The flag
ablations remove them from both. The metal / inorganic group is `no_carbons, has_metal, has_Mg, has_Ca,
has_Al, has_Si, has_Ti_Fe, has_Mn_Cr, is_salt, very_short`; it overlaps with the salt context, so
`no_flags_metal+no_salt` tests whether one covers for the other. `keep_flags(model, side, names)` keeps
any other subset.

`no_family` changes the data, not the model (`DATA_SETTINGS` in `variants.py` sets
`config.prior_lookup = "no_family"`; `prior_lookup.py` has the table). The family key of `src/pgb_prior.py`
strips water / solvents and keeps the longest fragment, so it groups all metal salts of one anion (all
phosphates, all stearates). Without it, `p_exact` changes for 24 of 2,770 train rows (leave-own-cluster-out),
4 of 774 val rows and 1 held-out pair; every other row keeps its prior. `prior1+no_family` has exactly the
weights of `prior1`, so compare it with `prior1`.

`train_global` (data only, `prior_global_rate = "train"`) replaces `GLOBAL_RATE = 0.094` of `src/pgb_prior.py`
(left over from an earlier split) with the train positive rate 276 / 2770 = 0.0996, in the Laplace
pseudocount `p = (pos + 4 * rate) / (n + 4)`, the unseen fallback and `p_delta`. The rate is stored in the
config (`prior_global_rate_value`), so evaluation uses the same table. Most excipients have 1-3 train rows,
but the change is small: `p_exact` moves by at most 0.0056 (+0.001 on average). Compare
`prior1+no_family+train_global` with `prior1+no_family` (identical weights). `prior_lookup.py` builds the
table class from both settings; with neither it is `src`'s `ExcipientPriorTable` itself.

Each variant first builds the full model and then removes its part, so every layer it keeps starts
from exactly the same weights as the full model with the same seed. Removed inputs are cut out of the
existing weight matrices (no new layers, no extra random draws). One side effect: the cut prior trust
layer in `prior1` keeps init weights drawn for 5 inputs, so they are smaller than a fresh 1-input layer's.

## Run

```bash
python ablations/run_ablation.py train --ablation no_w
python ablations/run_ablation.py train --ablation prior1
python ablations/run_ablation.py train --ablation prior1+no_w
python ablations/run_ablation.py train --ablation no_gate
python ablations/run_ablation.py train --ablation no_bilinear
python ablations/run_ablation.py train --ablation no_prior
python ablations/run_ablation.py train --ablation no_salt
python ablations/run_ablation.py train --ablation no_avail
python ablations/run_ablation.py train --ablation no_mech
python ablations/run_ablation.py train --ablation no_flags
python ablations/run_ablation.py train --ablation no_flags_api
python ablations/run_ablation.py train --ablation no_flags_exc
python ablations/run_ablation.py train --ablation no_flags_metal
python ablations/run_ablation.py train --ablation prior1+no_family
python ablations/run_ablation.py train --ablation prior1+no_family+train_global
python ablations/run_ablation.py train --ablation full
python ablations/run_ablation.py evaluate --ablation no_w
python ablations/run_ablation.py list
```

Options for `train`: `--device cuda|cpu`, `--seed N` (default 42), `--epochs N`, `--overwrite`.

`full` trained here should match `checkpoints/morgan_pgb_8020_retrain_seed42` (20/24) when it runs on
the same device; train it once so every ablation is compared against a baseline from the same harness.

## Output

```
ablations/checkpoints/<ablation>_seed42/   best_model.pt, prior_table.json, pgb_norm_stats.npz, threshold.json
ablations/results/<ablation>_seed42/       val / heldout / heldout_unseen metrics, predictions, history, run_info
ablations/results/summary.csv              one row per ablation (rewritten on every train / evaluate)
```

## Adding an ablation

Write a function `(model) -> model` in `variants.py` and add it to `ABLATIONS` and `DESCRIPTIONS`.
Ablations of the head call `ablation_head(model)` and set its switches (`prior_cols`, `use_prior_trust`, `use_gate`, `use_bilinear`, `mech_cols`, `api_flag_cols`, `exc_flag_cols`);
tower ablations call `ablation_tower(tower)` (`flag_cols`) or set `use_salt_context` / `use_missing_embedding`
so they combine with the others.

## Mechanism flags v2 (`--mech v2`)

`mech_v2.py` replaces the 5 SMARTS mechanism flags of `src/lookup.py` with 13 pair-level rules written
from the API-excipient incompatibility literature (Maillard, base / acid / metal-catalysed hydrolysis,
acid-base reactions, salt disproportionation, chelation, strong oxidisers, sulfite addition, ion pairs,
aluminosilicate surfaces). The docstring of `mech_v2.py` lists what changed against v1 and why.
Four more literature rules (metal-catalysed oxidation, peroxide oxidation, Michael addition, amine
acylation) are kept in `RULES_NOT_USED`: on train / val they never mark incompatible pairs more often
than the base rate.

Rules were checked on train and val only; the held-out pairs are not read:

```bash
python ablations/check_mech_v2.py                # per-rule fires, precision, lift, recall: v1 vs v2
python ablations/check_mech_v2.py --unused --examples 5
```

| Any flag fires | v1 precision (lift) | v1 recall | v2 precision (lift) | v2 recall |
|---|---|---|---|---|
| train (base 0.100) | 0.106 (1.06) | 0.27 | 0.327 (3.28) | 0.30 |
| val (base 0.088) | 0.084 (0.96) | 0.27 | 0.459 (5.22) | 0.57 |

Train with the v2 flags (any ablation works with it):

```bash
python ablations/run_ablation.py train --ablation full --mech v2
python ablations/run_ablation.py evaluate --ablation full --mech v2
```

The head grows from 5 to 13 mechanism inputs (gate 46 -> 54, MLP 410 -> 418, 237,499 parameters).
Output goes to `ablations/checkpoints/full_mechv2_seed42/` and `ablations/results/full_mechv2_seed42/`,
and `summary.csv` gets a `mech` column. The larger layers draw more random numbers at init, so `full`
with v1 and v2 do not start from identical weights (unlike the ablations above).

## Global-rate fix and 5-seed runs

`src/pgb_prior.py` used a hard-coded global rate 0.094 (left over from an earlier split) as the smoothing
pseudocount, unseen fallback and `p_delta` reference. It now takes the train positive rate (276 / 2770 =
0.0996) from the caller and saves it in `prior_table.json`; `run.py train` and every new ablation run use it.
Tables saved before load with 0.094, so the shipped model (20/24) and all earlier checkpoints score exactly
as before. `old_global` reproduces the old rate in a new run.

The single-seed runs made with 0.094 are in `archive_global0094/` (checkpoints, results, summary). With the
fix, `full` is the old `train_global`; its seeds 42-46 are in `checkpoints/full_seed4*` (copies of
`train_global_seed4*`) and `results/full_5seeds/`.

```bash
python ablations/run_seeds.py                       # every ablation, seeds 42-46, skips runs that exist
python ablations/run_seeds.py --ablations no_w no_prior --seeds 42 43 44 45 46
python ablations/run_seeds.py --collect-only        # rebuild results/<ablation>_5seeds/
```

Each `results/<ablation>_5seeds/` holds per-seed val / held-out predictions, mean +- sd and a 5-seed ensemble
(threshold from val), as written by `collect_seeds.py`. Logs go to `results/logs/`.

## Candidate final models

| Name | Keeps | Removes | Ablations | Parameters |
|---|---|---|---|---|
| `design_a` | `w`, 1-D prior, chemistry flags, bilinear term | family lookup, salt context, gate, availability flag, mechanism flags | `prior1+no_family+no_salt+no_gate+no_avail+no_mech` | 231,147 |
| `design_b` | as `design_a`, plus salt context and family lookup | gate, availability flag, mechanism flags | `prior1+no_gate+no_avail+no_mech` | 232,683 |

Without the gate, the chemistry flags reach the model only through the towers. Both designs use the train-rate
prior. Train them on seeds 42-46 and compare with `results/full_5seeds/`:

```bash
python ablations/run_seeds.py --ablations design_a design_b
```

Output goes to `checkpoints/design_*_seed*`, `results/design_*_seed*` and `results/design_*_5seeds/`.
