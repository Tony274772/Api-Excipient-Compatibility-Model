# Ablations of `chemberta__x_pubchemfp`

The same kind of study as the ARES (Morgan-PGB) ablations in `final_project/docs/ablations.md`,
for the encoder-grid cell chosen as the candidate final model. The plan, including the decision rule
that was fixed before any run, is in `paper/chemberta_pubchemfp_ablation_plan.md`.

`final_project/` is neither modified nor imported. The two pieces taken from it (the `no_family`
prior table and the v2 mechanism rules) are ported here onto `src/` and the encoder-grid code.

## Run on the DGX

From the project root:

```bash
bash experiments/ablations_chemberta/run_dgx.sh
```

It runs these steps in order:

1. **Step 0.** Trains `full` for 5 seeds and checks that it reproduces the grid's
   `chemberta__x_pubchemfp` (val PR-AUC within 0.005 per seed). If it doesn't, the script stops,
   because every comparison is made against this baseline. Pass `SKIP_REPRO_CHECK=1` to continue
   anyway.
2. **Step 1.** Runs the other 22 variants for 5 seeds each, sharded over all GPUs.
3. **Step 2.** Runs `collect.py` and `token_subsets.py`.

That makes 115 runs in total, about 1.2 min each on one GPU (grid timing), or roughly 2.3 GPU-hours.
The script is safe to re-run and to Ctrl-C: a run counts as done only when its `COMPLETE` file
exists, and that file is written last.

It needs the same environment and files as the grid:
- `models/pretrained/chemberta`
- `experiments/results/encoder_grid/_lookup/pubchemfp_lookup.csv`, which is rebuilt from
  `data/pubchem_fps*.csv` if it is missing
- `final_project/data`
- the grid's own results under `experiments/results/encoder_grid/`, for the step-0 check

Individual pieces:

```bash
python -m experiments.ablations_chemberta.run_ablation --list                      # variants + parameters
python -m experiments.ablations_chemberta.run_ablation --ablation no_w --seed 42   # one run
python -m experiments.ablations_chemberta.run_ablation --ablation no_w --smoke     # 1 epoch, _smoke/
python -m experiments.ablations_chemberta.run_seeds --dry-run                      # what is left
python -m experiments.ablations_chemberta.run_seeds --ablations full no_w --seeds 42 43
python -m experiments.ablations_chemberta.collect                                  # validation only
python -m experiments.ablations_chemberta.token_subsets                            # section E
python -m experiments.ablations_chemberta.collect --reveal-heldout                 # AFTER the decision
```

## Files

| File | What it does |
|---|---|
| `variants.py` | Every ablation as `(model) -> model`, the switchable head and tower, `PLAN`, `BUILD_SETTINGS`, `REFERENCE` |
| `run_ablation.py` | One ablation × one seed, in the same order as `encoder_comparison/run_grid.py`; `--list`, `--smoke` |
| `run_seeds.py` | All of `PLAN` × seeds, one subprocess per run, resumable, `--shard I/N` |
| `collect.py` | Step-0 check, paired tests vs the reference, BH, verdicts, 5-seed ensembles |
| `token_subsets.py` | Section E: val pairs split by ChemBERTa's untrained tokens |
| `prior_lookup.py` | `no_family` prior table (exact → nearest → global), also for the leave-cluster-out train priors |
| `mech_v2.py` | The 13 v2 mechanism rules (ported verbatim), on the grid dataset |
| `parent_smiles.py` | `chemberta_parent`: the molecule ChemBERTa reads, and the encoder wrapper |
| `run_dgx.sh` | The whole study on the DGX |

## Outputs

```
experiments/results/ablations_chemberta/
    runs/<ablation>_seed<seed>/          val_metrics.json  run_info.json  training_history.json
                                         predictions_val.csv  COMPLETE
    _heldout_sealed/<ablation>_seed<seed>/   heldout_metrics.json  predictions_heldout.csv
    multi_seed/<ablation>_5seeds/        ensemble predictions (+ held-out only after --reveal-heldout)
    token_subsets/                       val_pair_subsets.csv  per_seed.csv  summary.csv  vs_no_chemberta.csv
    summary.md  summary.csv  per_seed.csv  reproduction.csv
    logs/                                one log per run, plus sweep logs
    _smoke/                              1-epoch test runs; never collected by default
checkpoints/ablations_chemberta/<ablation>_seed<seed>/   best_model.pt (trainable part only)
                                                         prior_table.json  pgb_norm_stats.npz
```

## How variants are built

Each variant builds the full model with the run's seed, exactly as the grid does, and then cuts its
part out of the existing weight matrices. No new layers are added and no extra random numbers are
drawn, so every layer a variant keeps starts from the same weights as `full` at that seed. The
comparison with `full` is therefore paired seed by seed.

Some variants change the build or the data instead:

| Variant | What changes | Still weight-paired? |
|---|---|---|
| `pool_gated` | adds the gated-attention pooling layers; they are created last, so every layer `full` has gets the same initial weights | yes |
| `mech_v2` | the head is created with 13 mechanism inputs, so it draws different random numbers | **no**, marked ‡ in the summary |
| `shuf_pubchemfp` | same model; PubChem bits are deranged over chemical identity (seed 20251110, the grid's control seed) | yes |
| `chemberta_parent` | same model; the ChemBERTa wrapper reads the parent molecule (below) | yes |
| `prior1+no_family` | same weights as `prior1`; only the prior lookup differs | compared with `prior1`, not `full` |

**`no_cross_attn`:** the structural vector is the molecule's own projected ChemBERTa embedding,
which is the vector the CLS position of the cross-attention starts from. The two attention layers,
their LayerNorms and the token placeholder are removed. This is not the grid's `chemberta__noca`,
which put a 384-d table vector through a new 64-wide primary block.

**`no_chemberta`:** removes the projections, cross-attention, LayerNorms, placeholders and the 128
struct columns of both mix layers. The frozen encoder is still loaded but never called.

**`chemberta_parent`:** applies only to molecules containing `.`, `[`, `@`, `/` or `\`. For those,
ChemBERTa reads:
1. the largest organic fragment, with water dropped,
2. neutralised where a neutral form exists,
3. without stereochemistry.

Every other molecule keeps its exact dataset string. Cl and Br stay, because they are part of the
molecule. The salt context, flags, descriptors, PubChem, prior and mechanism flags still read the
original SMILES. `run_info.json` records how many molecules changed.

## Decision rule (plan section 5, fixed before running)

- Each variant is compared with its reference using a paired t-test over 5 seeds on val PR-AUC,
  with BH correction over all ablations at q = 0.10.
- **NEEDED:** removing the part lowers val PR-AUC significantly after BH.
- **REMOVABLE:** the mean drop is ≤ 0.01 PR-AUC and ≤ 0.02 MCC.
- **HELPS when removed:** removing the part raises val PR-AUC significantly after BH.
- **UNCLEAR:** anything else.
- Removable parts are combined into at most two designs. Add them to `variants.DESIGNS`, then run
  `run_seeds --ablations <design>`.
- Held-out is revealed only after the choice is made, and is never used to make it.
