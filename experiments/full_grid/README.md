# Full model grid on the 80/20 split

Trains every configuration that has a folder in `metrics/` on the ARES 80/20 split
(train 2770 / val 774), tested on the 22 external held-out pairs.

**60 configurations**: 51 baseline models (`CompatibilityModel`) and 9 PGB models, across
6 encoders x 2 fusions x 3 poolings x 4 losses, plus the 4 fixed-vector sources.

## Steps

### 1. On your machine

```bash
python experiments/full_grid/prepare_data.py      # builds data/grid_8020/
python experiments/full_grid/grid.py              # lists the 60 configurations
python experiments/full_grid/make_bundle.py       # writes ../grid_bundle.zip (279 MB)
```

Upload `grid_bundle.zip` to the **top level of your Google Drive**.

### 2. In Colab

Open `experiments/full_grid/run_grid_colab.ipynb` (upload it to Colab, or
File > Open notebook > Upload). Set **Runtime > Change runtime type > GPU**, then run the cells in
order. They mount Drive, unpack the bundle, install the extra packages, point the output folders at
Drive, and train.

Training is split in two cells on purpose:

```bash
python experiments/full_grid/run_grid.py --all --only fixed_vector --device cuda
python experiments/full_grid/run_grid.py --all --device cuda --time-budget 240
```

The first does the 16 cheap fixed-vector models and tells you how fast your GPU is. The second does
the rest and stops after 4 hours so Colab does not kill it mid-run. **Re-run that cell as often as
needed** — finished runs are skipped and results live on Drive, so progress survives a disconnect.

### 3. Bring the results back

The last cell writes `grid_results_out.zip` to Drive. Download it and unzip into
`project/experiments/results/full_grid/`, then:

```bash
python experiments/full_grid/collect.py
```

which prints the table and writes `experiments/results/full_grid/summary.csv`.

## How long it takes

Measured here on an RTX 4050, one epoch:

| Family | One epoch | Likely per run (early stopping, patience 6) |
|---|---|---|
| fixed_vector (16 runs) | 0.12 min | 2-5 min |
| chemberta (10 runs) | 0.74 min | 10-25 min |
| molformer / GNN (34 runs) | slower still | 20-60 min |

A full sweep is plausibly **15-30 hours** on a Colab GPU, so expect several sessions. Colab free
disconnects after a few hours and has a daily cap; Pro gives longer sessions and a faster GPU. The
`--time-budget` flag and the skip-if-done behaviour exist for exactly this.

To shorten it: `--only fixed_vector chemberta` runs the cheap half, and `--epochs N` caps training
(useful for a smoke test, not for results).

## What is held fixed

Every run uses the same data (`data/grid_8020/`), seed 42, and the recipe its configuration
originally had — the grid varies encoder, fusion, pooling, loss and head, nothing else. The
descriptor normalisation statistics are computed from the **train split only**, and
`prepare_data.py` refuses to build the folder if any held-out pair appears in train or val.

The baseline models tune their threshold on validation and report the held-out pairs as `test`;
the PGB models follow their own pipeline (`src/single_run.py`), which also tunes on validation.

## Files

```
prepare_data.py       builds data/grid_8020/ (split, descriptors, train-only norm stats)
grid.py               the 60 configurations, generated from the project's own naming code
run_grid.py           trains one run or the whole grid; resumable, --time-budget
collect.py            summary.csv ranked by validation PR-AUC
make_bundle.py        zips what Colab needs
run_grid_colab.ipynb  the notebook
```

`grid.py` does not parse folder names by hand: it generates each name with
`Config.resolve_checkpoint_paths()` and matches it against `metrics/`, so the mapping from folder
to configuration is the project's own. 54 folders match exactly. Nine use an older scheme that left
the pooling out of the name; those are added with the default pooling
(`global_gated_attention`) and marked `legacy`, and three of them turn out to be duplicates of a
4-token name and are dropped. Run `python experiments/full_grid/grid.py --check` to see this.

## Verified before shipping

One epoch of each kind ran end to end on this machine:

- `fixed_vector_morgan_concat_asl` (baseline, fixed-vector) — 0.12 min
- `chemberta_cross_attn_cls_bce` (baseline, transformer + cross-attention) — 0.74 min
- `pgb_fixed_vector_morgan_concat_asl` (PGB head) — 0.57 min, and it reproduced the shipped ARES
  numbers exactly (val PR-AUC 0.6859, held-out PR-AUC 0.9415), since that model's best epoch is 1
