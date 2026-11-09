# Salt-context ablation

The best grid model (`pgb_dmpnn_chemprop_cross_attn_cls_asl`) and every grid, top-5 and reagent-ablation run
fed the 8 salt-context values raw into each tower's mix layer. `src/single_run.py` never turned on
`config.pgb_normalize_salt`. On train rows, `organic_heavy_atoms` is 32 ± 19 for APIs (maximum 291) and
16 ± 16 for excipients (maximum 116). The other inputs to the mix layer are around ±1, and training reaches its
best val score at epoch 2-4. A second issue is that water of hydration counts as a fragment, so lactose
monohydrate looks like a two-fragment salt while lactose does not.

| Arm | Salt context | Where it comes from |
|---|---|---|
| `control` | raw (water counted) | `experiments/results/reagent_ablation/control/`: same recipe and data, not retrained |
| `salt_norm` | z-scored with train-only mean / std | this folder |
| `salt_norm_nowater` | z-scored, water-of-hydration fragments ignored | this folder |

All arms use `data/grid_8020/` (reagents kept), seeds 42-46, and the grid recipe through `run_one`. The runner
checks that every checkpoint's `pgb_norm_stats.npz` carries the settings of its arm.

## Code changes (all opt-in, defaults reproduce every earlier run)

- `src/config.py`: new `pgb_salt_drop_water` (default False), next to the existing `pgb_normalize_salt`.
- `src/single_run.py`: passes both settings to `PGBNormStats.from_smiles_list`. Before this change, the setting was ignored.
- `src/features.py`: `salt_context(smiles, drop_water=False)`.
- `src/dataset.py`: `PGBNormStats` stores `salt_drop_water` in `pgb_norm_stats.npz` and builds the salt context
  through `norm.salt(smiles)`. A checkpoint therefore always gets the features it was trained with. Old
  `.npz` files have no salt keys and still give the raw vector. A salt column that is constant on train is
  left unscaled instead of being divided by ~0.

After z-scoring, the largest train value is |z| = 12.6. That is the 291-heavy-atom API, which remains an
outlier but is no longer 290× the scale of the other inputs.

## Commands (DGX, from `project/`)

```bash
bash experiments/salt_ablation/run_dgx.sh
```

With one GPU: `NGPU=1 bash experiments/salt_ablation/run_dgx.sh`. Step by step:

```bash
python experiments/salt_ablation/run.py --device cuda
python experiments/salt_ablation/collect.py
```

`run.py --list` shows progress. Finished runs are skipped. Bring back `experiments/results/salt_ablation/`.

## Output

`collect.py` writes `per_seed.csv`, `summary.csv` and `delta.csv`, each arm against control paired by seed. It
reports three views:

- `heldout_*`: the 22 pairs.
- `val_*`: all 774 val rows.
- `valclean_*`: the 758 val rows without reagents.
