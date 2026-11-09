"""Salt-context ablation for the best grid model: does z-scoring the 8 salt-context values help?

The grid / top-5 / reagent-ablation runs fed the salt context raw (organic_heavy_atoms up to 291 next to
inputs of about +-1), because src/single_run.py never turned on config.pgb_normalize_salt. Arms:

    control             raw salt context. NOT retrained here: the control arm of
                        experiments/reagent_ablation is exactly this recipe on the same data.
    salt_norm           salt context z-scored with train-only statistics
    salt_norm_nowater   z-scored, and water-of-hydration fragments ignored (hydrate = anhydrous form)

Everything else is the grid recipe (experiments/full_grid/run_grid.py:run_one) on data/grid_8020/
(reagents kept), seeds 42-46. Each run writes predictions_val.csv / predictions_heldout.csv, and the
script checks that the saved pgb_norm_stats.npz really carries the salt statistics.

    python experiments/salt_ablation/run.py --list
    python experiments/salt_ablation/run.py --device cuda
    python experiments/salt_ablation/run.py --device cuda --arms salt_norm
    python experiments/salt_ablation/collect.py
"""

import argparse
import json
import os
import sys
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

import experiments.full_grid.run_grid as rg  # noqa: E402
from experiments.full_grid.grid import GRID  # noqa: E402
from experiments.reagent_ablation.run import Capture, save_predictions  # noqa: E402

DATA = os.path.join(PROJECT_ROOT, "data", "grid_8020")
ARMS = {
    "salt_norm": {"pgb_normalize_salt": True, "pgb_salt_drop_water": False},
    "salt_norm_nowater": {"pgb_normalize_salt": True, "pgb_salt_drop_water": True},
}
MODELS = ["pgb_dmpnn_chemprop_cross_attn_cls_asl"]
SEEDS = [42, 43, 44, 45, 46]
OUT = os.path.join(PROJECT_ROOT, "experiments", "results", "salt_ablation")
CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "salt_ablation")


def run_dir(root, arm, seed, name):
    return os.path.join(root, arm, f"seed_{seed}", name)


def is_done(arm, seed, name):
    return os.path.exists(os.path.join(run_dir(OUT, arm, seed, name), "run_info.json"))


def run(arm, name, seed, device, epochs):
    settings = ARMS[arm]
    rg.DATA = DATA
    orig_make_config = rg.make_config

    def make_config(*a, **k):                    # run_one calls this module global
        c = orig_make_config(*a, **k)
        for key, value in settings.items():
            setattr(c, key, value)
        return c

    res_root = os.path.dirname(run_dir(OUT, arm, seed, name))
    ck_root = os.path.dirname(run_dir(CKPT, arm, seed, name))
    rg.make_config = make_config
    try:
        with Capture() as cap:
            rg.run_one(name, seed, device, epochs, results_root=res_root, ckpt_root=ck_root)
    finally:
        rg.make_config = orig_make_config

    # the setting must have reached the saved normalisation stats, or the arm tested nothing
    stats = np.load(os.path.join(run_dir(CKPT, arm, seed, name), "pgb_norm_stats.npz"))
    if settings["pgb_normalize_salt"] != ("salt_mean" in stats.files) or \
            settings["pgb_salt_drop_water"] != ("salt_drop_water" in stats.files):
        raise RuntimeError(f"pgb_norm_stats.npz has {stats.files}, expected the settings {settings}")

    out = run_dir(OUT, arm, seed, name)
    val_m = json.load(open(os.path.join(out, "val_metrics.json")))
    n_val = len(pd.read_csv(os.path.join(DATA, "val.csv")))
    vp, vl = [c for c in cap.calls if len(c[0]) == n_val][-1]
    hp, hl = cap.calls[-1]
    if abs(average_precision_score(vl.astype(int), vp) - val_m["pr_auc"]) > 1e-6:
        raise RuntimeError("captured val predictions do not reproduce the saved val PR-AUC")
    save_predictions(os.path.join(DATA, "val.csv"), vp, vl, val_m["threshold"], os.path.join(out, "predictions_val.csv"))
    save_predictions(os.path.join(DATA, "test.csv"), hp, hl, val_m["threshold"],
                     os.path.join(out, "predictions_heldout.csv"))

    info_path = os.path.join(out, "run_info.json")
    info = json.load(open(info_path))
    info.update(arm=arm, settings=settings,
                rows={s: len(pd.read_csv(os.path.join(DATA, f"{s}.csv"))) for s in ("train", "val", "test")})
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


def main():
    p = argparse.ArgumentParser(description="Salt-context ablation x seeds.")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=None, help="cap max epochs (smoke tests only)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    unknown = [m for m in args.models if m not in GRID]
    if unknown:
        raise SystemExit(f"unknown run(s): {unknown}")
    planned = [(a, m, s) for a in args.arms for m in args.models for s in args.seeds]
    if args.list:
        for a, m, s in planned:
            print(f"{'done ' if is_done(a, s, m) else '     '}{a:18s} seed {s}  {m}")
        print(f"\n{sum(is_done(a, s, m) for a, m, s in planned)} of {len(planned)} finished")
        return
    if not os.path.exists(os.path.join(DATA, "train.csv")):
        raise SystemExit("data/grid_8020/ is missing. Run: python experiments/full_grid/prepare_data.py")

    todo = [(a, m, s) for a, m, s in planned if args.overwrite or not is_done(a, s, m)]
    print(f"{len(planned)} runs planned, {len(todo)} to do | out {os.path.relpath(OUT, PROJECT_ROOT)}", flush=True)
    t0, failed = time.time(), []
    for i, (arm, name, seed) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {arm}  seed {seed}  {name}  ({(time.time() - t0) / 60:.0f} min spent)", flush=True)
        try:
            run(arm, name, seed, args.device, args.epochs)
        except Exception as e:
            failed.append((arm, name, seed, f"{type(e).__name__}: {e}"))
            print(f"[{arm} {name} seed {seed}] FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    missing = sum(not is_done(a, s, m) for a, m, s in planned)
    print(f"\nfailed {len(failed)}, {missing} of {len(planned)} still missing, {(time.time() - t0) / 60:.0f} min")
    for a, n, s, e in failed:
        print(f"  failed: {a} seed {s} {n} — {e}")
    print("\nCollect with: python experiments/salt_ablation/collect.py")


if __name__ == "__main__":
    main()
