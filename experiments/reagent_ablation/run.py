"""Reagent ablation: train the best grid model with and without the reagent rows, 5 seeds each.

Two arms, identical except for the data folder:

    control       data/grid_8020/                 (the split every grid / top-5 run used)
    no_reagents   data/grid_8020_no_reagents/     (reagent rows removed from train and val)

Training goes through experiments/full_grid/run_grid.py:run_one, so the recipe (encoder, PGB head,
ASL loss, early stopping, threshold tuning on val) is exactly the one in
experiments/results/top5_seeds_depth6/. The control arm re-runs that recipe here because the grid
runs saved no per-row predictions: with them, both arms can be scored on the same reagent-free val
rows (see collect.py). The control arm should land close to top5_seeds_depth6.

Every run also writes predictions_val.csv and predictions_heldout.csv next to its metrics.
A run whose run_info.json exists is skipped, so an interrupted run resumes with the same command.

    python experiments/reagent_ablation/reagents.py                      # once: build the data
    python experiments/reagent_ablation/run.py --list
    python experiments/reagent_ablation/run.py --device cuda             # both arms, seeds 42-46
    python experiments/reagent_ablation/run.py --device cuda --arms no_reagents
    python experiments/reagent_ablation/collect.py
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
from experiments.reagent_ablation.reagents import DST as NO_REAGENTS_DATA  # noqa: E402

ARMS = {
    "control": os.path.join(PROJECT_ROOT, "data", "grid_8020"),
    "no_reagents": NO_REAGENTS_DATA,
}
MODELS = ["pgb_dmpnn_chemprop_cross_attn_cls_asl"]
SEEDS = [42, 43, 44, 45, 46]
OUT = os.path.join(PROJECT_ROOT, "experiments", "results", "reagent_ablation")
CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "reagent_ablation")


def run_dir(root, arm, seed, name):
    return os.path.join(root, arm, f"seed_{seed}", name)


def is_done(arm, seed, name):
    return os.path.exists(os.path.join(run_dir(OUT, arm, seed, name), "run_info.json"))


class Capture:
    """Records every (probs, labels) that collect_predictions returns during one run.

    Early stopping also scores val after every epoch, so many calls are val-sized. After training,
    both paths score val once more with the final model and then the held-out pairs (the PGB path
    scores the held-out file twice, as "test" and as "heldout"). So the held-out predictions are the
    last call, and the final val predictions are the last val-sized call; run() checks the latter
    against the PR-AUC the run saved.
    """

    def __init__(self):
        import src.evaluate as ev
        import src.single_run as sr
        self.mods, self.orig, self.calls = (ev, sr), ev.collect_predictions, []

        def wrapped(*a, **k):
            probs, labels = self.orig(*a, **k)
            self.calls.append((np.asarray(probs).ravel(), np.asarray(labels).ravel()))
            return probs, labels
        self.wrapped = wrapped

    def __enter__(self):
        for m in self.mods:
            m.collect_predictions = self.wrapped
        return self

    def __exit__(self, *exc):
        for m in self.mods:
            m.collect_predictions = self.orig


def save_predictions(csv, probs, labels, threshold, path):
    df = pd.read_csv(csv)
    if len(df) != len(probs):
        raise RuntimeError(f"{len(probs)} predictions for {len(df)} rows of {csv}")
    if (df["Outcome1"].to_numpy().astype(int) != labels.astype(int)).any():
        raise RuntimeError(f"captured labels do not match {csv}: prediction order is not the file order")
    out = df[[c for c in ("Api_name", "Excipient_name", "API_CID", "Excipient_CID") if c in df]].copy()
    out["label"] = labels.astype(int)
    out["probability"] = probs
    out["prediction"] = (probs >= threshold).astype(int)
    out.to_csv(path, index=False)


def run(arm, name, seed, device, epochs):
    data = ARMS[arm]
    rg.DATA = data                               # make_config reads this module global
    res_root = os.path.dirname(run_dir(OUT, arm, seed, name))
    ck_root = os.path.dirname(run_dir(CKPT, arm, seed, name))
    with Capture() as cap:
        rg.run_one(name, seed, device, epochs, results_root=res_root, ckpt_root=ck_root)

    out = run_dir(OUT, arm, seed, name)
    val_m = json.load(open(os.path.join(out, "val_metrics.json")))
    thr = val_m["threshold"]
    n_val = len(pd.read_csv(os.path.join(data, "val.csv")))
    vp, vl = [c for c in cap.calls if len(c[0]) == n_val][-1]
    hp, hl = cap.calls[-1]
    pr = average_precision_score(vl.astype(int), vp)
    if abs(pr - val_m["pr_auc"]) > 1e-6:
        raise RuntimeError(f"captured val predictions give PR-AUC {pr:.6f}, the run saved {val_m['pr_auc']:.6f}")
    save_predictions(os.path.join(data, "val.csv"), vp, vl, thr, os.path.join(out, "predictions_val.csv"))
    save_predictions(os.path.join(data, "test.csv"), hp, hl, thr, os.path.join(out, "predictions_heldout.csv"))

    # run_one hard-codes the grid row counts; record the real ones and the arm
    info_path = os.path.join(out, "run_info.json")
    info = json.load(open(info_path))
    info["arm"] = arm
    info["rows"] = {s: len(pd.read_csv(os.path.join(data, f"{s}.csv"))) for s in ("train", "val", "test")}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


def main():
    p = argparse.ArgumentParser(description="Reagent ablation (control vs no_reagents) x seeds.")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=None, help="cap max epochs (smoke tests only)")
    p.add_argument("--overwrite", action="store_true", help="retrain runs that already finished")
    p.add_argument("--list", action="store_true", help="show what is done and stop")
    args = p.parse_args()

    unknown = [m for m in args.models if m not in GRID]
    if unknown:
        raise SystemExit(f"unknown run(s): {unknown}. See python experiments/full_grid/grid.py")
    planned = [(a, m, s) for a in args.arms for m in args.models for s in args.seeds]
    if args.list:
        for a, m, s in planned:
            print(f"{'done ' if is_done(a, s, m) else '     '}{a:12s} seed {s}  {m}")
        print(f"\n{sum(is_done(a, s, m) for a, m, s in planned)} of {len(planned)} finished")
        return

    for a in args.arms:
        marker = "split_log.txt" if a == "no_reagents" else "train.csv"
        if not os.path.exists(os.path.join(ARMS[a], marker)):
            fix = ("experiments/reagent_ablation/reagents.py" if a == "no_reagents"
                   else "experiments/full_grid/prepare_data.py")
            raise SystemExit(f"{os.path.relpath(ARMS[a], PROJECT_ROOT)} is missing. Run: python {fix}")
    if args.epochs:
        print(f"WARNING: --epochs {args.epochs} caps training; use only for a smoke test")

    todo = [(a, m, s) for a, m, s in planned if args.overwrite or not is_done(a, s, m)]
    print(f"{len(planned)} runs planned, {len(todo)} to do | out {os.path.relpath(OUT, PROJECT_ROOT)}", flush=True)
    t0, failed = time.time(), []
    for i, (arm, name, seed) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {arm}  seed {seed}  {name}  ({(time.time() - t0) / 60:.0f} min spent)",
              flush=True)
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
    print("\nCollect with: python experiments/reagent_ablation/collect.py")


if __name__ == "__main__":
    main()
