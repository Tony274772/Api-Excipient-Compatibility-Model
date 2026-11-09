"""Collect the salt-context ablation: control (raw salt) vs salt_norm vs salt_norm_nowater.

    python experiments/salt_ablation/collect.py

The control runs are read from experiments/results/reagent_ablation/control/ (same recipe, same data,
raw salt context). All arms use data/grid_8020/, so val and held-out rows are identical across arms.
Views, at each run's own val-tuned threshold:

    heldout_*    the 22 external pairs
    val_*        all 774 val rows
    valclean_*   the 758 val rows without laboratory reagents

Writes experiments/results/salt_ablation/per_seed.csv, summary.csv and delta.csv (per-seed paired
difference of each arm against control).
"""

import glob
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.reagent_ablation.collect import metrics  # noqa: E402
from experiments.reagent_ablation.reagents import REAGENTS  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "experiments", "results", "salt_ablation")
CONTROL = os.path.join(PROJECT_ROOT, "experiments", "results", "reagent_ablation", "control")
VIEWS = ["heldout", "val", "valclean"]
KEYS = ["pr_auc", "roc_auc", "mcc", "f1", "accuracy", "correct"]
ORDER = ["control", "salt_norm", "salt_norm_nowater"]


def load_runs():
    dirs = [("control", d) for d in glob.glob(os.path.join(CONTROL, "seed_*", "*"))]
    dirs += [(None, d) for d in glob.glob(os.path.join(OUT, "*", "seed_*", "*"))]
    rows = []
    for arm, d in sorted(dirs, key=lambda x: x[1]):
        if not os.path.exists(os.path.join(d, "run_info.json")):
            continue
        info = json.load(open(os.path.join(d, "run_info.json")))
        val = pd.read_csv(os.path.join(d, "predictions_val.csv"))
        held = pd.read_csv(os.path.join(d, "predictions_heldout.csv"))
        r = {"arm": arm or info["arm"], "run": info["run"], "seed": info["seed"],
             "threshold": json.load(open(os.path.join(d, "val_metrics.json")))["threshold"]}
        clean = val[~val["Excipient_CID"].isin(REAGENTS)]
        for view, df in (("heldout", held), ("val", val), ("valclean", clean)):
            for k, v in metrics(df).items():
                r[f"{view}_{k}"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    per = load_runs()
    if per.empty or set(per["arm"]) == {"control"}:
        raise SystemExit(f"no finished salt-ablation runs under {OUT}")
    per["arm"] = pd.Categorical(per["arm"], ORDER, ordered=True)
    per = per.sort_values(["run", "arm", "seed"]).reset_index(drop=True)
    os.makedirs(OUT, exist_ok=True)
    per.to_csv(os.path.join(OUT, "per_seed.csv"), index=False)

    cols = [f"{v}_{k}" for v in VIEWS for k in KEYS]
    g = per.groupby(["run", "arm"], observed=True)
    summary = g[cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "n_seeds", g.size())
    summary = summary.reset_index()
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    base = per[per["arm"] == "control"].set_index(["run", "seed"])[cols]
    deltas = []
    for arm in ORDER[1:]:
        a = per[per["arm"] == arm].set_index(["run", "seed"])[cols]
        d = (a - base).dropna(how="all")
        if len(d):
            deltas.append(d.assign(arm=arm).reset_index())
    delta = pd.concat(deltas, ignore_index=True)
    delta.to_csv(os.path.join(OUT, "delta.csv"), index=False)

    show = ["heldout_pr_auc", "heldout_roc_auc", "heldout_mcc", "heldout_correct",
            "val_pr_auc", "val_roc_auc", "val_mcc", "valclean_pr_auc", "valclean_roc_auc", "valclean_mcc"]
    t = summary[["run", "arm", "n_seeds"]].copy()
    for m in show:
        dg = 1 if m.endswith("correct") else 4
        t[m] = [f"{x:.{dg}f} +/- {s:.{dg}f}" if pd.notna(s) else f"{x:.{dg}f}"
                for x, s in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
    pd.set_option("display.width", 250)
    print("mean +/- std over seeds (control = raw salt context, from reagent_ablation/control)\n")
    print(t.T.to_string(header=False))
    print("\npaired difference vs control (mean +/- std over seeds):")
    for (run, arm), d in delta.groupby(["run", "arm"]):
        print(f"  {arm}  ({run}, n={len(d)})")
        for m in show:
            print(f"    {m:18s} {d[m].mean():+.4f} +/- {d[m].std():.4f}  (worse in {(d[m] < 0).sum()}/{len(d)})")
    incomplete = summary[summary["n_seeds"] < 5]
    if len(incomplete):
        print("\nfewer than 5 seeds: " + ", ".join(f"{a}" for a in incomplete["arm"]))
    print(f"\nwritten to {os.path.relpath(OUT, PROJECT_ROOT)}/ (per_seed.csv, summary.csv, delta.csv)")


if __name__ == "__main__":
    main()
