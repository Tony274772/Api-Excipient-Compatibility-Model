"""Collect the top-5 x 5-seed runs into mean +/- std per model.

    python experiments/top5_seeds/collect.py
    python experiments/top5_seeds/collect.py --out <results folder>

Reads <out>/seed_<s>/<run>/{val,test}_metrics.json ("test" is the 22 external held-out pairs) and
writes two files into <out>:

    per_seed.csv   one row per (model, seed)
    summary.csv    one row per model: n_seeds, mean and std (ddof=1) of every metric,
                   ranked by mean validation PR-AUC
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

OUT = os.path.join(PROJECT_ROOT, "experiments", "results", "top5_seeds")
KEYS = ["pr_auc", "roc_auc", "mcc", "f1", "precision", "recall", "accuracy"]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def correct(m):
    cm = m.get("confusion_matrix") if m else None
    return cm[0][0] + cm[1][1] if cm else None


def main():
    ap = argparse.ArgumentParser(description="Collect the top-5 multi-seed runs.")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.out, "seed_*", "*"))):
        info = load(os.path.join(d, "run_info.json"))
        if not info:            # unfinished run
            continue
        val, test = load(os.path.join(d, "val_metrics.json")), load(os.path.join(d, "test_metrics.json"))
        r = {"run": info["run"], "head": "PGB" if info.get("pgb") else "baseline", "seed": info["seed"]}
        for k in KEYS:
            r[f"val_{k}"] = val.get(k)
        r["val_correct"] = correct(val)
        for k in KEYS:
            r[f"heldout_{k}"] = test.get(k) if test else None
        r["heldout_correct"] = correct(test)
        r["val_threshold"] = val.get("threshold")
        r["minutes"] = info.get("minutes")
        rows.append(r)

    if not rows:
        raise SystemExit(f"no finished runs under {args.out}")
    per_seed = pd.DataFrame(rows).sort_values(["run", "seed"]).reset_index(drop=True)
    per_seed.to_csv(os.path.join(args.out, "per_seed.csv"), index=False)

    metrics = [c for c in per_seed.columns if c.startswith(("val_", "heldout_")) and c != "val_threshold"]
    g = per_seed.groupby(["run", "head"])
    summary = g[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "seeds", g["seed"].apply(lambda s: " ".join(map(str, sorted(s)))))
    summary.insert(0, "n_seeds", g.size())
    summary = summary.reset_index().sort_values("val_pr_auc_mean", ascending=False).reset_index(drop=True)
    summary.index += 1
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    show = ["val_pr_auc", "val_roc_auc", "val_mcc", "val_f1", "heldout_pr_auc", "heldout_correct"]
    table = summary[["run", "head", "n_seeds"]].copy()
    for m in show:
        digits = 1 if m.endswith("correct") else 4
        table[m] = [f"{a:.{digits}f} +/- {b:.{digits}f}" if pd.notna(b) else f"{a:.{digits}f}"
                    for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
    pd.set_option("display.width", 250)
    print(table.to_string())
    incomplete = summary[summary["n_seeds"] < 5]
    if len(incomplete):
        print(f"\nfewer than 5 seeds: {', '.join(incomplete['run'])}")
    print(f"\nwritten to {os.path.join(args.out, 'summary.csv')} and per_seed.csv")


if __name__ == "__main__":
    main()
