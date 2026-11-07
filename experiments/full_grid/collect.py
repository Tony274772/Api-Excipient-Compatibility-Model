"""Collect the full grid into one table.

    python experiments/full_grid/collect.py
    python experiments/full_grid/collect.py --csv   # just write the file, no printing

Reads experiments/results/full_grid/<run>/{val,test}_metrics.json and run_info.json, and writes
experiments/results/full_grid/summary.csv ranked by validation PR-AUC. "test" is the 22 external
held-out pairs. Runs that have not finished are listed as missing.
"""

import argparse
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.full_grid.grid import GRID, ORDER  # noqa: E402
from experiments.full_grid.run_grid import RESULTS  # noqa: E402

KEYS = ["pr_auc", "roc_auc", "mcc", "f1", "precision", "recall", "accuracy"]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser(description="Collect the full grid.")
    ap.add_argument("--csv", action="store_true", help="write the file without printing the table")
    args = ap.parse_args()

    rows, missing = [], []
    for name in ORDER:
        d = os.path.join(RESULTS, name)
        val, test, info = load(f"{d}/val_metrics.json"), load(f"{d}/test_metrics.json"), load(f"{d}/run_info.json")
        if not val:
            missing.append(name)
            continue
        g = GRID[name]
        r = {"run": name, "encoder": g["encoder"], "source": g["source"] or "", "fusion": g["fusion"],
             "pooling": g["pooling"], "loss": g["loss"], "head": "PGB" if g["pgb"] else "baseline"}
        for k in KEYS:
            r[f"val_{k}"] = round(float(val[k]), 4) if k in val else None
        cm = val.get("confusion_matrix")
        if cm:
            r["val_correct"] = cm[0][0] + cm[1][1]
        if test:
            for k in KEYS:
                r[f"heldout_{k}"] = round(float(test[k]), 4) if k in test else None
            cm = test.get("confusion_matrix")
            if cm:
                r["heldout_correct"] = f"{cm[0][0] + cm[1][1]}/{sum(map(sum, cm))}"
        if info:
            r["minutes"] = info.get("minutes")
            r["parameters"] = info.get("parameters")
        rows.append(r)

    if not rows:
        raise SystemExit(f"no finished runs in {RESULTS}")
    df = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    df.index += 1
    out = os.path.join(RESULTS, "summary.csv")
    df.to_csv(out, index=False)

    if not args.csv:
        show = ["run", "head", "val_pr_auc", "val_roc_auc", "val_mcc", "val_f1", "val_correct",
                "heldout_correct", "heldout_pr_auc", "minutes"]
        pd.set_option("display.width", 250)
        print(df[[c for c in show if c in df]].to_string())
        print(f"\n{len(df)} finished of {len(ORDER)}")
        if missing:
            print(f"missing ({len(missing)}): {', '.join(missing[:8])}" + (" ..." if len(missing) > 8 else ""))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
