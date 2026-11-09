"""Collect the reagent ablation: control vs no_reagents, mean +/- std over seeds, paired deltas.

    python experiments/reagent_ablation/collect.py

Three views per run, all at the threshold the run tuned on its own val set:

    heldout_*   the 22 external pairs: identical in both arms, the main comparison
    valclean_*  val rows WITHOUT reagents (the same 758 rows in both arms): fair val comparison
    val_*       the run's own val set (control 774 rows incl. reagents, no_reagents 758): as reported

plus, for the control arm only, how the model does on the reagent rows it was validated on
(reagent_recall: share of incompatible reagent rows caught).

Writes experiments/results/reagent_ablation/per_seed.csv, summary.csv and delta.csv (per-seed paired
difference no_reagents - control). The seed-42..46 top5_seeds_depth6 numbers are shown as a reference
for the control arm when that folder exists.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
                             roc_auc_score)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.reagent_ablation.reagents import REAGENTS  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "experiments", "results", "reagent_ablation")
REF = os.path.join(PROJECT_ROOT, "experiments", "results", "top5_seeds_depth6")
VIEWS = ["heldout", "valclean", "val"]
KEYS = ["pr_auc", "roc_auc", "mcc", "f1", "accuracy", "correct"]


def metrics(df):
    y, p, yhat = df["label"].to_numpy(), df["probability"].to_numpy(), df["prediction"].to_numpy()
    two = len(np.unique(y)) == 2
    return {"pr_auc": average_precision_score(y, p) if two else np.nan,
            "roc_auc": roc_auc_score(y, p) if two else np.nan,
            "mcc": matthews_corrcoef(y, yhat), "f1": f1_score(y, yhat, zero_division=0),
            "accuracy": accuracy_score(y, yhat), "correct": int((y == yhat).sum()), "n": len(df)}


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(OUT, "*", "seed_*", "*"))):
        if not os.path.exists(os.path.join(d, "run_info.json")):
            continue
        info = json.load(open(os.path.join(d, "run_info.json")))
        val = pd.read_csv(os.path.join(d, "predictions_val.csv"))
        held = pd.read_csv(os.path.join(d, "predictions_heldout.csv"))
        reag = val["Excipient_CID"].isin(REAGENTS)
        r = {"arm": info["arm"], "run": info["run"], "seed": info["seed"],
             "threshold": json.load(open(os.path.join(d, "val_metrics.json")))["threshold"]}
        for view, df in (("heldout", held), ("valclean", val[~reag]), ("val", val)):
            for k, v in metrics(df).items():
                r[f"{view}_{k}"] = v
        pos = val[reag & (val["label"] == 1)]
        r["reagent_rows_in_val"] = int(reag.sum())
        r["reagent_recall"] = pos["prediction"].mean() if len(pos) else np.nan
        rows.append(r)
    if not rows:
        raise SystemExit(f"no finished runs under {OUT}")

    per = pd.DataFrame(rows).sort_values(["run", "arm", "seed"]).reset_index(drop=True)
    per.to_csv(os.path.join(OUT, "per_seed.csv"), index=False)

    cols = [f"{v}_{k}" for v in VIEWS for k in KEYS] + ["reagent_recall"]
    g = per.groupby(["run", "arm"])
    summary = g[cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "seeds", g["seed"].apply(lambda s: " ".join(map(str, sorted(s)))))
    summary.insert(0, "n_seeds", g.size())
    summary = summary.reset_index()
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    # paired by seed: no_reagents - control
    piv = per.set_index(["run", "seed", "arm"])[cols[:-1]].unstack("arm")
    deltas = []
    if {"control", "no_reagents"} <= set(per["arm"]):
        delta = (piv.xs("no_reagents", axis=1, level="arm") - piv.xs("control", axis=1, level="arm")).dropna(how="all")
        delta.reset_index().to_csv(os.path.join(OUT, "delta.csv"), index=False)
        deltas = delta.groupby("run").agg(["mean", "std", "count"])

    pd.set_option("display.width", 250)
    show = ["heldout_pr_auc", "heldout_roc_auc", "heldout_mcc", "heldout_correct",
            "valclean_pr_auc", "valclean_roc_auc", "valclean_mcc", "val_pr_auc", "reagent_recall"]
    t = summary[["run", "arm", "n_seeds"]].copy()
    for m in show:
        dg = 1 if m.endswith("correct") else 4
        t[m] = [f"{a:.{dg}f} +/- {b:.{dg}f}" if pd.notna(b) else (f"{a:.{dg}f}" if pd.notna(a) else "-")
                for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
    print("mean +/- std over seeds (heldout = 22 pairs; valclean = val without reagent rows, same rows both arms)\n")
    print(t.T.to_string(header=False))

    if len(deltas):
        print("\npaired difference no_reagents - control (mean +/- std over seeds, n = paired seeds):")
        for run in deltas.index:
            print(f"  {run}")
            for m in show[:-1]:
                mu, sd, n = deltas.loc[run, (m, "mean")], deltas.loc[run, (m, "std")], deltas.loc[run, (m, "count")]
                print(f"    {m:18s} {mu:+.4f} +/- {sd:.4f}  (n={int(n)})")

    ref = os.path.join(REF, "summary.csv")
    if os.path.exists(ref):
        r = pd.read_csv(ref)
        r = r[r["run"].isin(per["run"])]
        if len(r):
            print("\nreference, top5_seeds_depth6 (control should be close):")
            for _, x in r.iterrows():
                print(f"  {x['run']}: val PR-AUC {x['val_pr_auc_mean']:.4f} +/- {x['val_pr_auc_std']:.4f} | "
                      f"held-out PR-AUC {x['heldout_pr_auc_mean']:.4f} +/- {x['heldout_pr_auc_std']:.4f} | "
                      f"held-out correct {x['heldout_correct_mean']:.1f} +/- {x['heldout_correct_std']:.1f}")
    incomplete = summary[summary["n_seeds"] < 5]
    if len(incomplete):
        print("\nfewer than 5 seeds: " + ", ".join(f"{a} {b}" for a, b in zip(incomplete["run"], incomplete["arm"])))
    print(f"\nwritten to {os.path.relpath(OUT, PROJECT_ROOT)}/ (per_seed.csv, summary.csv, delta.csv)")


if __name__ == "__main__":
    main()
