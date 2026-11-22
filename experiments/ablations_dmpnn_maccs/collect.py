"""Collect the dmpnn__x_maccs ablations and apply the pre-registered decision rule.

Plan: paper/dmpnn_maccs_ablation_plan.md, section 5. Fixed before any run:

  1. Each ablation vs its reference (`full`, or variants.REFERENCE) with a paired t-test over the
     seeds both share, on val PR-AUC; Benjamini-Hochberg over all ablations at q = 0.10.
     Reported with the mean difference, its 95% CI and the smallest difference n seeds can detect.
  2. NEEDED     removing the part lowers val PR-AUC significantly after BH.
  3. REMOVABLE  mean val PR-AUC drop <= 0.01 AND mean val MCC drop <= 0.02.
     (HELPS: removing it raises val PR-AUC significantly after BH.) Otherwise UNCLEAR.
  6. Held-out (22 pairs) is not read unless --reveal-heldout is passed, and never enters a rule.

Also:
  - step 0: `full` from this harness against the grid's dmpnn__x_maccs, seed by seed
    (|delta val PR-AUC| <= 0.005 expected on the same device)
  - a 5-seed ensemble per ablation: mean probability, threshold from val with the same rule
    (best MCC with TNR >= 0.97)

Writes experiments/results/ablations_dmpnn_maccs/:
    summary.csv  summary.md  reproduction.csv  multi_seed/<ablation>_<n>seeds/...

    python -m experiments.ablations_dmpnn_maccs.collect
    python -m experiments.ablations_dmpnn_maccs.collect --reveal-heldout      # after the decision
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.ablations_dmpnn_maccs import variants as V  # noqa: E402
from experiments.ablations_dmpnn_maccs.run_ablation import (MODEL_CELL, RESULTS_ROOT,  # noqa: E402
                                                           RUNS_ROOT, SEALED_ROOT, SEEDS, SENTINEL,
                                                           run_name)
from experiments.encoder_comparison.collect_grid import (benjamini_hochberg,  # noqa: E402
                                                         detectable_delta, paired_test)
from experiments.encoder_comparison.run_grid import metric_suite  # noqa: E402
from src.config import Config  # noqa: E402
from src.evaluate import tune_threshold  # noqa: E402

GRID_ROOT = os.path.join(PROJECT_ROOT, "experiments", "results", "encoder_grid")
MULTI_ROOT = os.path.join(RESULTS_ROOT, "multi_seed")

Q_BH = 0.10
REMOVABLE_PR = 0.01
REMOVABLE_MCC = 0.02
REPRO_TOL = 0.005
VAL_KEYS = ["pr_auc", "roc_auc", "mcc", "f1", "precision", "recall", "accuracy", "correct"]
ID_COLS = ["Api_name", "Excipient_name", "API_CID", "Excipient_CID", "label"]


# --------------------------------------------------------------------------- reading
def read_runs(ablations, seeds) -> pd.DataFrame:
    rows = []
    for a in ablations:
        for s in seeds:
            d = os.path.join(RUNS_ROOT, run_name(a, s))
            if not os.path.exists(os.path.join(d, SENTINEL)):
                continue
            with open(os.path.join(d, "val_metrics.json")) as f:
                m = json.load(f)
            with open(os.path.join(d, "run_info.json")) as f:
                info = json.load(f)
            row = {"ablation": a, "seed": s, "params": info["parameters_trainable"],
                   "epochs": info["epochs_trained"], "threshold": info["threshold_from_val"],
                   "weight_paired": info.get("weight_paired_with_full", True)}
            row.update({f"val_{k}": m.get(k) for k in VAL_KEYS})
            rows.append(row)
    return pd.DataFrame(rows)


def series(df, ablation, col="val_pr_auc") -> dict:
    sub = df[df["ablation"] == ablation]
    return dict(zip(sub["seed"], sub[col]))


# --------------------------------------------------------------------------- step 0
def check_reproduction(df: pd.DataFrame) -> pd.DataFrame:
    """Compare `full` from this harness against the grid baseline seed by seed."""
    grid_rows = []
    for s in SEEDS:
        p = os.path.join(GRID_ROOT, f"{MODEL_CELL}_seed{s}", "val_metrics.json")
        if os.path.exists(p):
            with open(p) as f:
                m = json.load(f)
            grid_rows.append({"seed": s, "grid_pr_auc": m["pr_auc"], "grid_mcc": m["mcc"],
                              "grid_roc_auc": m["roc_auc"], "grid_f1": m["f1"]})
    if not grid_rows:
        print(f"[step 0] WARNING: no grid runs found under {GRID_ROOT}/{MODEL_CELL}_seed*")
        return pd.DataFrame()
    gdf = pd.DataFrame(grid_rows)
    fdf = df[df["ablation"] == "full"][["seed", "val_pr_auc", "val_mcc", "val_roc_auc", "val_f1"]].rename(
        columns={"val_pr_auc": "harness_pr_auc", "val_mcc": "harness_mcc",
                 "val_roc_auc": "harness_roc_auc", "val_f1": "harness_f1"})
    merged = pd.merge(gdf, fdf, on="seed", how="left")
    merged["delta_pr_auc"] = merged["harness_pr_auc"] - merged["grid_pr_auc"]
    merged["delta_mcc"] = merged["harness_mcc"] - merged["grid_mcc"]
    merged["ok"] = merged["delta_pr_auc"].abs() <= REPRO_TOL
    return merged


# --------------------------------------------------------------------------- ensembling
def ensemble_predictions(split_dir, name, seeds):
    probs, labels, base = [], None, None
    for s in seeds:
        csv = os.path.join(split_dir, run_name(name, s), "predictions_val.csv")
        if not os.path.exists(csv):
            return None, None
        df = pd.read_csv(csv)
        probs.append(df["prob"].values)
        if labels is None:
            labels = df["label"].values
            base = df[[c for c in ID_COLS if c in df.columns]].copy()
    mean_prob = np.mean(probs, axis=0)
    return mean_prob, labels


def ensemble_heldout(split_dir, name, seeds):
    probs, labels, base = [], None, None
    for s in seeds:
        csv = os.path.join(split_dir, run_name(name, s), "predictions_heldout.csv")
        if not os.path.exists(csv):
            return None, None
        df = pd.read_csv(csv)
        probs.append(df["prob"].values)
        if labels is None:
            labels = df["label"].values
            base = df[[c for c in ID_COLS if c in df.columns]].copy()
    mean_prob = np.mean(probs, axis=0)
    return mean_prob, labels


# --------------------------------------------------------------------------- collect
def collect(ablations=None, reveal_heldout=False):
    ablations = ablations or V.PLAN
    for a in ablations:
        V.parse_ablation(a)
    df = read_runs(ablations, SEEDS)
    if df.empty:
        print(f"No completed runs found under {RUNS_ROOT}")
        return

    print("=" * 100)
    print(f"Collecting {MODEL_CELL} ablations | {len(df)} runs found across {df['ablation'].nunique()} variants")
    print("=" * 100)

    # step 0
    repro = check_reproduction(df)
    if not repro.empty:
        print("\n## Step 0: Reproduction of grid baseline\n")
        print(repro.to_string(index=False))
        if not repro["ok"].all():
            bad = repro[~repro["ok"]]
            print(f"\nWARNING: Step 0 tolerance ({REPRO_TOL}) exceeded on seed(s): {bad['seed'].tolist()}")
        os.makedirs(RESULTS_ROOT, exist_ok=True)
        repro.to_csv(os.path.join(RESULTS_ROOT, "reproduction.csv"), index=False)

    # paired tests
    full_pr = series(df, "full", "val_pr_auc")
    full_mcc = series(df, "full", "val_mcc")
    full_p = df[df["ablation"] == "full"]["params"].iloc[0] if "full" in df["ablation"].values else None

    records = []
    pvals = []
    for a in ablations:
        sub = df[df["ablation"] == a]
        n_seeds = len(sub)
        if n_seeds == 0:
            continue
        ref = V.REFERENCE.get(a, "full")
        ref_pr = series(df, ref, "val_pr_auc")
        ref_mcc = series(df, ref, "val_mcc")
        common = sorted(set(sub["seed"]) & set(ref_pr))
        a_pr = [sub[sub["seed"] == s]["val_pr_auc"].iloc[0] for s in common]
        r_pr = [ref_pr[s] for s in common]
        a_mcc = [sub[sub["seed"] == s]["val_mcc"].iloc[0] for s in common]
        r_mcc = [ref_mcc[s] for s in common]

        t_res = paired_test(a_pr, r_pr)
        delta_pr = t_res["delta_mean"]
        pval = t_res["p_value"]
        delta_mcc = float(np.mean(np.array(a_mcc) - np.array(r_mcc))) if common else math.nan

        p = sub["params"].iloc[0]
        rec = {
            "ablation": a, "reference": ref, "n_seeds": n_seeds, "common_seeds": len(common),
            "val_pr_auc": float(sub["val_pr_auc"].mean()), "val_pr_sd": float(sub["val_pr_auc"].std()),
            "val_mcc": float(sub["val_mcc"].mean()), "val_mcc_sd": float(sub["val_mcc"].std()),
            "val_roc_auc": float(sub["val_roc_auc"].mean()), "val_f1": float(sub["val_f1"].mean()),
            "val_precision": float(sub["val_precision"].mean()), "val_recall": float(sub["val_recall"].mean()),
            "val_accuracy": float(sub["val_accuracy"].mean()), "val_correct": float(sub["val_correct"].mean()),
            "delta_pr_auc": delta_pr, "p_raw": pval, "delta_mcc": delta_mcc,
            "ci95_low": t_res["ci95_low"], "ci95_high": t_res["ci95_high"],
            "trainable_params": p,
            "delta_params": (p - full_p) if full_p is not None else math.nan,
            "weight_paired": sub["weight_paired"].all(), "description": V.describe(a)
        }
        records.append(rec)
        if a != "full" and not math.isnan(pval):
            pvals.append(pval)

    # BH correction
    p_bh = benjamini_hochberg(pvals, Q_BH) if pvals else []
    bh_iter = iter(p_bh)
    for r in records:
        if r["ablation"] == "full" or math.isnan(r["p_raw"]):
            r["p_bh"] = math.nan
            r["sig_bh"] = False
        else:
            p_adj, sig = next(bh_iter)
            r["p_bh"] = p_adj
            r["sig_bh"] = sig

        # verdict
        if r["ablation"] == "full":
            r["verdict"] = "BASELINE"
        elif r["sig_bh"] and r["delta_pr_auc"] < 0:
            r["verdict"] = "NEEDED"
        elif r["sig_bh"] and r["delta_pr_auc"] > 0:
            r["verdict"] = "HELPS (removal improves)"
        elif r["delta_pr_auc"] >= -REMOVABLE_PR and r["delta_mcc"] >= -REMOVABLE_MCC:
            r["verdict"] = "REMOVABLE"
        else:
            r["verdict"] = "UNCLEAR"

    summary_df = pd.DataFrame(records)

    # 5-seed ensembling
    ens_rows = []
    cfg = Config()
    for a in ablations:
        avail_seeds = df[df["ablation"] == a]["seed"].tolist()
        if len(avail_seeds) < 2:
            continue
        p_val, l_val = ensemble_predictions(RUNS_ROOT, a, avail_seeds)
        if p_val is not None:
            th, _ = tune_threshold(p_val, l_val, cfg.threshold_step, tnr_floor=cfg.pgb_tnr_floor)
            th = float(th)
            val_m = metric_suite(p_val, l_val, th)
            ho_m = None
            if reveal_heldout:
                p_ho, l_ho = ensemble_heldout(SEALED_ROOT, a, avail_seeds)
                if p_ho is not None:
                    ho_m = metric_suite(p_ho, l_ho, th)
            ens_rows.append({
                "ablation": a, "seeds_ensembled": len(avail_seeds), "ens_threshold": th,
                "ens_val_pr_auc": val_m["pr_auc"], "ens_val_mcc": val_m["mcc"],
                "ens_val_correct": val_m["correct"],
                "ens_heldout_correct": ho_m["correct"] if ho_m else math.nan,
                "ens_heldout_mcc": ho_m["mcc"] if ho_m else math.nan
            })
    if ens_rows:
        ens_df = pd.DataFrame(ens_rows)
        summary_df = pd.merge(summary_df, ens_df, on="ablation", how="left")

    print("\n## Ablation Results Summary (paired vs reference)\n")
    cols_show = ["ablation", "val_pr_auc", "delta_pr_auc", "val_mcc", "delta_mcc",
                 "p_raw", "p_bh", "verdict", "trainable_params"]
    print(summary_df[cols_show].to_string(index=False))

    summary_df.to_csv(os.path.join(RESULTS_ROOT, "summary.csv"), index=False)
    with open(os.path.join(RESULTS_ROOT, "summary.md"), "w") as f:
        f.write("# `dmpnn__x_maccs` Ablation Study Results\n\n")
        f.write(summary_df.to_markdown(index=False) if hasattr(summary_df, "to_markdown") else summary_df.to_string())
    print(f"\nSaved summary to {RESULTS_ROOT}/summary.csv and summary.md")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ablations", nargs="+", default=None)
    p.add_argument("--reveal-heldout", action="store_true", help="evaluate on sealed held-out test pairs")
    args = p.parse_args()
    collect(args.ablations, reveal_heldout=args.reveal_heldout)


if __name__ == "__main__":
    main()

