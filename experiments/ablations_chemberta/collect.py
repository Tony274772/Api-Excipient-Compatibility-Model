"""Collect the chemberta__x_pubchemfp ablations and apply the pre-registered decision rule.

Plan: paper/chemberta_pubchemfp_ablation_plan.md, section 5. Fixed before any run:

  1. Each ablation vs its reference (`full`, or variants.REFERENCE) with a paired t-test over the
     seeds both share, on val PR-AUC; Benjamini-Hochberg over all ablations at q = 0.10.
     Reported with the mean difference, its 95% CI and the smallest difference n seeds can detect.
  2. NEEDED     removing the part lowers val PR-AUC significantly after BH.
  3. REMOVABLE  mean val PR-AUC drop <= 0.01 AND mean val MCC drop <= 0.02.
     (HELPS: removing it raises val PR-AUC significantly after BH.)  Otherwise UNCLEAR.
  6. Held-out (22 pairs) is not read unless --reveal-heldout is passed, and never enters a rule.

Also:
  - step 0: `full` from this harness against the grid's chemberta__x_pubchemfp, seed by seed
    (|delta val PR-AUC| <= 0.005 expected on the same device)
  - a 5-seed ensemble per ablation: mean probability, threshold from val with the same rule
    (best MCC with TNR >= 0.97)

Writes experiments/results/ablations_chemberta/:
    summary.csv  summary.md  reproduction.csv  multi_seed/<ablation>_<n>seeds/...

    python -m experiments.ablations_chemberta.collect
    python -m experiments.ablations_chemberta.collect --reveal-heldout      # after the decision
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

from experiments.ablations_chemberta import variants as V  # noqa: E402
from experiments.ablations_chemberta.run_ablation import (MODEL_CELL, RESULTS_ROOT,  # noqa: E402
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


def ci95(delta, sd, n):
    if n < 2 or not np.isfinite(sd):
        return (np.nan, np.nan)
    try:
        from scipy import stats
        t = stats.t.ppf(0.975, n - 1)
    except ImportError:
        t = 2.776 if n == 5 else 1.96
    half = t * sd / math.sqrt(n)
    return (delta - half, delta + half)


# --------------------------------------------------------------------------- step 0
def reproduction(df, seeds) -> pd.DataFrame:
    rows = []
    for s in seeds:
        g = os.path.join(GRID_ROOT, f"{MODEL_CELL}_seed{s}", "val_metrics.json")
        mine = df[(df["ablation"] == "full") & (df["seed"] == s)]
        if not os.path.exists(g) or mine.empty:
            rows.append({"seed": s, "grid": np.nan, "harness": np.nan, "delta": np.nan,
                         "ok": False})
            continue
        with open(g) as f:
            grid_pr = json.load(f)["pr_auc"]
        h = float(mine["val_pr_auc"].iloc[0])
        rows.append({"seed": s, "grid": grid_pr, "harness": h, "delta": h - grid_pr,
                     "ok": abs(h - grid_pr) <= REPRO_TOL})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- decision table
def decision_table(df, ablations) -> pd.DataFrame:
    rows = []
    for a in ablations:
        sub = df[df["ablation"] == a]
        if sub.empty:
            rows.append({"ablation": a, "n_seeds": 0})
            continue
        ref = V.REFERENCE.get(a, "full")
        row = {"ablation": a, "reference": ref if a != "full" else "", "n_seeds": len(sub),
               "params": int(sub["params"].iloc[0]),
               "weight_paired": bool(sub["weight_paired"].iloc[0]),
               "description": V.describe(a)}
        for k in ("pr_auc", "mcc", "roc_auc", "f1"):
            row[f"val_{k}"] = sub[f"val_{k}"].mean()
            row[f"val_{k}_sd"] = sub[f"val_{k}"].std(ddof=1) if len(sub) > 1 else np.nan
        if a != "full":
            t_pr = paired_test(series(df, a), series(df, ref))
            t_mcc = paired_test(series(df, a, "val_mcc"), series(df, ref, "val_mcc"))
            row.update({"ref_params": int(df[df["ablation"] == ref]["params"].iloc[0])
                        if not df[df["ablation"] == ref].empty else np.nan,
                        "paired_seeds": t_pr["n_seeds"],
                        "d_pr_auc": t_pr["delta"], "sd_diff": t_pr["sd_diff"],
                        "p_t": t_pr["p_t"], "p_wilcoxon": t_pr["p_wilcoxon"],
                        "d_mcc": t_mcc["delta"]})
            lo, hi = ci95(t_pr["delta"], t_pr["sd_diff"], t_pr["n_seeds"])
            row["ci95_lo"], row["ci95_hi"] = lo, hi
            row["detectable"] = (detectable_delta(t_pr["n_seeds"], t_pr["sd_diff"])
                                 if t_pr["n_seeds"] >= 2 and np.isfinite(t_pr["sd_diff"])
                                 and t_pr["sd_diff"] > 0 else np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    if "p_t" not in out:
        return out

    tested = out["ablation"] != "full"
    adj, reject = benjamini_hochberg(out.loc[tested, "p_t"].tolist(), q=Q_BH)
    out.loc[tested, "p_bh"] = adj
    out.loc[tested, "bh_reject"] = reject

    def verdict(r):
        if r["ablation"] == "full" or not np.isfinite(r.get("d_pr_auc", np.nan)):
            return ""
        if r.get("paired_seeds", 0) < len(SEEDS):
            note = f" ({int(r['paired_seeds'])} seeds)"
        else:
            note = ""
        if r["bh_reject"] and r["d_pr_auc"] < 0:
            return "NEEDED" + note
        if r["bh_reject"] and r["d_pr_auc"] > 0:
            return "HELPS when removed" + note
        if r["d_pr_auc"] >= -REMOVABLE_PR and r["d_mcc"] >= -REMOVABLE_MCC:
            return "REMOVABLE" + note
        return "UNCLEAR" + note

    out["verdict"] = out.apply(verdict, axis=1)
    return out


# --------------------------------------------------------------------------- ensembles
def ensemble(ablation, seeds, reveal: bool):
    """Mean-probability ensemble over the seeds that exist; threshold from val only."""
    have = [s for s in seeds if os.path.exists(os.path.join(RUNS_ROOT, run_name(ablation, s),
                                                            SENTINEL))]
    if len(have) < 2:
        return None
    cfg = Config()
    out_dir = os.path.join(MULTI_ROOT, f"{ablation}_{len(have)}seeds")
    os.makedirs(out_dir, exist_ok=True)

    def table(split):
        base = None
        for s in have:
            root = RUNS_ROOT if split == "val" else SEALED_ROOT
            pred = pd.read_csv(os.path.join(root, run_name(ablation, s), f"predictions_{split}.csv"))
            if base is None:
                base = pred[[c for c in ID_COLS if c in pred]].copy()
            if len(pred) != len(base) or not (pred["label"].values == base["label"].values).all():
                raise SystemExit(f"{ablation}: {split} rows of seed {s} differ from seed {have[0]}")
            base[f"prob_seed{s}"] = pred["probability"].values
        probs = base[[f"prob_seed{s}" for s in have]].values
        base["prob_mean"] = probs.mean(1)
        base["prob_sd"] = probs.std(1, ddof=1)
        return base

    val = table("val")
    thr, _ = tune_threshold(val["prob_mean"].values, val["label"].values, cfg.threshold_step,
                            tnr_floor=cfg.pgb_tnr_floor)
    thr = float(thr)
    val["pred_ensemble"] = (val["prob_mean"] >= thr).astype(int)
    val.to_csv(os.path.join(out_dir, "predictions_val.csv"), index=False)
    m = metric_suite(val["prob_mean"].values, val["label"].values, thr)
    res = {"ablation": ablation, "ens_seeds": len(have), "ens_threshold": thr,
           "ens_val_pr_auc": m["pr_auc"], "ens_val_mcc": m["mcc"], "ens_val_correct": m["correct"]}
    if reveal:
        ho = table("heldout")
        ho["pred_ensemble"] = (ho["prob_mean"] >= thr).astype(int)
        ho.to_csv(os.path.join(out_dir, "predictions_heldout.csv"), index=False)
        hm = metric_suite(ho["prob_mean"].values, ho["label"].values, thr)
        res.update({"ens_heldout_correct": hm["correct"], "ens_heldout_pr_auc": hm["pr_auc"],
                    "ens_heldout_mcc": hm["mcc"]})
    with open(os.path.join(out_dir, "ensemble.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def heldout_per_seed(ablations, seeds) -> pd.DataFrame:
    rows = []
    for a in ablations:
        for s in seeds:
            p = os.path.join(SEALED_ROOT, run_name(a, s), "heldout_metrics.json")
            if os.path.exists(p):
                with open(p) as f:
                    m = json.load(f)
                rows.append({"ablation": a, "seed": s, "heldout_correct": m["correct"],
                             "heldout_pr_auc": m["pr_auc"]})
    if not rows:
        return pd.DataFrame(columns=["ablation"])
    df = pd.DataFrame(rows)
    return df.groupby("ablation").agg(heldout_correct=("heldout_correct", "mean"),
                                      heldout_correct_sd=("heldout_correct", "std"),
                                      heldout_pr_auc=("heldout_pr_auc", "mean")).reset_index()


# --------------------------------------------------------------------------- report
def fmt(x, nd=4):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def markdown(dec, repro, missing, reveal) -> str:
    L = [f"# Ablations of `{MODEL_CELL}`", "",
         "Generated by `python -m experiments.ablations_chemberta.collect`. Decision rule: "
         "paper/chemberta_pubchemfp_ablation_plan.md section 5 "
         f"(BH q = {Q_BH}; removable = drop <= {REMOVABLE_PR} PR-AUC and <= {REMOVABLE_MCC} MCC).",
         ""]
    L += ["## Step 0: harness `full` vs grid", ""]
    if repro.empty or repro["harness"].isna().all():
        L += ["`full` has not been run yet.", ""]
    else:
        L += ["| seed | grid | harness | delta | within 0.005 |", "|---|---|---|---|---|"]
        for _, r in repro.iterrows():
            ok = "missing" if not np.isfinite(r["delta"]) else ("yes" if r["ok"] else "NO")
            L.append(f"| {r['seed']} | {fmt(r['grid'])} | {fmt(r['harness'])} | "
                     f"{fmt(r['delta'], 4)} | {ok} |")
        L.append("")
    L += ["## Ablations (validation; held-out not used)", "",
          "| ablation | params | seeds | val PR-AUC | d PR-AUC [95% CI] | p (BH) | d MCC | "
          "detectable | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in dec.iterrows():
        if r.get("n_seeds", 0) == 0:
            continue
        pr = f"{fmt(r['val_pr_auc'])} ± {fmt(r['val_pr_auc_sd'])}"
        if r["ablation"] == "full":
            L.append(f"| **full** | {int(r['params']):,} | {int(r['n_seeds'])} | {pr} | | | | | |")
            continue
        ref = "" if r["reference"] == "full" else f" vs {r['reference']}"
        star = "" if r["weight_paired"] else " ‡"
        L.append(f"| {r['ablation']}{ref}{star} | {int(r['params']):,} | {int(r['n_seeds'])} | {pr} | "
                 f"{fmt(r['d_pr_auc'])} [{fmt(r['ci95_lo'])}, {fmt(r['ci95_hi'])}] | "
                 f"{fmt(r['p_bh'], 3)} | {fmt(r['d_mcc'])} | {fmt(r['detectable'], 3)} | "
                 f"{r['verdict']} |")
    L += ["", "‡ not weight-paired with full (the head is re-drawn).", ""]
    if missing:
        L += ["## Missing runs", "", ", ".join(missing), ""]
    if not reveal:
        L += ["Held-out is sealed. Run with `--reveal-heldout` only after the decision is made.", ""]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablations", nargs="+", default=None, help="default: variants.PLAN")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--reveal-heldout", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="collect the 1-epoch smoke runs (tests this script; numbers mean nothing)")
    args = p.parse_args()

    global RESULTS_ROOT, RUNS_ROOT, SEALED_ROOT, MULTI_ROOT
    if args.smoke:
        from experiments.ablations_chemberta.run_ablation import SMOKE_ROOT
        RESULTS_ROOT = SMOKE_ROOT
        RUNS_ROOT = os.path.join(SMOKE_ROOT, "runs")
        SEALED_ROOT = os.path.join(SMOKE_ROOT, "_heldout_sealed")
        MULTI_ROOT = os.path.join(SMOKE_ROOT, "multi_seed")

    ablations = args.ablations or V.PLAN
    if "full" not in ablations:
        ablations = ["full"] + ablations
    df = read_runs(ablations, args.seeds)
    missing = [run_name(a, s) for a in ablations for s in args.seeds
               if df.empty or df[(df["ablation"] == a) & (df["seed"] == s)].empty]
    if df.empty:
        raise SystemExit("no COMPLETE runs found under " + RUNS_ROOT)

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    repro = reproduction(df, args.seeds)
    repro.to_csv(os.path.join(RESULTS_ROOT, "reproduction.csv"), index=False)

    dec = decision_table(df, ablations)
    ens = [e for e in (ensemble(a, args.seeds, args.reveal_heldout) for a in ablations) if e]
    if ens:
        dec = dec.merge(pd.DataFrame(ens), on="ablation", how="left")
    if args.reveal_heldout:
        dec = dec.merge(heldout_per_seed(ablations, args.seeds), on="ablation", how="left")
    df.to_csv(os.path.join(RESULTS_ROOT, "per_seed.csv"), index=False)
    dec.to_csv(os.path.join(RESULTS_ROOT, "summary.csv"), index=False)
    md = markdown(dec, repro, missing, args.reveal_heldout)
    if args.smoke:
        md = ("> SMOKE RUNS (1 epoch): this only tests the scripts. Step 0 compares 1-epoch runs "
              "with 40-epoch grid runs and is expected to fail.\n\n" + md)
    with open(os.path.join(RESULTS_ROOT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(md)
    print(f"\nwritten: {RESULTS_ROOT}/summary.csv, summary.md, per_seed.csv, reproduction.csv, "
          f"multi_seed/")


if __name__ == "__main__":
    main()
