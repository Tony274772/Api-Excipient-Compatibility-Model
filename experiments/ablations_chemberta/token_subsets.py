"""Section E of the plan: does ChemBERTa's blind spot hurt the predictions? (evaluation only)

ChemBERTa-77M-MTR has untrained token embeddings for '.', bracket atoms ([Na+], [O-], [C@@H]),
'/', '\\' and it misreads Cl / Br (audit_pretrained.py). Each validation pair is put in one subset:

    clean        neither molecule contains any of those
    fixable      at least one does, but only salt / charge / stereo characters (no Cl, Br):
                 chemberta_parent can remove them
    halogen      at least one molecule contains Cl or Br (with or without salt characters):
                 chemberta_parent cannot remove those

For every ablation given (default full, no_chemberta, chemberta_parent) and every seed, PR-AUC and
MCC (at that run's own val threshold) are computed per subset, then mean +- sd over seeds. Each
model is compared with no_chemberta subset by subset, paired over seeds: what ChemBERTa adds where
it can read the molecule vs where it cannot. The subsets differ in difficulty, so only the
within-subset comparison between models means anything, not the raw gap between subsets.

Validation only. Held-out is not read.

    python -m experiments.ablations_chemberta.token_subsets
    python -m experiments.ablations_chemberta.token_subsets --ablations full no_chemberta
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.ablations_chemberta.parent_smiles import FIXABLE_CHARS, HALOGENS  # noqa: E402
from experiments.ablations_chemberta.run_ablation import (RESULTS_ROOT, RUNS_ROOT, SEEDS,  # noqa: E402
                                                          SENTINEL, run_name)
from experiments.encoder_comparison.collect_grid import paired_test  # noqa: E402
from experiments.encoder_comparison.data import SplitPaths  # noqa: E402

OUT_DIR = os.path.join(RESULTS_ROOT, "token_subsets")
SUBSETS = ["clean", "fixable", "halogen", "affected (fixable + halogen)", "all"]
BASELINE = "no_chemberta"


def molecule_class(smiles: str) -> str:
    s = str(smiles)
    if any(h in s for h in HALOGENS):
        return "halogen"
    if any(c in s for c in FIXABLE_CHARS):
        return "fixable"
    return "clean"


def pair_subset(api: str, exc: str) -> str:
    classes = {molecule_class(api), molecule_class(exc)}
    if "halogen" in classes:
        return "halogen"
    if "fixable" in classes:
        return "fixable"
    return "clean"


def val_subsets(paths: SplitPaths) -> pd.DataFrame:
    val = pd.read_csv(paths.val)
    out = val[["API_CID", "Excipient_CID", "Outcome1"]].copy()
    out["subset"] = [pair_subset(a, e) for a, e in zip(val["API_Smiles"], val["Excipient_Smiles"])]
    return out


def scores(pred: pd.DataFrame, mask: np.ndarray) -> dict:
    y, p, yhat = pred["label"].values[mask], pred["probability"].values[mask], pred["prediction"].values[mask]
    n_pos = int(y.sum())
    pr = float(average_precision_score(y, p)) if 0 < n_pos < len(y) else np.nan
    mcc = float(matthews_corrcoef(y, yhat)) if len(set(y)) > 1 else np.nan
    return {"n": int(mask.sum()), "n_pos": n_pos, "pr_auc": pr, "mcc": mcc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations", nargs="+", default=["full", BASELINE, "chemberta_parent"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--smoke", action="store_true", help="read the 1-epoch smoke runs (script test)")
    args = ap.parse_args()

    global RUNS_ROOT, OUT_DIR
    if args.smoke:
        from experiments.ablations_chemberta.run_ablation import SMOKE_ROOT
        RUNS_ROOT = os.path.join(SMOKE_ROOT, "runs")
        OUT_DIR = os.path.join(SMOKE_ROOT, "token_subsets")

    subs = val_subsets(SplitPaths(None, "api_cluster"))
    masks = {"clean": subs["subset"].values == "clean",
             "fixable": subs["subset"].values == "fixable",
             "halogen": subs["subset"].values == "halogen"}
    masks["affected (fixable + halogen)"] = masks["fixable"] | masks["halogen"]
    masks["all"] = np.ones(len(subs), dtype=bool)

    rows = []
    for a in args.ablations:
        for s in args.seeds:
            d = os.path.join(RUNS_ROOT, run_name(a, s))
            if not os.path.exists(os.path.join(d, SENTINEL)):
                continue
            pred = pd.read_csv(os.path.join(d, "predictions_val.csv"))
            if len(pred) != len(subs) or not (
                    (pred["API_CID"].values == subs["API_CID"].values).all()
                    and (pred["Excipient_CID"].values == subs["Excipient_CID"].values).all()
                    and (pred["label"].values == subs["Outcome1"].values).all()):
                raise SystemExit(f"{run_name(a, s)}: predictions_val.csv rows do not match val.csv")
            for name, m in masks.items():
                rows.append({"ablation": a, "seed": s, "subset": name, **scores(pred, m)})
    if not rows:
        raise SystemExit("none of the requested runs is COMPLETE yet")
    per_seed = pd.DataFrame(rows)

    summary = (per_seed.groupby(["subset", "ablation"])
               .agg(seeds=("seed", "nunique"), n=("n", "first"), n_pos=("n_pos", "first"),
                    pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
                    mcc=("mcc", "mean"), mcc_sd=("mcc", "std"))
               .reset_index())

    comp = []
    if BASELINE in per_seed["ablation"].values:
        for name in SUBSETS:
            sub = per_seed[per_seed["subset"] == name]
            base = dict(zip(sub[sub["ablation"] == BASELINE]["seed"],
                            sub[sub["ablation"] == BASELINE]["pr_auc"]))
            for a in args.ablations:
                if a == BASELINE:
                    continue
                mine = dict(zip(sub[sub["ablation"] == a]["seed"], sub[sub["ablation"] == a]["pr_auc"]))
                t = paired_test(mine, base)
                comp.append({"subset": name, "ablation": a, "vs": BASELINE,
                             "seeds": t["n_seeds"], "d_pr_auc": t["delta"], "p_t": t["p_t"]})
    comp = pd.DataFrame(comp)

    os.makedirs(OUT_DIR, exist_ok=True)
    subs.to_csv(os.path.join(OUT_DIR, "val_pair_subsets.csv"), index=False)
    per_seed.to_csv(os.path.join(OUT_DIR, "per_seed.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    comp.to_csv(os.path.join(OUT_DIR, "vs_no_chemberta.csv"), index=False)

    pd.set_option("display.width", 200)
    counts = subs.groupby("subset")["Outcome1"].agg(pairs="size", positives="sum")
    print("validation pairs per subset:\n" + counts.to_string() + "\n")
    print(summary.round(4).to_string(index=False))
    if not comp.empty:
        print("\nPR-AUC difference from no_chemberta, paired over seeds:")
        print(comp.round(4).to_string(index=False))
    small = counts[counts["positives"] < 15]
    if len(small):
        print(f"\nNOTE: few positives in {', '.join(small.index)} - read those PR-AUCs as rough.")
    print(f"\nwritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
