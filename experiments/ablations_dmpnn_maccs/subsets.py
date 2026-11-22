"""Chemical subgroup diagnostics for dmpnn__x_maccs ablations (evaluation only).

Evaluates whether the D-MPNN + MACCS combination specifically resolves chemical classes
that challenge molecular encoders:
    1. clean_organic: single-fragment neutral molecules
    2. salts_and_ions: multi-fragment salts, hydrates, or charged ions
    3. sugars_polyols: polyhydroxy excipients (glucose, lactose, mannitol, sorbitol)

Usage:
    python -m experiments.ablations_dmpnn_maccs.subsets
    python -m experiments.ablations_dmpnn_maccs.subsets --ablations full no_secondary dmpnn_parent
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "results", "ablations_dmpnn_maccs")
RUNS_ROOT = os.path.join(RESULTS_ROOT, "runs")
SEEDS = [42, 43, 44, 45, 46]
SENTINEL = "COMPLETE"


def run_name(ablation: str, seed: int) -> str:
    return f"{ablation}_seed{seed}"


OUT_DIR = os.path.join(RESULTS_ROOT, "chemical_subsets")
SUBSETS = ["clean_organic", "salts_and_ions", "sugars_polyols", "all"]
BASELINE = "no_secondary"


def is_sugar_or_polyol(smiles: str) -> bool:
    """Identify carbohydrates / polyols (at least 3 OH groups and no aromatic rings)."""
    s = str(smiles)
    # Common sugar / polyol substrings or patterns
    if "c" in s:  # aromatic carbon
        return False
    oh_count = s.count("O")
    c_count = s.count("C")
    return oh_count >= 3 and c_count >= 3


def is_salt_or_ion(smiles: str) -> bool:
    """Identify salts, hydrates, or charged counterions."""
    s = str(smiles)
    return "." in s or "+" in s or "-" in s


def pair_class(api: str, exc: str) -> str:
    if is_salt_or_ion(api) or is_salt_or_ion(exc):
        return "salts_and_ions"
    if is_sugar_or_polyol(api) or is_sugar_or_polyol(exc):
        return "sugars_polyols"
    return "clean_organic"


def score_subset(df_preds: pd.DataFrame, mask: pd.Series, threshold: float) -> dict:
    sub = df_preds[mask]
    n = len(sub)
    n_pos = int(sub["label"].sum())
    if n == 0 or n_pos == 0 or n_pos == n:
        return {"n": n, "n_pos": n_pos, "pr_auc": np.nan, "mcc": np.nan}
    pr = float(average_precision_score(sub["label"], sub["prob"]))
    pred_bin = (sub["prob"] >= threshold).astype(int)
    mcc = float(matthews_corrcoef(sub["label"], pred_bin))
    return {"n": n, "n_pos": n_pos, "pr_auc": pr, "mcc": mcc}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ablations", nargs="+", default=["full", "no_secondary", "no_dmpnn", "dmpnn_parent"])
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = p.parse_args()

    from experiments.encoder_comparison.data import SplitPaths
    paths = SplitPaths(None, "api_cluster")
    val_df = pd.read_csv(paths.val)
    val_classes = [pair_class(r["API_Smiles"], r["Excipient_Smiles"]) for _, r in val_df.iterrows()]
    val_df["subset"] = val_classes

    print("=" * 90)
    print("Validation set chemical subset distribution:")
    for sub, count in val_df["subset"].value_counts().items():
        pos = val_df[val_df["subset"] == sub]["label"].sum()
        print(f"  {sub:20s}: {count:4d} pairs ({pos:3d} positive, {pos/count*100:.1f}%)")
    print("=" * 90)

    rows = []
    for a in args.ablations:
        for s in args.seeds:
            d = os.path.join(RUNS_ROOT, run_name(a, s))
            preds_path = os.path.join(d, "predictions_val.csv")
            info_path = os.path.join(d, "run_info.json")
            if not os.path.exists(preds_path) or not os.path.exists(info_path):
                continue
            preds = pd.read_csv(preds_path)
            with open(info_path) as f:
                th = json.load(f)["threshold_from_val"]

            for sub in SUBSETS:
                mask = pd.Series(True, index=preds.index) if sub == "all" else (val_df["subset"] == sub)
                res = score_subset(preds, mask, th)
                rows.append({"ablation": a, "seed": s, "subset": sub, "n": res["n"],
                             "n_pos": res["n_pos"], "pr_auc": res["pr_auc"], "mcc": res["mcc"]})

    if not rows:
        print("No predictions found to evaluate. Run ablations first.")
        return

    res_df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    res_df.to_csv(os.path.join(OUT_DIR, "subset_scores_per_seed.csv"), index=False)

    summary = res_df.groupby(["ablation", "subset"]).agg(
        n=("n", "first"), n_pos=("n_pos", "first"),
        pr_mean=("pr_auc", "mean"), pr_sd=("pr_auc", "std"),
        mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std")
    ).reset_index()

    print("\n## Chemical Subgroup Results (Mean over seeds):\n")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUT_DIR, "subset_summary.csv"), index=False)
    print(f"\nSaved diagnostics to {OUT_DIR}/")


if __name__ == "__main__":
    main()
