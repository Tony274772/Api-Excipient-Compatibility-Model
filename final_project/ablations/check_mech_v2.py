"""
Check the mechanism flags on train and val: how often each rule fires, the incompatible rate when it
fires (precision) against the split's base rate (lift), and the share of incompatible pairs it covers
(recall). Compares v1 (src/lookup.py) with v2 (ablations/mech_v2.py).

The held-out pairs are never read here: rules are judged on train / val only.

Usage (from final_project/):
    python ablations/check_mech_v2.py
    python ablations/check_mech_v2.py --examples 5     # also list the most common pairs per v2 rule
    python ablations/check_mech_v2.py --unused         # also score the rules left out of v2
"""

import argparse
import os
import sys

FINAL_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FINAL_PROJECT)
os.chdir(FINAL_PROJECT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ablations.mech_v2 import MECH_V2_NAMES, RULES, RULES_NOT_USED, mechanism_flags_v2, properties  # noqa: E402
from src.lookup import get_mechanism_matches, get_mechanism_names  # noqa: E402

SPLITS = {"train": "data/train.csv", "val": "data/val.csv"}


def flag_table(df, version):
    if version == "unused":
        return np.array([[int(bool(rule(properties(a), properties(e)))) for _, _, rule in RULES_NOT_USED]
                         for a, e in zip(df["API_Smiles"], df["Excipient_Smiles"])])
    if version == "v1":
        return np.array([get_mechanism_matches(a, e) for a, e in zip(df["API_Smiles"], df["Excipient_Smiles"])])
    return np.array([mechanism_flags_v2(a, e) for a, e in zip(df["API_Smiles"], df["Excipient_Smiles"])]).astype(int)


def rule_stats(M, y, names):
    base = y.mean()
    rows = []
    for j, n in enumerate(names):
        k = M[:, j] == 1
        rows.append({"rule": n, "fires": int(k.sum()), "incompatible": int(y[k].sum()),
                     "precision": round(float(y[k].mean()), 3) if k.any() else np.nan,
                     "lift": round(float(y[k].mean() / base), 2) if k.any() else np.nan,
                     "recall": round(float(y[k].sum() / y.sum()), 3)})
    k = M.any(1)
    rows.append({"rule": "ANY", "fires": int(k.sum()), "incompatible": int(y[k].sum()),
                 "precision": round(float(y[k].mean()), 3), "lift": round(float(y[k].mean() / base), 2),
                 "recall": round(float(y[k].sum() / y.sum()), 3)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=0, help="list the N most common pairs per v2 rule (train)")
    ap.add_argument("--unused", action="store_true", help="also score the rules left out of v2")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 90)

    for split, path in SPLITS.items():
        df = pd.read_csv(path)
        y = df["Outcome1"].values
        print("=" * 100)
        print(f"{split}: {len(df)} pairs, base incompatible rate {y.mean():.3f}")
        versions = [("v1", [n.lower().replace(" ", "_") for n in get_mechanism_names()]), ("v2", MECH_V2_NAMES)]
        if args.unused:
            versions.append(("unused", [name for name, _, _ in RULES_NOT_USED]))
        for version, names in versions:
            print(f"\n{version}")
            print(rule_stats(flag_table(df, version), y, names).to_string(index=False))

    if args.examples:
        df = pd.read_csv(SPLITS["train"])
        M = flag_table(df, "v2")
        for j, (name, desc, _) in enumerate(RULES):
            k = M[:, j] == 1
            if not k.any():
                continue
            print("\n" + "-" * 100)
            print(f"{name}: {desc}")
            sub = df[k].groupby(["Excipient_Smiles"]).Outcome1.agg(["count", "mean"]).sort_values("count", ascending=False)
            print(sub.head(args.examples).round(2).to_string())


if __name__ == "__main__":
    main()
