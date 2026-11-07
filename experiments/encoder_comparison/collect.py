"""Build the comparison table and apply the selection rule.

Aggregates every seed of every family. With one seed it reports that seed; with several it
reports mean +- sd, which is the number to trust.

Selection rule, fixed before any run (paper/encoder_comparison_plan.md):
    1. highest mean validation PR-AUC
    2. tie-break: mean validation MCC
    3. tie-break: fewer trainable parameters
The 22 held-out pairs play no part. Families whose mean val PR-AUC is within --noise-band of the
best are reported as not separable.

    python -m experiments.encoder_comparison.collect                   # selection table
    python -m experiments.encoder_comparison.collect --per-seed        # every run, not aggregated
    python -m experiments.encoder_comparison.collect --reveal-heldout  # add held-out, AFTER choosing
"""

import argparse
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.encoder_comparison.families import FAMILIES, ORDER  # noqa: E402
from experiments.encoder_comparison.run_comparison import RESULTS_ROOT, SEALED_ROOT  # noqa: E402

VAL_COLS = ["pr_auc", "roc_auc", "mcc", "f1", "precision", "recall", "accuracy", "correct"]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def gather(reveal: bool) -> pd.DataFrame:
    """One row per (family, seed) that has results."""
    rows = []
    for family in ORDER:
        for entry in sorted(os.listdir(RESULTS_ROOT)):
            if not entry.startswith(f"{family}_seed"):
                continue
            seed = int(entry.rsplit("seed", 1)[1])
            val = load(os.path.join(RESULTS_ROOT, entry, "val_metrics.json"))
            info = load(os.path.join(RESULTS_ROOT, entry, "run_info.json"))
            if val is None or info is None:
                continue
            row = {"family": family, "seed": seed,
                   "params_trainable": info["parameters_trainable"],
                   "best_epoch": info["best_epoch"], "threshold": round(info["threshold_from_val"], 3),
                   "minutes": info["minutes"]}
            for k in VAL_COLS:
                row[f"val_{k}"] = val[k]
            if reveal:
                ho = load(os.path.join(SEALED_ROOT, entry, "heldout_metrics.json"))
                if ho:
                    row["heldout_correct"] = ho["correct"]
                    for k in ("pr_auc", "roc_auc", "mcc"):
                        row[f"heldout_{k}"] = ho[k]
            rows.append(row)
    if not rows:
        raise SystemExit("no results found. Run run_comparison.py or run_all_seeds.py first.")
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, reveal: bool) -> pd.DataFrame:
    metrics = [c for c in df.columns if c.startswith(("val_", "heldout_"))]
    out = []
    for family, g in df.groupby("family"):
        row = {"family": family, "seeds": len(g),
               "params_trainable": int(g.params_trainable.iloc[0]),
               "mean_minutes": round(g.minutes.mean(), 1)}
        for m in metrics:
            row[f"{m}_mean"] = round(float(g[m].mean()), 4)
            row[f"{m}_sd"] = round(float(g[m].std(ddof=1)), 4) if len(g) > 1 else float("nan")
        if reveal and "heldout_correct" in g:
            row["heldout_range"] = f"{int(g.heldout_correct.min())}-{int(g.heldout_correct.max())}"
        out.append(row)
    return pd.DataFrame(out).sort_values("val_pr_auc_mean", ascending=False).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="Collect the encoder comparison and apply the selection rule.")
    p.add_argument("--reveal-heldout", action="store_true",
                   help="add the sealed held-out column; only after the family has been chosen")
    p.add_argument("--per-seed", action="store_true", help="print every run instead of the aggregate")
    p.add_argument("--noise-band", type=float, default=0.03,
                   help="mean val PR-AUC within this of the best counts as not separable (default 0.03)")
    args = p.parse_args()

    runs = gather(args.reveal_heldout)
    agg = aggregate(runs, args.reveal_heldout)
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    runs.to_csv(os.path.join(RESULTS_ROOT, "summary_per_seed.csv"), index=False)
    agg.to_csv(os.path.join(RESULTS_ROOT, "summary.csv"), index=False)

    pd.set_option("display.width", 220)
    n_seeds = sorted(runs.seed.unique())
    print("\n" + "=" * 110)
    print(f"ENCODER COMPARISON — {runs.family.nunique()} families, seeds {n_seeds} ({len(runs)} runs)")
    print("=" * 110)

    if args.per_seed:
        show = ["family", "seed", "val_pr_auc", "val_mcc", "val_correct", "best_epoch", "threshold", "minutes"]
        if args.reveal_heldout and "heldout_correct" in runs:
            show += ["heldout_correct", "heldout_pr_auc"]
        print(runs.sort_values(["family", "seed"])[show].to_string(index=False))
        return

    show = ["family", "seeds", "val_pr_auc_mean", "val_pr_auc_sd", "val_mcc_mean", "val_mcc_sd",
            "val_correct_mean", "params_trainable"]
    if args.reveal_heldout and "heldout_correct_mean" in agg.columns:
        show += ["heldout_correct_mean", "heldout_correct_sd", "heldout_range", "heldout_pr_auc_mean"]
    print(agg[show].to_string(index=False))

    best = agg.iloc[0]
    band = agg[agg.val_pr_auc_mean >= best.val_pr_auc_mean - args.noise_band]
    winner = agg.sort_values(["val_pr_auc_mean", "val_mcc_mean", "params_trainable"],
                             ascending=[False, False, True]).iloc[0]
    one_seed = agg.seeds.max() < 2

    print(f"\nSelected by the rule (mean val PR-AUC, then mean val MCC, then fewer parameters): {winner['family']}")
    print(f"  {FAMILIES[winner['family']]['label']}")
    sd = winner.get("val_pr_auc_sd", float("nan"))
    print(f"  val PR-AUC {winner['val_pr_auc_mean']:.4f}" + (f" +- {sd:.4f}" if sd == sd else "")
          + f" over {int(winner['seeds'])} seed(s) | {winner['params_trainable']:,} trainable parameters")

    print(f"\nWithin {args.noise_band} mean val PR-AUC of the best ({len(band)} of {len(agg)}): "
          f"{', '.join(band['family'])}")
    if len(band) > 1:
        if one_seed:
            print("  -> one seed per family cannot separate these. Run more seeds:")
            print("     python -m experiments.encoder_comparison.run_all_seeds")
        else:
            worst_sd = agg.val_pr_auc_sd.max()
            print(f"  -> the largest seed-to-seed sd is {worst_sd:.4f}; gaps smaller than that are noise.")
    if not args.reveal_heldout:
        print("\nHeld-out results stay sealed in _heldout_sealed/ until the family is chosen.")
        print("After choosing: python -m experiments.encoder_comparison.collect --reveal-heldout")
    print(f"\nTables written to {os.path.join(RESULTS_ROOT, 'summary.csv')} and summary_per_seed.csv")


if __name__ == "__main__":
    main()
