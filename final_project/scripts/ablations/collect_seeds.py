"""
Collect the val and held-out predictions of one ablation trained with several seeds.

Reads results/ablations/runs/<ablation>_seed<seed>/predictions_{val,heldout}.csv (written by run_ablation.py)
and the thresholds in checkpoints/ablations/<ablation>_seed<seed>/, and writes to
results/ablations/multi_seed/<ablation>_<n>seeds/:

    predictions_val.csv        one row per pair: probability and prediction of every seed, the mean and sd
    predictions_heldout.csv    of the probabilities, and the ensemble (mean-probability) prediction
    metrics_per_seed.csv       val / held-out metrics per seed, mean +- sd, and ensemble
    summary.txt                the same as text

Each seed keeps its own val-tuned threshold. The ensemble averages the 5 probabilities and gets its own
threshold from val with the same rule (best val MCC with true-negative rate >= 0.97); held-out plays no
part in any threshold.

Usage (from any directory):
    python scripts/ablations/collect_seeds.py --ablation full --seeds 42 43 44 45 46
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # project root

import pandas as pd  # noqa: E402

from apiexc import paths  # noqa: E402
from apiexc.evaluation.metrics import metric_suite, tune_threshold  # noqa: E402
from apiexc.pipeline import get_split, make_config, read_threshold  # noqa: E402

SPLITS = ["val", "heldout"]
METRIC_KEYS = ["correct", "n", "pr_auc", "roc_auc", "mcc", "f1", "precision", "recall", "accuracy", "tp", "tn", "fp", "fn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    out_dir = paths.ABLATION_MULTI_SEED / f"{args.ablation}_{len(args.seeds)}seeds"
    os.makedirs(out_dir, exist_ok=True)
    config = make_config(get_split())

    tables, thresholds, rows = {}, {}, []
    for split in SPLITS:
        base = None
        for seed in args.seeds:
            pred = pd.read_csv(paths.ABLATION_RUNS / f"{args.ablation}_seed{seed}" / f"predictions_{split}.csv")
            if base is None:
                id_cols = [c for c in ["Api_name", "Excipient_name", "API_CID", "Excipient_CID", "API_Smiles",
                                       "Excipient_Smiles", "label"] if c in pred]
                base = pred[id_cols].copy()
            if len(pred) != len(base) or not (pred["label"].values == base["label"].values).all():
                raise SystemExit(f"{split} predictions of seed {seed} do not have the same rows as seed {args.seeds[0]}")
            base[f"prob_seed{seed}"] = pred["probability"].values
            base[f"pred_seed{seed}"] = pred["prediction"].values
        probs = base[[f"prob_seed{s}" for s in args.seeds]].values
        base["prob_mean"] = probs.mean(1)
        base["prob_sd"] = probs.std(1, ddof=1)
        base["votes_incompatible"] = base[[f"pred_seed{s}" for s in args.seeds]].sum(1)
        tables[split] = base

    for seed in args.seeds:
        thresholds[seed] = read_threshold(paths.ABLATION_CHECKPOINTS / f"{args.ablation}_seed{seed}")

    # ensemble threshold from val only
    val = tables["val"]
    ens_thr, _ = tune_threshold(val["prob_mean"].values, val["label"].values, config.threshold_step,
                                tnr_floor=config.pgb_tnr_floor)
    for split in SPLITS:
        t = tables[split]
        t["pred_ensemble"] = (t["prob_mean"] >= ens_thr).astype(int)
        t["correct_ensemble"] = (t["pred_ensemble"] == t["label"]).astype(int)
        t.to_csv(out_dir / f"predictions_{split}.csv", index=False)

    def score(name, probs_by_split, thr):
        row = {"model": name, "threshold": round(float(thr), 3)}
        for split, p in probs_by_split.items():
            m = metric_suite(p, tables[split]["label"].values, thr)
            for k in METRIC_KEYS:
                row[f"{split}_{k}"] = m[k]
        return row

    for seed in args.seeds:
        rows.append(score(f"seed {seed}", {s: tables[s][f"prob_seed{seed}"].values for s in SPLITS}, thresholds[seed]))
    per_seed = pd.DataFrame(rows)
    num = per_seed.drop(columns=["model"]).astype(float)
    mean_row = {"model": "mean", **num.mean().round(4).to_dict()}
    sd_row = {"model": "sd", **num.std(ddof=1).round(4).to_dict()}
    ens_row = score("ensemble (mean prob)", {s: tables[s]["prob_mean"].values for s in SPLITS}, ens_thr)
    metrics = pd.concat([per_seed, pd.DataFrame([mean_row, sd_row, ens_row])], ignore_index=True)
    metrics.to_csv(out_dir / "metrics_per_seed.csv", index=False)

    # text summary
    show = ["model", "threshold", "heldout_correct", "heldout_tp", "heldout_tn", "heldout_fp", "heldout_fn",
            "heldout_pr_auc", "heldout_roc_auc", "heldout_mcc",
            "val_correct", "val_pr_auc", "val_roc_auc", "val_mcc", "val_f1"]
    pd.set_option("display.width", 250)
    lines = [f"{args.ablation}: seeds {args.seeds}", "", metrics[show].round(4).to_string(index=False), ""]
    h = tables["heldout"]
    cols = ["Api_name", "Excipient_name", "label"] + [f"prob_seed{s}" for s in args.seeds] + \
           ["prob_mean", "prob_sd", "votes_incompatible", "pred_ensemble", "correct_ensemble"]
    lines += [f"Held-out predictions (ensemble threshold {ens_thr:.3f}, from val):",
              h[[c for c in cols if c in h]].round(3).to_string()]
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
