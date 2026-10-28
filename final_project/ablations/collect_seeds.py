"""
Collect the val and held-out predictions of one ablation trained with several seeds.

Reads ablations/results/<ablation>_seed<seed>/predictions_{val,heldout}.csv (written by run_ablation.py)
and writes to ablations/results/<ablation>_<n>seeds/:

    predictions_val.csv        one row per pair: probability and prediction of every seed, the mean and sd
    predictions_heldout.csv    of the probabilities, and the ensemble (mean-probability) prediction
    metrics_per_seed.csv       val / held-out / held-out-unseen metrics per seed, mean +- sd, and ensemble
    summary.txt                the same as text

Each seed keeps its own val-tuned threshold. The ensemble averages the 5 probabilities and gets its own
threshold from val with the same rule (best val MCC with true-negative rate >= 0.97); held-out plays no
part in any threshold.

Usage (from final_project/):
    python ablations/collect_seeds.py --ablation train_global --seeds 42 43 44 45 46
"""

import argparse
import json
import os
import sys

FINAL_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FINAL_PROJECT)

import run  # noqa: E402  (chdirs into final_project/)
from run import frame_for, heldout_pairs_in_train, make_config, metric_suite  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import tune_threshold  # noqa: E402

RESULTS_ROOT = "ablations/results"
SPLITS = ["val", "heldout"]
METRIC_KEYS = ["correct", "n", "pr_auc", "roc_auc", "mcc", "f1", "precision", "recall", "accuracy", "tp", "tn", "fp", "fn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    out_dir = os.path.join(RESULTS_ROOT, f"{args.ablation}_{len(args.seeds)}seeds")
    os.makedirs(out_dir, exist_ok=True)
    config = make_config()
    unseen = ~heldout_pairs_in_train(frame_for("heldout"))

    tables, thresholds, rows = {}, {}, []
    for split in SPLITS:
        base = None
        for seed in args.seeds:
            pred = pd.read_csv(os.path.join(RESULTS_ROOT, f"{args.ablation}_seed{seed}", f"predictions_{split}.csv"))
            if base is None:
                id_cols = [c for c in ["Api_name", "Excipient_name", "API_CID", "Excipient_CID", "API_Smiles",
                                       "Excipient_Smiles", "label"] if c in pred]
                base = pred[id_cols].copy()
            assert (pred["label"].values == base["label"].values).all(), f"row order differs for seed {seed}"
            base[f"prob_seed{seed}"] = pred["probability"].values
            base[f"pred_seed{seed}"] = pred["prediction"].values
        probs = base[[f"prob_seed{s}" for s in args.seeds]].values
        base["prob_mean"] = probs.mean(1)
        base["prob_sd"] = probs.std(1, ddof=1)
        base["votes_incompatible"] = base[[f"pred_seed{s}" for s in args.seeds]].sum(1)
        tables[split] = base

    for seed in args.seeds:
        thr_file = os.path.join("ablations/checkpoints", f"{args.ablation}_seed{seed}", "threshold.json")
        thresholds[seed] = json.load(open(thr_file))["threshold"]

    # ensemble threshold from val only
    val = tables["val"]
    ens_thr, _ = tune_threshold(val["prob_mean"].values, val["label"].values, config.threshold_step,
                                tnr_floor=config.pgb_tnr_floor)
    for split in SPLITS:
        t = tables[split]
        t["pred_ensemble"] = (t["prob_mean"] >= ens_thr).astype(int)
        t["correct_ensemble"] = (t["pred_ensemble"] == t["label"]).astype(int)
        t.to_csv(os.path.join(out_dir, f"predictions_{split}.csv"), index=False)

    def score(name, probs_by_split, thr):
        row = {"model": name, "threshold": round(float(thr), 3)}
        for split, p in probs_by_split.items():
            y = tables[split]["label"].values
            for part, mask in ([("heldout", np.ones(len(y), bool)), ("heldout_unseen", unseen)] if split == "heldout"
                               else [("val", np.ones(len(y), bool))]):
                m = metric_suite(p[mask], y[mask], thr)
                for k in METRIC_KEYS:
                    row[f"{part}_{k}"] = m[k]
        return row

    for seed in args.seeds:
        rows.append(score(f"seed {seed}", {s: tables[s][f"prob_seed{seed}"].values for s in SPLITS}, thresholds[seed]))
    per_seed = pd.DataFrame(rows)
    num = per_seed.drop(columns=["model"]).astype(float)
    mean_row = {"model": "mean", **num.mean().round(4).to_dict()}
    sd_row = {"model": "sd", **num.std(ddof=1).round(4).to_dict()}
    ens_row = score("ensemble (mean prob)", {s: tables[s]["prob_mean"].values for s in SPLITS}, ens_thr)
    metrics = pd.concat([per_seed, pd.DataFrame([mean_row, sd_row, ens_row])], ignore_index=True)
    metrics.to_csv(os.path.join(out_dir, "metrics_per_seed.csv"), index=False)

    # text summary
    show = ["model", "threshold", "heldout_correct", "heldout_tp", "heldout_tn", "heldout_fp", "heldout_fn",
            "heldout_pr_auc", "heldout_roc_auc", "heldout_mcc", "heldout_unseen_correct",
            "val_correct", "val_pr_auc", "val_roc_auc", "val_mcc", "val_f1"]
    pd.set_option("display.width", 250)
    lines = [f"{args.ablation}: seeds {args.seeds}", "", metrics[show].round(4).to_string(index=False), ""]
    h = tables["heldout"]
    cols = ["Api_name", "Excipient_name", "label"] + [f"prob_seed{s}" for s in args.seeds] + \
           ["prob_mean", "prob_sd", "votes_incompatible", "pred_ensemble", "correct_ensemble"]
    lines += [f"Held-out predictions (ensemble threshold {ens_thr:.3f}, from val):",
              h[[c for c in cols if c in h]].round(3).to_string()]
    text = "\n".join(lines)
    open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8").write(text)
    print(text)
    print(f"\nSaved to {os.path.join(FINAL_PROJECT, out_dir)}")


if __name__ == "__main__":
    main()
