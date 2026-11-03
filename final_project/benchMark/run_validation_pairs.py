"""
Inference of the final Morgan-PGB model on benchMark/validation_pairs.csv.

Only scores the pairs: no training, no threshold tuning. The threshold is the one saved with the checkpoint.
Pairs with an empty ground_truth are scored but left out of the correct count.

Usage (from any directory):
    python benchMark/run_validation_pairs.py
    python benchMark/run_validation_pairs.py --checkpoint checkpoints/morgan_pgb_8020_retrain_seed42 --device cpu
"""

import argparse
import json
import os
import sys

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BENCH_DIR))

import pandas as pd

from run import CALLER_CWD, FINAL_CKPT, TRAIN_CSV, VAL_CSV, load_checkpoint, score_frame   # run.py chdirs to final_project/
from src.utils import get_device

PAIRS_CSV = os.path.join(BENCH_DIR, "validation_pairs.csv")
OUT_CSV = os.path.join(BENCH_DIR, "validation_pairs_predictions.csv")
COLUMNS = {"drug_name": "Api_name", "drug_cid": "API_CID", "drug_smiles": "API_Smiles",
           "excipient_name": "Excipient_name", "excipient_cid": "Excipient_CID", "excipient_smiles": "Excipient_Smiles"}


def pair_in(df, csv):
    """Boolean list: does the (API_CID, Excipient_CID) of each row also occur in csv?"""
    other = pd.read_csv(csv)
    seen = set(zip(other["API_CID"], other["Excipient_CID"]))
    return [(a, e) in seen for a, e in zip(df["API_CID"], df["Excipient_CID"])]


def main():
    p = argparse.ArgumentParser(description="Score benchMark/validation_pairs.csv with the final Morgan-PGB model.")
    p.add_argument("--csv", default=PAIRS_CSV)
    p.add_argument("--checkpoint", default=FINAL_CKPT, help="path relative to final_project/, or absolute")
    p.add_argument("--out", default=OUT_CSV)
    p.add_argument("--device", default=None, help="cpu, cuda (default: cuda if available)")
    args = p.parse_args()
    csv = os.path.join(CALLER_CWD, args.csv)   # join keeps absolute paths as they are
    out = os.path.join(CALLER_CWD, args.out)

    device = get_device(args.device or "auto")
    model, config, encoder, prior_table, pgb_norm, _ = load_checkpoint(args.checkpoint, device)
    threshold = json.load(open(os.path.join(args.checkpoint, "threshold.json")))["threshold"]

    df = pd.read_csv(csv).rename(columns=COLUMNS)
    df["Outcome1"] = 0   # PGBDataset needs a label column; the value is not used for scoring
    probs, _ = score_frame(model, df, config, prior_table, pgb_norm, encoder, device)
    df = df.drop(columns="Outcome1")

    df["probability"] = probs
    df["prediction"] = (probs >= threshold).astype(int)
    labeled = df["ground_truth"].notna()
    df["correct"] = (df["prediction"] == df["ground_truth"]).astype(int).where(labeled).astype("Int64")
    df["pair_in_train"] = pair_in(df, TRAIN_CSV)
    df["pair_in_val"] = pair_in(df, VAL_CSV)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)

    show = ["pair_no", "Api_name", "Excipient_name", "ground_truth", "deinteract_incompat_prob", "formulationde_prediction",
            "probability", "prediction", "correct", "pair_in_train", "pair_in_val"]
    print(df[[c for c in show if c in df]].round(4).to_string(index=False))
    print(f"\nlabeled pairs correct: {int(df.loc[labeled, 'correct'].sum())}/{int(labeled.sum())}")
    print(f"checkpoint {args.checkpoint} | device {device} | threshold {threshold:.3f} | label 1 = incompatible")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
