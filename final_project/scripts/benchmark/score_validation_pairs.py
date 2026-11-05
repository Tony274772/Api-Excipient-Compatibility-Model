"""
Score the benchmark validation pairs (data/benchmark/) with the final Morgan-PGB model.

Only scores the pairs: no training, no threshold tuning. The threshold is the one saved with the checkpoint.
Pairs with an empty ground_truth are scored but left out of the correct count. Each row also says whether
the pair occurs in the train or val split.

Usage (from any directory):
    python scripts/benchmark/score_validation_pairs.py
    python scripts/benchmark/score_validation_pairs.py --csv data/benchmark/validation_pairs_formulationDE.csv
    python scripts/benchmark/score_validation_pairs.py --checkpoint checkpoints/morgan_pgb_8020_retrain_seed42 --device cpu

Output: results/benchmark/<csv name>_predictions.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # project root

import pandas as pd  # noqa: E402
from rdkit import RDLogger  # noqa: E402

from apiexc import paths  # noqa: E402
from apiexc.pipeline import get_split, load_checkpoint  # noqa: E402
from apiexc.utils import get_device  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PAIRS_CSV = paths.BENCHMARK_DATA / "validation_pairs.csv"
COLUMNS = {"drug_name": "Api_name", "drug_cid": "API_CID", "drug_smiles": "API_Smiles",
           "excipient_name": "Excipient_name", "excipient_cid": "Excipient_CID", "excipient_smiles": "Excipient_Smiles"}


def pair_in(df, csv):
    """Boolean list: does the (API_CID, Excipient_CID) of each row also occur in csv?"""
    other = pd.read_csv(csv)
    seen = set(zip(other["API_CID"], other["Excipient_CID"]))
    return [(a, e) in seen for a, e in zip(df["API_CID"], df["Excipient_CID"])]


def main():
    p = argparse.ArgumentParser(description="Score the benchmark validation pairs with the final Morgan-PGB model.")
    p.add_argument("--csv", default=str(PAIRS_CSV))
    p.add_argument("--checkpoint", default=str(paths.FINAL_CHECKPOINT))
    p.add_argument("--out", default=None, help="output CSV (default results/benchmark/<csv name>_predictions.csv)")
    p.add_argument("--device", default=None, help="cpu, cuda (default: cuda if available)")
    args = p.parse_args()
    csv = paths.resolve(args.csv)
    out = os.path.abspath(args.out) if args.out else paths.BENCHMARK_RESULTS / f"{csv.stem}_predictions.csv"

    lm = load_checkpoint(paths.resolve(args.checkpoint), get_device(args.device or "auto"))
    split = get_split()

    df = pd.read_csv(csv).rename(columns=COLUMNS)
    df["Outcome1"] = 0   # PGBDataset needs a label column; the value is not used for scoring
    probs, _ = lm.score(df)
    df = df.drop(columns="Outcome1")

    df["probability"] = probs
    df["prediction"] = (probs >= lm.threshold).astype(int)
    labeled = df["ground_truth"].notna()
    df["correct"] = (df["prediction"] == df["ground_truth"]).astype(int).where(labeled).astype("Int64")
    df["pair_in_train"] = pair_in(df, split.train_csv)
    df["pair_in_val"] = pair_in(df, split.val_csv)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)

    show = ["pair_no", "Api_name", "Excipient_name", "ground_truth", "deinteract_incompat_prob", "formulationde_prediction",
            "probability", "prediction", "correct", "pair_in_train", "pair_in_val"]
    print(df[[c for c in show if c in df]].round(4).to_string(index=False))
    print(f"\nlabeled pairs correct: {int(df.loc[labeled, 'correct'].sum())}/{int(labeled.sum())}")
    print(f"checkpoint {paths.rel(lm.folder)} | device {lm.device} | threshold {lm.threshold:.3f} | label 1 = incompatible")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
