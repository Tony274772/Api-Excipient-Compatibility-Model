"""Standard random split (60/20/20) stratified on Outcome1.

Run this script once from the project root:
    python data/random_data_split.py
"""

import logging
import os
import sys

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__))
RANDOM_SPLIT_DIR = os.path.join(DATA_DIR, "random_split")
RAW_CSV = os.path.join(DATA_DIR, "start_dataset.csv")

TRAIN_CSV = os.path.join(RANDOM_SPLIT_DIR, "train.csv")
VAL_CSV = os.path.join(RANDOM_SPLIT_DIR, "val.csv")
TEST_CSV = os.path.join(RANDOM_SPLIT_DIR, "test.csv")
SPLIT_LOG = os.path.join(RANDOM_SPLIT_DIR, "splitlog.log")


def main():
    if not os.path.exists(RANDOM_SPLIT_DIR):
        os.makedirs(RANDOM_SPLIT_DIR)

    logging.basicConfig(
        filename=SPLIT_LOG,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
    )
    logger = logging.getLogger(__name__)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    df = pd.read_csv(RAW_CSV)
    logger.info(f"Loaded {len(df)} rows from {RAW_CSV}")

    # ------------------------------------------------------------------
    # StratifiedShuffleSplit → 60/20/20 split
    # ------------------------------------------------------------------
    # First split: 60% train, 40% (val + test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, temp_idx = next(sss1.split(df, df["Outcome1"]))

    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    # Second split: 50% val, 50% test of the remaining 40% (so 20% each of total)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx_temp, test_idx_temp = next(sss2.split(temp_df, temp_df["Outcome1"]))

    val_df = temp_df.iloc[val_idx_temp]
    test_df = temp_df.iloc[test_idx_temp]

    logger.info(
        f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
        f"total={len(train_df) + len(val_df) + len(test_df)}"
    )
    logger.info(
        f"Positive rates: train={train_df['Outcome1'].mean():.4f}, "
        f"val={val_df['Outcome1'].mean():.4f}, test={test_df['Outcome1'].mean():.4f}"
    )

    # ------------------------------------------------------------------
    # Write CSVs
    # ------------------------------------------------------------------
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    logger.info(f"Wrote {TRAIN_CSV}, {VAL_CSV}, {TEST_CSV}")


if __name__ == "__main__":
    main()
