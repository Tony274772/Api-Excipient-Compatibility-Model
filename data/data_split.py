"""Leakage-safe data splitting using API-level Butina clustering.

Section 4.2 of the spec.  Run this script once from the project root:
    python data/data_split.py
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__))
RAW_CSV = os.path.join(DATA_DIR, "start_dataset.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SPLIT_LOG = os.path.join(DATA_DIR, "splitlog.log")


def _morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def butina_cluster(unique_smiles: list[str], dist_thresh: float = 0.15):
    """Return a dict mapping each SMILES to its cluster index."""
    fps = [_morgan_fp(s) for s in unique_smiles]
    n = len(fps)

    # Build condensed distance matrix (lower triangular, row-major)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, dist_thresh, isDistData=True)

    smiles_to_cluster = {}
    for cluster_idx, member_indices in enumerate(clusters):
        for member_idx in member_indices:
            smiles_to_cluster[unique_smiles[member_idx]] = cluster_idx

    return smiles_to_cluster, clusters


def main():
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
    # Butina clustering on unique API SMILES
    # ------------------------------------------------------------------
    unique_api_smiles = df["API_Smiles"].unique().tolist()
    logger.info(f"Unique APIs: {len(unique_api_smiles)}")

    smiles_to_cluster, clusters = butina_cluster(unique_api_smiles)

    num_clusters = len(clusters)
    multi_member = sum(1 for c in clusters if len(c) > 1)
    logger.info(f"Butina clusters: {num_clusters} total, {multi_member} multi-member")

    df["cluster_id"] = df["API_Smiles"].map(smiles_to_cluster)

    # ------------------------------------------------------------------
    # StratifiedGroupKFold → 60/20/20 split
    # ------------------------------------------------------------------
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # We take the first split's assignments: 3 folds train, 1 val, 1 test
    # StratifiedGroupKFold yields (train_idx, test_idx) for each fold.
    # To get 60/20/20 we collect all 5 fold assignments, then combine.
    fold_indices = {}
    for fold_i, (_, test_idx) in enumerate(
        sgkf.split(df, y=df["Outcome1"], groups=df["cluster_id"])
    ):
        fold_indices[fold_i] = test_idx

    # folds 0,1,2 -> train (60%), fold 3 -> val (20%), fold 4 -> test (20%)
    train_idx = np.concatenate([fold_indices[0], fold_indices[1], fold_indices[2]])
    val_idx = fold_indices[3]
    test_idx = fold_indices[4]

    train_df = df.iloc[train_idx].drop(columns=["cluster_id"])
    val_df = df.iloc[val_idx].drop(columns=["cluster_id"])
    test_df = df.iloc[test_idx].drop(columns=["cluster_id"])

    logger.info(
        f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
        f"total={len(train_df) + len(val_df) + len(test_df)}"
    )
    logger.info(
        f"Positive rates: train={train_df['Outcome1'].mean():.4f}, "
        f"val={val_df['Outcome1'].mean():.4f}, test={test_df['Outcome1'].mean():.4f}"
    )

    # ------------------------------------------------------------------
    # Verify no cluster leaks across splits
    # ------------------------------------------------------------------
    train_clusters = set(df.iloc[train_idx]["cluster_id"].unique())
    val_clusters = set(df.iloc[val_idx]["cluster_id"].unique())
    test_clusters = set(df.iloc[test_idx]["cluster_id"].unique())

    assert train_clusters.isdisjoint(val_clusters), "Cluster leak: train ∩ val"
    assert train_clusters.isdisjoint(test_clusters), "Cluster leak: train ∩ test"
    assert val_clusters.isdisjoint(test_clusters), "Cluster leak: val ∩ test"
    logger.info("✓ No cluster leakage across splits.")

    # ------------------------------------------------------------------
    # Write CSVs
    # ------------------------------------------------------------------
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    logger.info(f"Wrote {TRAIN_CSV}, {VAL_CSV}, {TEST_CSV}")


if __name__ == "__main__":
    main()
