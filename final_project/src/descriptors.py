"""RDKit descriptor computation and normalization – Section 4.3.

Usage (from project root):
    python -m src.descriptors            # compute descriptors + norm stats
    python -m src.descriptors --compute  # same
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# The exact 21 descriptors specified in Section 4.3.
DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "fr_NH2",
    "fr_NH1",
    "fr_ester",
    "fr_amide",
    "fr_aldehyde",
    "fr_ketone",
    "fr_phenol",
    "fr_ether",
    "fr_epoxide",
    "fr_halogen",
    "fr_COO",
]


def _compute_descriptors(smiles: str) -> list[float]:
    """Compute the 21-dim descriptor vector for a single SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return NaNs so the row is detectable downstream
        return [float("nan")] * len(DESCRIPTOR_NAMES)

    values = []
    for name in DESCRIPTOR_NAMES:
        if name == "FractionCSP3":
            values.append(rdMolDescriptors.CalcFractionCSP3(mol))
        else:
            func = getattr(Descriptors, name, None)
            if func is None:
                raise AttributeError(f"RDKit Descriptors has no attribute '{name}'")
            values.append(func(mol))
    return values


def compute_descriptor_table(df: pd.DataFrame, cid_col: str, smiles_col: str) -> pd.DataFrame:
    """Return a DataFrame with one row per unique CID, columns = descriptor names."""
    unique = df[[cid_col, smiles_col]].drop_duplicates(subset=[cid_col])
    records = []
    for _, row in unique.iterrows():
        vals = _compute_descriptors(row[smiles_col])
        records.append({cid_col: row[cid_col], **dict(zip(DESCRIPTOR_NAMES, vals))})
    return pd.DataFrame(records)


def compute_norm_stats(
    train_df: pd.DataFrame,
    api_desc_df: pd.DataFrame,
    exc_desc_df: pd.DataFrame,
) -> dict:
    """Compute per-column mean/std from training-split molecules only."""
    train_api_cids = set(train_df["API_CID"].unique())
    train_exc_cids = set(train_df["Excipient_CID"].unique())

    api_train = api_desc_df[api_desc_df["API_CID"].isin(train_api_cids)][DESCRIPTOR_NAMES]
    exc_train = exc_desc_df[exc_desc_df["Excipient_CID"].isin(train_exc_cids)][DESCRIPTOR_NAMES]

    stats = {
        "api_mean": api_train.mean().tolist(),
        "api_std": api_train.std().tolist(),
        "exc_mean": exc_train.mean().tolist(),
        "exc_std": exc_train.std().tolist(),
    }
    return stats


def normalize_descriptors(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize: (x - mean) / (std + 1e-8)."""
    return (values - mean) / (std + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", default=True)
    parser.parse_args()

    data_dir = "data"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Load full dataset to get unique CIDs/SMILES
    raw = pd.read_csv(os.path.join(data_dir, "start_dataset.csv"))
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))

    # Compute descriptor tables
    print("Computing API descriptors...")
    api_desc = compute_descriptor_table(raw, "API_CID", "API_Smiles")
    api_desc.to_csv(os.path.join(data_dir, "api_descriptors.csv"), index=False)
    print(f"  Saved {len(api_desc)} rows to data/api_descriptors.csv")

    print("Computing Excipient descriptors...")
    exc_desc = compute_descriptor_table(raw, "Excipient_CID", "Excipient_Smiles")
    exc_desc.to_csv(os.path.join(data_dir, "excipient_descriptors.csv"), index=False)
    print(f"  Saved {len(exc_desc)} rows to data/excipient_descriptors.csv")

    # Compute normalization stats from training split only
    print("Computing normalization stats (training split only)...")
    stats = compute_norm_stats(train, api_desc, exc_desc)
    stats_path = os.path.join(models_dir, "descriptor_norm_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved to {stats_path}")


if __name__ == "__main__":
    main()
