"""Dataset and DataLoader utilities – Section 4 / 11."""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.descriptors import DESCRIPTOR_NAMES, normalize_descriptors


class CompatibilityDataset(Dataset):
    """Dataset for API–Excipient compatibility prediction.

    Each item returns:
        api_smiles    : str
        exc_smiles    : str
        api_desc      : FloatTensor [21]
        exc_desc      : FloatTensor [21]
        exc_available : float (1.0 if excipient SMILES is present, else 0.0)
        label         : float (0 or 1)
        sample_weight : float (1.0 for real data, configurable for weak labels)
    """

    def __init__(
        self,
        csv_path: str,
        api_desc_path: str,
        exc_desc_path: str,
        norm_stats_path: str,
        sample_weight: float = 1.0,
    ):
        self.df = pd.read_csv(csv_path)
        self.sample_weight = sample_weight

        # Load descriptor look-ups
        api_desc_df = pd.read_csv(api_desc_path)
        exc_desc_df = pd.read_csv(exc_desc_path)

        self.api_desc_map = {}
        for _, row in api_desc_df.iterrows():
            self.api_desc_map[row["API_CID"]] = np.array(
                [row[n] for n in DESCRIPTOR_NAMES], dtype=np.float32
            )

        self.exc_desc_map = {}
        for _, row in exc_desc_df.iterrows():
            self.exc_desc_map[row["Excipient_CID"]] = np.array(
                [row[n] for n in DESCRIPTOR_NAMES], dtype=np.float32
            )

        # Load normalization stats
        with open(norm_stats_path, "r") as f:
            stats = json.load(f)
        self.api_mean = np.array(stats["api_mean"], dtype=np.float32)
        self.api_std = np.array(stats["api_std"], dtype=np.float32)
        self.exc_mean = np.array(stats["exc_mean"], dtype=np.float32)
        self.exc_std = np.array(stats["exc_std"], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        api_smiles = str(row["API_Smiles"])
        exc_smiles = str(row["Excipient_Smiles"]) if pd.notna(row.get("Excipient_Smiles")) else ""

        # Descriptors (normalized)
        api_desc_raw = self.api_desc_map.get(row["API_CID"], np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32))
        exc_desc_raw = self.exc_desc_map.get(row["Excipient_CID"], np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32))

        api_desc = normalize_descriptors(api_desc_raw, self.api_mean, self.api_std)
        exc_desc = normalize_descriptors(exc_desc_raw, self.exc_mean, self.exc_std)

        # Replace any NaN from failed RDKit parsing with 0
        api_desc = np.nan_to_num(api_desc, nan=0.0)
        exc_desc = np.nan_to_num(exc_desc, nan=0.0)

        exc_available = 1.0 if exc_smiles != "" else 0.0
        label = float(row["Outcome1"])

        return {
            "api_smiles": api_smiles,
            "exc_smiles": exc_smiles,
            "api_desc": torch.tensor(api_desc, dtype=torch.float32),
            "exc_desc": torch.tensor(exc_desc, dtype=torch.float32),
            "exc_available": torch.tensor(exc_available, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "sample_weight": torch.tensor(self.sample_weight, dtype=torch.float32),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate: stack tensors, keep SMILES as lists."""
    return {
        "api_smiles": [b["api_smiles"] for b in batch],
        "exc_smiles": [b["exc_smiles"] for b in batch],
        "api_desc": torch.stack([b["api_desc"] for b in batch]),
        "exc_desc": torch.stack([b["exc_desc"] for b in batch]),
        "exc_available": torch.stack([b["exc_available"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "sample_weight": torch.stack([b["sample_weight"] for b in batch]),
    }


def build_dataloaders(config):
    """Build train/val/test DataLoaders from config."""
    if hasattr(config, "resolve_paths"):
        config.resolve_paths()

    train_ds = CompatibilityDataset(
        config.train_csv,
        config.api_descriptors_path,
        config.excipient_descriptors_path,
        config.descriptor_norm_stats_path,
    )
    val_ds = CompatibilityDataset(
        config.val_csv,
        config.api_descriptors_path,
        config.excipient_descriptors_path,
        config.descriptor_norm_stats_path,
    )
    test_ds = CompatibilityDataset(
        config.test_csv,
        config.api_descriptors_path,
        config.excipient_descriptors_path,
        config.descriptor_norm_stats_path,
    )

    # Balanced sampler for training
    train_sampler = None
    train_shuffle = True
    if config.use_balanced_sampler:
        labels = train_ds.df["Outcome1"].values
        class_counts = np.bincount(labels.astype(int))
        weights = 1.0 / class_counts[labels.astype(int)]
        train_sampler = WeightedRandomSampler(
            weights=weights.tolist(),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_shuffle = False  # sampler and shuffle are mutually exclusive

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader
