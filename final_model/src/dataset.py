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
    if hasattr(config, "resolve_csv_paths"):
        config.resolve_csv_paths()

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


# ═══════════════════════════════════════════════════════════════════════════
# PGB dataset and collation
# ═══════════════════════════════════════════════════════════════════════════

from src.features import maccs_bits, morgan_bits, chemistry_flags, pgb_descriptors, salt_context
from src.pgb_prior import ExcipientPriorTable
from src.lookup import get_mechanism_matches


# PGB descriptor normalization stats — computed once per fold from training rows
class PGBNormStats:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std  = std

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)

    @classmethod
    def from_smiles_list(cls, smiles_list: list[str]) -> "PGBNormStats":
        from src.features import pgb_descriptors as _pgb_desc
        vecs = [_pgb_desc(s) for s in smiles_list]
        vecs = [v for v in vecs if not np.isnan(v).all()]
        mat  = np.stack(vecs)
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat, axis=0)
        return cls(mean, std)

    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "PGBNormStats":
        data = np.load(path)
        return cls(data["mean"], data["std"])


class PGBDataset(Dataset):
    """
    Dataset for PGBCompatibilityModel.
    Pre-computes all on-the-fly features in __getitem__:
        primary fingerprint (MACCS or Morgan etc.)
        secondary fingerprint (Morgan or MACCS)
        18 chemistry flags
        12 PGB descriptors (normalized)
        5-dim excipient prior vector
        5-dim mechanism flags
    """

    def __init__(
        self,
        csv_path: str,
        encoder_name: str,           # e.g. "fixed_vector", "molformer"
        fixed_vector_source: str,    # e.g. "maccs", "morgan", "pubchemfp", "mol2vec"
        prior_table: ExcipientPriorTable,
        pgb_norm_stats: PGBNormStats,
        fixed_vector_encoder=None,   # FixedVectorEncoder instance for lookup-based encoders
        sample_weight: float = 1.0,
        row_priors: list[np.ndarray] = None, # Precomputed per-row priors (for training)
    ):
        self.df = pd.read_csv(csv_path)
        self.enc_name = encoder_name
        self.fvs = fixed_vector_source
        self.prior_table = prior_table
        self.norm = pgb_norm_stats
        self.fv_enc = fixed_vector_encoder
        self.sample_weight = sample_weight
        self.row_priors = row_priors

    def _primary(self, smiles: str) -> tuple[np.ndarray, float]:
        """Primary fingerprint block for this family. Returns (vector, is_available)."""
        if self.enc_name == "fixed_vector":
            if self.fvs == "maccs":
                return maccs_bits(smiles), 1.0
            elif self.fvs == "morgan":
                return morgan_bits(smiles, n_bits=1024), 1.0
            elif self.fvs == "pubchemfp":
                if self.fv_enc is not None:
                    v = self.fv_enc.encode([smiles]).cpu().numpy()[0]
                    # If all-zero, it means the lookup failed
                    if not v.any():
                        return np.zeros(881, dtype=np.float32), 0.0
                    return v, 1.0
                return np.zeros(881, dtype=np.float32), 0.0
            elif self.fvs == "mol2vec":
                if self.fv_enc is not None:
                    v = self.fv_enc.encode([smiles]).cpu().numpy()[0]
                    if not v.any():
                        return np.zeros(300, dtype=np.float32), 0.0
                    return v, 1.0
                return np.zeros(300, dtype=np.float32), 0.0
        # Seq encoders: use appropriate primary fingerprint
        if self.enc_name == "chemberta":
            return morgan_bits(smiles, n_bits=512), 1.0
        return maccs_bits(smiles), 1.0

    def _secondary(self, smiles: str) -> np.ndarray | None:
        """Secondary fingerprint block (may be None)."""
        if self.enc_name == "fixed_vector":
            if self.fvs == "maccs":
                return morgan_bits(smiles, n_bits=512)  # [512]
            elif self.fvs == "morgan":
                return maccs_bits(smiles)               # [167]
            elif self.fvs == "pubchemfp":
                return morgan_bits(smiles, n_bits=512)  # [512]
            elif self.fvs == "mol2vec":
                return morgan_bits(smiles, n_bits=512)  # [512]
        return None  # seq encoders have no secondary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        api_smi = str(row["API_Smiles"])
        exc_smi = str(row["Excipient_Smiles"]) if pd.notna(row.get("Excipient_Smiles")) else ""
        exc_available = 1.0 if exc_smi else 0.0
        label = float(row["Outcome1"])

        api_prim, api_prim_avail = self._primary(api_smi)
        if exc_smi:
            exc_prim, exc_prim_avail = self._primary(exc_smi)
        else:
            exc_prim, exc_prim_avail = np.zeros_like(api_prim), 0.0

        api_sec  = self._secondary(api_smi)
        exc_sec  = self._secondary(exc_smi) if exc_smi else (
            np.zeros_like(api_sec) if api_sec is not None else None
        )

        api_flags = chemistry_flags(api_smi)
        exc_flags = chemistry_flags(exc_smi) if exc_smi else np.zeros(18, dtype=np.float32)

        api_desc_raw = pgb_descriptors(api_smi)
        exc_desc_raw = pgb_descriptors(exc_smi) if exc_smi else np.full(12, 0.0, dtype=np.float32)
        api_desc_raw = np.nan_to_num(api_desc_raw, nan=0.0)
        exc_desc_raw = np.nan_to_num(exc_desc_raw, nan=0.0)
        api_desc_norm = np.nan_to_num(self.norm.normalize(api_desc_raw), nan=0.0)
        exc_desc_norm = np.nan_to_num(self.norm.normalize(exc_desc_raw), nan=0.0)

        api_salt = salt_context(api_smi)
        exc_salt = salt_context(exc_smi) if exc_smi else np.zeros(8, dtype=np.float32)

        if self.row_priors is not None:
            prior_vec = self.row_priors[idx]
        else:
            prior_info = self.prior_table.lookup(exc_smi) if exc_smi else {
                "p_exact": 0.094, "p_delta": 0.0, "p_family": 0.094, "log_n": 0.0, "unseen": 1.0,
            }
            prior_vec = np.array([
                prior_info["p_exact"],
                prior_info["p_delta"],
                prior_info["p_family"],
                prior_info["log_n"],
                prior_info["unseen"],
            ], dtype=np.float32)

        mech = get_mechanism_matches(api_smi, exc_smi) if exc_smi else [0] * 5
        mech_flags = np.array(mech, dtype=np.float32)

        item = {
            "api_smiles":       api_smi,
            "exc_smiles":       exc_smi,
            "api_primary":      torch.tensor(api_prim,      dtype=torch.float32),
            "exc_primary":      torch.tensor(exc_prim,      dtype=torch.float32),
            "api_prim_avail":   torch.tensor(api_prim_avail,dtype=torch.float32),
            "exc_prim_avail":   torch.tensor(exc_prim_avail,dtype=torch.float32),
            "api_pgb_desc":     torch.tensor(api_desc_raw,  dtype=torch.float32),
            "exc_pgb_desc":     torch.tensor(exc_desc_raw,  dtype=torch.float32),
            "api_pgb_desc_norm":torch.tensor(api_desc_norm, dtype=torch.float32),
            "exc_pgb_desc_norm":torch.tensor(exc_desc_norm, dtype=torch.float32),
            "api_salt":         torch.tensor(api_salt,      dtype=torch.float32),
            "exc_salt":         torch.tensor(exc_salt,      dtype=torch.float32),
            "api_flags":        torch.tensor(api_flags,     dtype=torch.float32),
            "exc_flags":        torch.tensor(exc_flags,     dtype=torch.float32),
            "prior_vec":        torch.tensor(prior_vec,     dtype=torch.float32),
            "mech_flags":       torch.tensor(mech_flags,    dtype=torch.float32),
            "exc_available":    torch.tensor(exc_available, dtype=torch.float32),
            "label":            torch.tensor(label,         dtype=torch.float32),
            "sample_weight":    torch.tensor(self.sample_weight, dtype=torch.float32),
        }
        if api_sec is not None:
            item["api_secondary"] = torch.tensor(api_sec, dtype=torch.float32)
            item["exc_secondary"] = torch.tensor(exc_sec, dtype=torch.float32)

        return item


def pgb_collate_fn(batch: list[dict]) -> dict:
    """Collate for PGBDataset — handles optional secondary fingerprint."""
    keys = list(batch[0].keys())
    out = {}
    for k in keys:
        if k in ("api_smiles", "exc_smiles"):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
