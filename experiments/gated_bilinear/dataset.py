"""Extended dataset for the Gated Bilinear (PGB) architecture.

Adds chemistry flags, mechanism flags, prior block features, and on-the-fly
fingerprints (MACCS, Morgan) on top of the base dataset. Does NOT modify
the base CompatibilityDataset.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from experiments.gated_bilinear.features import (
    compute_maccs,
    compute_morgan,
    compute_gb_descriptors,
    compute_gb_descriptor_norm_stats,
    normalize_gb_descriptors,
    compute_chem_flags,
    compute_mechanism_flags,
    GB_DESCRIPTOR_NAMES,
)
from experiments.gated_bilinear.prior import ExcipientPriorTable


class GBCompatibilityDataset(Dataset):
    """Dataset for PGB training/evaluation.

    Each item returns a dict with all tensors needed by GatedBilinearModel:
        api_smiles, exc_smiles, exc_available,
        api_maccs [166], exc_maccs [166],
        api_morgan [morgan_bits], exc_morgan [morgan_bits],
        api_descriptors [12], exc_descriptors [12],
        api_chem_flags [18], exc_chem_flags [18],
        mechanism_flags [5],
        prior_block [5],
        p_exact (scalar),
        label (scalar),
        sample_weight (scalar),
        # Optional for specific families:
        api_pubchem [881], exc_pubchem [881],
        api_mol2vec [300], exc_mol2vec [300],
    """

    def __init__(
        self,
        csv_path: str,
        prior_table: ExcipientPriorTable,
        descriptor_norm_stats: dict,
        family: str = "maccs",
        morgan_bits: int = 512,
        is_train: bool = False,
        pubchem_vectors: Optional[dict] = None,
        mol2vec_vectors: Optional[dict] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.prior_table = prior_table
        self.family = family.lower()
        self.morgan_bits = morgan_bits
        self.is_train = is_train

        # Normalization stats for 12 descriptors
        self.desc_mean = np.array(descriptor_norm_stats["mean"], dtype=np.float32)
        self.desc_std = np.array(descriptor_norm_stats["std"], dtype=np.float32)

        # Precomputed vectors for PubChem / Mol2vec (keyed by SMILES or CID)
        self.pubchem_vectors = pubchem_vectors or {}
        self.mol2vec_vectors = mol2vec_vectors or {}

        # Build SMILES→CID mapping for lookup
        self._smiles_to_cid: dict[str, str] = {}
        for _, row in self.df.iterrows():
            if pd.notna(row.get("API_Smiles")) and pd.notna(row.get("API_CID")):
                self._smiles_to_cid[str(row["API_Smiles"])] = str(int(row["API_CID"]))
            if pd.notna(row.get("Excipient_Smiles")) and pd.notna(row.get("Excipient_CID")):
                self._smiles_to_cid[str(row["Excipient_Smiles"])] = str(int(row["Excipient_CID"]))

        # Detect cluster column (for leave-own-API-cluster-out prior)
        self._cluster_col = None
        for col in ["API_Cluster", "api_cluster", "Cluster"]:
            if col in self.df.columns:
                self._cluster_col = col
                break

    def __len__(self):
        return len(self.df)

    def _lookup_vector(self, smiles: str, vector_dict: dict, dim: int) -> np.ndarray:
        """Lookup a precomputed vector by SMILES or CID."""
        if smiles in vector_dict:
            return vector_dict[smiles]
        cid = self._smiles_to_cid.get(smiles)
        if cid is not None and cid in vector_dict:
            return vector_dict[cid]
        return np.zeros(dim, dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        api_smiles = str(row["API_Smiles"])
        exc_smiles = str(row["Excipient_Smiles"]) if pd.notna(row.get("Excipient_Smiles")) else ""
        exc_available = 1.0 if exc_smiles != "" else 0.0
        label = float(row["Outcome1"])

        # API cluster for leave-own-API-cluster-out
        api_cluster = None
        if self._cluster_col is not None and pd.notna(row.get(self._cluster_col)):
            api_cluster = int(row[self._cluster_col])

        # ── MACCS keys (166 bits, on-the-fly) ────────────────────────────
        api_maccs = compute_maccs(api_smiles)
        exc_maccs = compute_maccs(exc_smiles) if exc_available else np.zeros(166, dtype=np.float32)

        # ── Morgan fingerprints (on-the-fly) ─────────────────────────────
        api_morgan = compute_morgan(api_smiles, n_bits=self.morgan_bits)
        exc_morgan = compute_morgan(exc_smiles, n_bits=self.morgan_bits) if exc_available else np.zeros(self.morgan_bits, dtype=np.float32)

        # ── 12 RDKit descriptors (z-scored) ──────────────────────────────
        api_desc_raw = compute_gb_descriptors(api_smiles)
        exc_desc_raw = compute_gb_descriptors(exc_smiles) if exc_available else np.zeros(len(GB_DESCRIPTOR_NAMES), dtype=np.float32)

        api_desc = normalize_gb_descriptors(api_desc_raw, self.desc_mean, self.desc_std)
        exc_desc = normalize_gb_descriptors(exc_desc_raw, self.desc_mean, self.desc_std)
        api_desc = np.nan_to_num(api_desc, nan=0.0)
        exc_desc = np.nan_to_num(exc_desc, nan=0.0)

        # ── 18 chemistry flags ───────────────────────────────────────────
        api_chem_flags = compute_chem_flags(api_smiles)
        exc_chem_flags = compute_chem_flags(exc_smiles) if exc_available else np.zeros(18, dtype=np.float32)

        # ── 5 mechanism flags (per-pair) ─────────────────────────────────
        mechanism_flags = compute_mechanism_flags(api_smiles, exc_smiles) if exc_available else np.zeros(5, dtype=np.float32)

        # ── Prior block [5] ──────────────────────────────────────────────
        if exc_available:
            prior_block = self.prior_table.get_prior_block(
                exc_smiles,
                api_cluster=api_cluster,
                is_train=self.is_train,
            )
            p_exact = self.prior_table.get_logit_offset_p(exc_smiles)
        else:
            prior_block = np.array([
                self.prior_table.global_rate,
                0.0,
                self.prior_table.global_rate,
                0.0,
                1.0,  # unseen
            ], dtype=np.float32)
            p_exact = self.prior_table.global_rate

        # ── Build result dict ────────────────────────────────────────────
        result = {
            "api_smiles": api_smiles,
            "exc_smiles": exc_smiles,
            "exc_available": torch.tensor(exc_available, dtype=torch.float32),
            "api_maccs": torch.tensor(api_maccs, dtype=torch.float32),
            "exc_maccs": torch.tensor(exc_maccs, dtype=torch.float32),
            "api_morgan": torch.tensor(api_morgan, dtype=torch.float32),
            "exc_morgan": torch.tensor(exc_morgan, dtype=torch.float32),
            "api_descriptors": torch.tensor(api_desc, dtype=torch.float32),
            "exc_descriptors": torch.tensor(exc_desc, dtype=torch.float32),
            "api_chem_flags": torch.tensor(api_chem_flags, dtype=torch.float32),
            "exc_chem_flags": torch.tensor(exc_chem_flags, dtype=torch.float32),
            "mechanism_flags": torch.tensor(mechanism_flags, dtype=torch.float32),
            "prior_block": torch.tensor(prior_block, dtype=torch.float32),
            "p_exact": torch.tensor(p_exact, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "sample_weight": torch.tensor(1.0, dtype=torch.float32),
        }

        # ── Optional family-specific vectors ─────────────────────────────
        if self.family == "pubchem":
            api_pubchem = self._lookup_vector(api_smiles, self.pubchem_vectors, 881)
            exc_pubchem = self._lookup_vector(exc_smiles, self.pubchem_vectors, 881) if exc_available else np.zeros(881, dtype=np.float32)
            result["api_pubchem"] = torch.tensor(api_pubchem, dtype=torch.float32)
            result["exc_pubchem"] = torch.tensor(exc_pubchem, dtype=torch.float32)

        if self.family == "mol2vec":
            api_mol2vec = self._lookup_vector(api_smiles, self.mol2vec_vectors, 300)
            exc_mol2vec = self._lookup_vector(exc_smiles, self.mol2vec_vectors, 300) if exc_available else np.zeros(300, dtype=np.float32)
            result["api_mol2vec"] = torch.tensor(api_mol2vec, dtype=torch.float32)
            result["exc_mol2vec"] = torch.tensor(exc_mol2vec, dtype=torch.float32)

        return result


def gb_collate_fn(batch: list[dict]) -> dict:
    """Custom collate: stack tensors, keep SMILES as lists."""
    result = {
        "api_smiles": [b["api_smiles"] for b in batch],
        "exc_smiles": [b["exc_smiles"] for b in batch],
    }

    # Stack all tensor fields
    tensor_keys = [k for k in batch[0] if isinstance(batch[0][k], torch.Tensor)]
    for key in tensor_keys:
        result[key] = torch.stack([b[key] for b in batch])

    return result


def load_precomputed_vectors(csv_path: str) -> dict[str, np.ndarray]:
    """Load a precomputed vector CSV (PubChem FP or Mol2vec).

    Returns a dict mapping SMILES / CID strings to numpy arrays.
    """
    if not csv_path or not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    # Detect key column
    smiles_col = next((c for c in cols if "smiles" in c.lower()), None)
    cid_col = next((c for c in cols if "cid" in c.lower()), None)
    key_col = smiles_col or cid_col or cols[0]

    exclude = {key_col}
    if smiles_col and smiles_col != key_col:
        exclude.add(smiles_col)
    if cid_col and cid_col != key_col:
        exclude.add(cid_col)

    feature_cols = [c for c in cols if c not in exclude]
    vectors = {}
    for _, row in df.iterrows():
        vec = row[feature_cols].values.astype(np.float32)
        if smiles_col:
            vectors[str(row[smiles_col])] = vec
        if cid_col:
            vectors[str(row[cid_col])] = vec
        if not smiles_col and not cid_col:
            vectors[str(row[key_col])] = vec

    return vectors


def build_gb_dataloaders(config, prior_table, descriptor_norm_stats):
    """Build train/val/test DataLoaders for PGB training.

    Args:
        config: GBConfig instance.
        prior_table: ExcipientPriorTable built from training data.
        descriptor_norm_stats: dict with 'mean' and 'std' (12 floats each).

    Returns:
        train_loader, val_loader, test_loader
    """
    family = config.gb_family
    morgan_bits = 1024 if family == "morgan" else 512

    # Load precomputed vectors if needed
    pubchem_vectors = None
    mol2vec_vectors = None
    if family == "pubchem":
        pubchem_vectors = load_precomputed_vectors("data/pubchem_fps.csv")
    elif family == "mol2vec":
        mol2vec_vectors = load_precomputed_vectors("data/mol2vec_embeddings.csv")

    train_ds = GBCompatibilityDataset(
        csv_path=config.train_csv,
        prior_table=prior_table,
        descriptor_norm_stats=descriptor_norm_stats,
        family=family,
        morgan_bits=morgan_bits,
        is_train=True,
        pubchem_vectors=pubchem_vectors,
        mol2vec_vectors=mol2vec_vectors,
    )
    val_ds = GBCompatibilityDataset(
        csv_path=config.val_csv,
        prior_table=prior_table,
        descriptor_norm_stats=descriptor_norm_stats,
        family=family,
        morgan_bits=morgan_bits,
        is_train=False,
        pubchem_vectors=pubchem_vectors,
        mol2vec_vectors=mol2vec_vectors,
    )
    test_ds = GBCompatibilityDataset(
        csv_path=config.test_csv,
        prior_table=prior_table,
        descriptor_norm_stats=descriptor_norm_stats,
        family=family,
        morgan_bits=morgan_bits,
        is_train=False,
        pubchem_vectors=pubchem_vectors,
        mol2vec_vectors=mol2vec_vectors,
    )

    batch_size = config.gb_batch_size

    # No balanced sampler — plain shuffle (Section 3.5)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=gb_collate_fn,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=gb_collate_fn,
        drop_last=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=gb_collate_fn,
        drop_last=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader
