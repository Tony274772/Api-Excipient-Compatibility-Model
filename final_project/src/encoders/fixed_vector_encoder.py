"""Fixed-vector encoder – Section 5.5.

For encoders that output one fixed-length vector per molecule (mol2vec, PubChem FP,
or plain RDKit descriptors used alone as an encoder).
is_sequence_capable = False → model.py routes to concat fusion.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class FixedVectorEncoder(nn.Module):
    """Fixed-vector encoder from precomputed CSV vectors.
    
    Supports mol2vec (300-dim), pubchemfp (881-dim), or rdkit_descriptors (21-dim).
    Vectors are loaded from a CSV keyed by CID.
    """

    is_sequence_capable = False

    # Default output dimensions per source type
    _default_dims = {
        "mol2vec": 300,
        "pubchemfp": 881,
        "rdkit_descriptors": 21,
    }

    def __init__(self, source: str = "mol2vec", vector_path: str = None, device: str = "cpu"):
        super().__init__()
        self.source = source
        self.device = device

        # Set output_dim based on source
        self.output_dim = self._default_dims.get(source, 300)

        # Load precomputed vectors from CSV if provided
        self._vectors: dict[str, np.ndarray] = {}
        if vector_path and os.path.exists(vector_path):
            df = pd.read_csv(vector_path)
            
            # Detect key columns
            cols = list(df.columns)
            smiles_col = next((c for c in cols if "smiles" in c.lower()), None)
            cid_col = next((c for c in cols if "cid" in c.lower()), None)
            
            # Identify feature columns (numerical columns excluding identifier columns)
            exclude_cols = set([c for c in [smiles_col, cid_col] if c is not None])
            feature_cols = [c for c in cols if c not in exclude_cols]
            
            # If neither smiles nor cid column named explicitly, use first column as key
            if not exclude_cols:
                key_col = cols[0]
                feature_cols = cols[1:]
            else:
                key_col = smiles_col or cid_col

            self.output_dim = len(feature_cols)

            # Store vectors keyed by key_col and also by cid / smiles if present
            for _, row in df.iterrows():
                vec = row[feature_cols].values.astype(np.float32)
                if smiles_col:
                    self._vectors[str(row[smiles_col])] = vec
                if cid_col:
                    self._vectors[str(row[cid_col])] = vec
                if not smiles_col and not cid_col:
                    self._vectors[str(row[key_col])] = vec

        # Also load SMILES <-> CID mapping from data files so CID-based vectors can be looked up by SMILES
        self._smiles_to_cid: dict[str, str] = {}
        for fn in ["data/train.csv", "data/val.csv", "data/test.csv", "data/start_dataset.csv"]:
            if os.path.exists(fn):
                try:
                    df_data = pd.read_csv(fn)
                    if "API_Smiles" in df_data.columns and "API_CID" in df_data.columns:
                        for _, r in df_data[["API_Smiles", "API_CID"]].dropna().iterrows():
                            self._smiles_to_cid[str(r["API_Smiles"])] = str(int(r["API_CID"]) if isinstance(r["API_CID"], (int, float)) else r["API_CID"])
                    if "Excipient_Smiles" in df_data.columns and "Excipient_CID" in df_data.columns:
                        for _, r in df_data[["Excipient_Smiles", "Excipient_CID"]].dropna().iterrows():
                            self._smiles_to_cid[str(r["Excipient_Smiles"])] = str(int(r["Excipient_CID"]) if isinstance(r["Excipient_CID"], (int, float)) else r["Excipient_CID"])
                except Exception:
                    pass

    def to(self, device, *args, **kwargs):
        self.device = str(device)
        return super().to(device, *args, **kwargs)

    def encode(self, smiles_batch: list[str]) -> torch.Tensor:
        """Encode a batch of SMILES to fixed-length vectors.
        
        Returns:
            pooled_embedding: [B, D]
        """
        B = len(smiles_batch)
        D = self.output_dim

        vectors = torch.zeros(B, D, device=self.device)

        for i, smi in enumerate(smiles_batch):
            if smi in self._vectors:
                vectors[i] = torch.tensor(self._vectors[smi], device=self.device)
            elif smi in self._smiles_to_cid and self._smiles_to_cid[smi] in self._vectors:
                vectors[i] = torch.tensor(self._vectors[self._smiles_to_cid[smi]], device=self.device)
            # else: zeros (unknown molecule)

        return vectors
