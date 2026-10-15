"""
src/pgb_prior.py
Laplace-smoothed per-excipient incompatibility rate computed from training rows.
Used by PGBCompatibilityModel to provide the logit offset.
"""

import json
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

GLOBAL_RATE = 0.094          # global incompatibility rate across the full dataset
LAPLACE_ALPHA = 4            # Laplace smoothing count
LAPLACE_PRIOR = GLOBAL_RATE  # pseudocount prior


def _canonical_stereo_blind(smiles: str) -> str | None:
    """
    Canonical SMILES with stereo info stripped.
    Returns None if RDKit cannot parse the string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def _strip_hydrates_solvents(smiles: str) -> str:
    """
    Remove water and common solvent fragments from a SMILES.
    Returns the heaviest remaining fragment, stereo-blind canonical.
    Solvent/water patterns: O (water), CO (methanol), CCO (ethanol), OCC (same).
    """
    STRIP = {"O", "[H]O[H]", "CO", "CCO", "OCC", "CC(O)=O"}
    frags = smiles.split(".")
    # Remove known solvents
    keep = [f for f in frags if f.strip() not in STRIP]
    if not keep:
        keep = frags  # nothing to strip, keep original
    # Return canonical of heaviest fragment
    best = sorted(keep, key=len, reverse=True)[0]
    return _canonical_stereo_blind(best) or best


def _morgan512(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    return gen.GetFingerprint(mol)


class ExcipientPriorTable:
    """
    Holds a train-derived per-excipient incompatibility rate.
    Three lookup keys per excipient (exact → family → nearest-neighbour → global).
    """

    def __init__(self):
        self._exact: dict[str, tuple[float, int]] = {}   # key → (p, n)
        self._family: dict[str, tuple[float, int]] = {}  # stripped key → (p_family, n)
        self._fps: dict[str, object] = {}                 # exact_key → Morgan fp
        self._fitted = False

    def fit(self, train_df: pd.DataFrame, leave_out_cluster_id=None, _precomputed_fps=None):
        """
        Compute the prior table from train_df rows.
        train_df must have columns: Excipient_Smiles, Outcome1, [cluster_id optional].

        If leave_out_cluster_id is given, rows whose API belongs to that cluster
        are excluded from the prior computation (leave-own-cluster-out for training rows).
        """
        df = train_df
        if leave_out_cluster_id is not None and "cluster_id" in df.columns:
            df = df[df["cluster_id"] != leave_out_cluster_id]

        if df.empty:
            self._fitted = True
            return

        # Canonicalize and extract family key
        if "exc_key" not in df.columns:
            df = df.copy()
            df["exc_key"] = df["Excipient_Smiles"].astype(str).map(
                lambda x: _canonical_stereo_blind(x) if _canonical_stereo_blind(x) is not None else x
            )
            df["fam_key"] = df["Excipient_Smiles"].astype(str).map(_strip_hydrates_solvents)

        # Aggregate by canonical key (fixes Fix #4)
        exact_agg = df.groupby("exc_key").agg(n=("Outcome1", "size"), pos=("Outcome1", "sum"))
        for key, row in exact_agg.iterrows():
            pos, n = row["pos"], row["n"]
            p = (pos + LAPLACE_ALPHA * LAPLACE_PRIOR) / (n + LAPLACE_ALPHA)
            self._exact[key] = (float(p), int(n))
            
            # Save Morgan fingerprint for nearest neighbour
            if _precomputed_fps is not None and key in _precomputed_fps:
                fp = _precomputed_fps[key]
            else:
                orig_smiles = df[df["exc_key"] == key]["Excipient_Smiles"].iloc[0]
                fp = _morgan512(orig_smiles)
            if fp is not None:
                self._fps[key] = fp

        # Aggregate by family key (fixes Fix #5)
        fam_agg = df.groupby("fam_key").agg(n=("Outcome1", "size"), pos=("Outcome1", "sum"))
        for fam_key, row in fam_agg.iterrows():
            pos, n = row["pos"], row["n"]
            p = (pos + LAPLACE_ALPHA * LAPLACE_PRIOR) / (n + LAPLACE_ALPHA)
            self._family[fam_key] = (float(p), int(n))

        self._fitted = True

    def lookup(self, exc_smiles: str) -> dict:
        """
        Return a dict with keys:
            p_exact       float  Laplace-smoothed incompatibility rate
            p_delta       float  p_exact - GLOBAL_RATE
            p_family      float  hydrate-stripped family rate
            log_n         float  log(1 + n)
            unseen        float  1.0 if unseen, 0.0 otherwise

        Call this at inference time (for every batch row).
        p_exact is the value used as the logit offset.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before lookup().")

        if pd.isna(exc_smiles) or not exc_smiles:
            return {
                "p_exact": GLOBAL_RATE, "p_delta": 0.0, "p_family": GLOBAL_RATE,
                "log_n": 0.0, "unseen": 1.0,
            }

        key = _canonical_stereo_blind(exc_smiles)
        if key is None:
            key = exc_smiles

        fam_key = _strip_hydrates_solvents(exc_smiles)

        # Exact match
        if key in self._exact:
            p, n = self._exact[key]
            p_fam = self._family.get(fam_key, (p, n))[0]
            return {
                "p_exact": p,
                "p_delta": p - GLOBAL_RATE,
                "p_family": p_fam,
                "log_n": float(np.log1p(n)),
                "unseen": 0.0,
            }

        # Family key fallback
        if fam_key in self._family:
            p_fam, n_fam = self._family[fam_key]
            return {
                "p_exact": p_fam,
                "p_delta": p_fam - GLOBAL_RATE,
                "p_family": p_fam,
                "log_n": 0.0,
                "unseen": 0.0,
            }

        # Nearest-neighbour fallback (Tanimoto >= 0.6)
        fp_query = _morgan512(exc_smiles)
        if fp_query is not None and self._fps:
            from rdkit import DataStructs
            best_sim = 0.0
            best_p = GLOBAL_RATE
            for stored_key, stored_fp in self._fps.items():
                sim = DataStructs.TanimotoSimilarity(fp_query, stored_fp)
                if sim > best_sim:
                    best_sim = sim
                    best_p, _ = self._exact[stored_key]
            if best_sim >= 0.6:
                return {
                    "p_exact": best_p,
                    "p_delta": best_p - GLOBAL_RATE,
                    "p_family": best_p,
                    "log_n": 0.0,
                    "unseen": 0.0,
                }

        # Global fallback
        return {
            "p_exact": GLOBAL_RATE,
            "p_delta": 0.0,
            "p_family": GLOBAL_RATE,
            "log_n": 0.0,
            "unseen": 1.0,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        # Fix #6: Save the data needed to rebuild nearest-neighbor FPs
        fps_data = {}
        for k, fp in self._fps.items():
            fps_data[k] = list(fp.GetOnBits())
            
        data = {
            "exact": {k: list(v) for k, v in self._exact.items()},
            "family": {k: list(v) for k, v in self._family.items()},
            "fps_on_bits": fps_data,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExcipientPriorTable":
        t = cls()
        with open(path) as f:
            data = json.load(f)
        t._exact = {k: tuple(v) for k, v in data["exact"].items()}
        t._family = {k: tuple(v) for k, v in data["family"].items()}
        
        # Reconstruct FPs from saved ON bits
        from rdkit.DataStructs import ExplicitBitVect
        t._fps = {}
        if "fps_on_bits" in data:
            for k, on_bits in data["fps_on_bits"].items():
                fp = ExplicitBitVect(512)
                for bit in on_bits:
                    fp.SetBit(bit)
                t._fps[k] = fp
                
        t._fitted = True
        return t

def build_leave_cluster_out_prior_vectors(train_df: pd.DataFrame, cluster_col="cluster_id"):
    """
    For every training row, compute its excipient prior using
    all training rows except rows belonging to the same API cluster.
    Returns a list of 5-d prior vectors parallel to train_df rows.
    """
    priors = []
    
    # Precompute keys and fingerprints to dramatically speed up table fitting
    df = train_df.copy()
    df["exc_key"] = df["Excipient_Smiles"].astype(str).map(
        lambda x: _canonical_stereo_blind(x) if _canonical_stereo_blind(x) is not None else x
    )
    df["fam_key"] = df["Excipient_Smiles"].astype(str).map(_strip_hydrates_solvents)
    
    unique_keys = df["exc_key"].unique()
    fps_cache = {}
    for key in unique_keys:
        orig = df[df["exc_key"] == key]["Excipient_Smiles"].iloc[0]
        fps_cache[key] = _morgan512(orig)
    
    unique_clusters = df[cluster_col].unique() if cluster_col in df.columns else [None]
    tables = {}
    for c in unique_clusters:
        pt = ExcipientPriorTable()
        pt.fit(df, leave_out_cluster_id=c, _precomputed_fps=fps_cache)
        tables[c] = pt
        
    for _, row in train_df.iterrows():
        c = row.get(cluster_col)
        exc_smi = row.get("Excipient_Smiles")
        if pd.isna(exc_smi):
            exc_smi = ""
        
        pt = tables[c]
        res = pt.lookup(exc_smi)
        priors.append(np.array([
            res["p_exact"], res["p_delta"], res["p_family"], res["log_n"], res["unseen"]
        ], dtype=np.float32))
        
    return priors
