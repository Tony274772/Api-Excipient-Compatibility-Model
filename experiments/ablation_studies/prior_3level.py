"""
Excipient prior for the prior ablation study.

Baseline (src/pgb_prior.py) resolves an excipient prior through 4 levels:
    exact match -> hydrate/solvent-stripped family match -> Morgan-512 NN -> global
The ablation removes the family level:
    exact match -> Morgan-512 NN (Tanimoto >= 0.6) -> global (0.094)

Everything else (Laplace smoothing, canonicalisation, fingerprint, NN threshold,
leave-own-cluster-out priors for training rows) is identical to the baseline.
"""

import numpy as np
import pandas as pd
from rdkit import DataStructs

from src.pgb_prior import (
    GLOBAL_RATE,
    ExcipientPriorTable,
    _canonical_stereo_blind,
    _strip_hydrates_solvents,
    _morgan512,
)

NN_THRESHOLD = 0.6  # same Tanimoto cut-off as src/pgb_prior.py


class ThreeLevelPriorTable(ExcipientPriorTable):
    """
    ExcipientPriorTable without the family level.

    The family table is never used: it is cleared after fitting, so the saved
    prior_table.json also shows it as empty. The p_family slot of the returned
    dict is filled with the same value as p_exact (this is what the baseline
    already does in its NN and global branches), so the 5-d vector stays the
    same shape but carries no family information.
    """

    USE_FAMILY = False

    def fit(self, train_df: pd.DataFrame, leave_out_cluster_id=None, _precomputed_fps=None):
        super().fit(train_df, leave_out_cluster_id=leave_out_cluster_id,
                    _precomputed_fps=_precomputed_fps)
        self._family = {}

    def lookup(self, exc_smiles: str) -> dict:
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

        # Level 1: exact match
        if key in self._exact:
            p, n = self._exact[key]
            return {
                "p_exact": p,
                "p_delta": p - GLOBAL_RATE,
                "p_family": p,
                "log_n": float(np.log1p(n)),
                "unseen": 0.0,
            }

        # Level 2: Morgan-512 nearest neighbour (Tanimoto >= 0.6)
        best_sim, best_p = _nearest_neighbour(self, exc_smiles)
        if best_sim >= NN_THRESHOLD:
            return {
                "p_exact": best_p,
                "p_delta": best_p - GLOBAL_RATE,
                "p_family": best_p,
                "log_n": 0.0,
                "unseen": 0.0,
            }

        # Level 3: global fallback
        return {
            "p_exact": GLOBAL_RATE,
            "p_delta": 0.0,
            "p_family": GLOBAL_RATE,
            "log_n": 0.0,
            "unseen": 1.0,
        }


def _nearest_neighbour(table: ExcipientPriorTable, exc_smiles: str) -> tuple[float, float]:
    """Best Tanimoto similarity and that neighbour's prior (same loop as src/pgb_prior.py)."""
    fp_query = _morgan512(exc_smiles)
    best_sim, best_p = 0.0, GLOBAL_RATE
    if fp_query is None or not table._fps:
        return best_sim, best_p
    for stored_key, stored_fp in table._fps.items():
        sim = DataStructs.TanimotoSimilarity(fp_query, stored_fp)
        if sim > best_sim:
            best_sim = sim
            best_p, _ = table._exact[stored_key]
    return best_sim, best_p


def prior_level(table: ExcipientPriorTable, exc_smiles: str) -> str:
    """
    Which cascade level resolves this excipient: exact / family / nn / global / empty.
    Mirrors the lookup order of the table's class (family level only for the baseline).
    """
    if pd.isna(exc_smiles) or not exc_smiles:
        return "empty"
    key = _canonical_stereo_blind(exc_smiles)
    if key is None:
        key = exc_smiles
    if key in table._exact:
        return "exact"
    if getattr(table, "USE_FAMILY", True) and _strip_hydrates_solvents(exc_smiles) in table._family:
        return "family"
    best_sim, _ = _nearest_neighbour(table, exc_smiles)
    if best_sim >= NN_THRESHOLD:
        return "nn"
    return "global"


def build_leave_cluster_out_prior_vectors(train_df: pd.DataFrame, table_cls=ThreeLevelPriorTable,
                                          cluster_col: str = "cluster_id"):
    """
    Same as src.pgb_prior.build_leave_cluster_out_prior_vectors, but the table class
    is a parameter and the cascade level of every row is returned as well.

    Returns (priors, levels): a list of 5-d prior vectors and a list of level names,
    both parallel to train_df rows.
    """
    df = train_df.copy()
    df["exc_key"] = df["Excipient_Smiles"].astype(str).map(
        lambda x: _canonical_stereo_blind(x) if _canonical_stereo_blind(x) is not None else x
    )
    df["fam_key"] = df["Excipient_Smiles"].astype(str).map(_strip_hydrates_solvents)

    fps_cache = {}
    for key in df["exc_key"].unique():
        orig = df[df["exc_key"] == key]["Excipient_Smiles"].iloc[0]
        fps_cache[key] = _morgan512(orig)

    unique_clusters = df[cluster_col].unique() if cluster_col in df.columns else [None]
    tables = {}
    for c in unique_clusters:
        pt = table_cls()
        pt.fit(df, leave_out_cluster_id=c, _precomputed_fps=fps_cache)
        tables[c] = pt

    priors, levels = [], []
    for _, row in train_df.iterrows():
        exc_smi = row.get("Excipient_Smiles")
        if pd.isna(exc_smi):
            exc_smi = ""
        pt = tables[row.get(cluster_col)]
        res = pt.lookup(exc_smi)
        priors.append(np.array([
            res["p_exact"], res["p_delta"], res["p_family"], res["log_n"], res["unseen"]
        ], dtype=np.float32))
        levels.append(prior_level(pt, exc_smi))

    return priors, levels
