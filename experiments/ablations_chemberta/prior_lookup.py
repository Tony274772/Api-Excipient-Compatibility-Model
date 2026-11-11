"""Excipient prior tables for the ablation harness (port of final_project/apiexc/ablations/prior_lookup.py
to src/pgb_prior.py, which the encoder grid uses).

    full        src.pgb_prior.ExcipientPriorTable: exact -> family -> nearest neighbour -> global
    no_family   exact -> nearest neighbour (Morgan-512 Tanimoto >= 0.6) -> global

Both use the train positive rate as the global rate, as run_grid.py does. The leave-own-cluster-out
priors of the TRAIN rows are built with the same table class, so `no_family` changes train, val and
held-out priors consistently.
"""

import numpy as np
import pandas as pd
from rdkit import DataStructs

from src.pgb_prior import (ExcipientPriorTable, _canonical_stereo_blind, _morgan512,
                           _strip_hydrates_solvents)


class NoFamilyPriorTable(ExcipientPriorTable):
    """ExcipientPriorTable whose lookup skips the family fallback. For an exact match p_family is
    set to p_exact. Fitting, saving and loading are unchanged."""

    def _result(self, p, p_family, log_n, unseen):
        return {"p_exact": p, "p_delta": p - self.global_rate, "p_family": p_family,
                "log_n": log_n, "unseen": unseen}

    def lookup(self, exc_smiles: str) -> dict:
        if not self._fitted:
            raise RuntimeError("Call fit() before lookup().")
        g = self.global_rate
        if pd.isna(exc_smiles) or not exc_smiles:
            return {"p_exact": g, "p_delta": 0.0, "p_family": g, "log_n": 0.0, "unseen": 1.0}

        key = _canonical_stereo_blind(exc_smiles)
        if key is None:
            key = exc_smiles
        if key in self._exact:
            p, n = self._exact[key]
            return self._result(p, p, float(np.log1p(n)), 0.0)

        fp_query = _morgan512(exc_smiles)
        if fp_query is not None and self._fps:
            best_sim, best_p = 0.0, g
            for stored_key, stored_fp in self._fps.items():
                sim = DataStructs.TanimotoSimilarity(fp_query, stored_fp)
                if sim > best_sim:
                    best_sim = sim
                    best_p, _ = self._exact[stored_key]
            if best_sim >= 0.6:
                return self._result(best_p, best_p, 0.0, 0.0)

        return self._result(g, g, 0.0, 1.0)


def table_class(lookup: str):
    if lookup in (None, "full"):
        return ExcipientPriorTable
    if lookup == "no_family":
        return NoFamilyPriorTable
    raise ValueError(f"unknown prior lookup {lookup!r}")


def leave_cluster_out_priors(train_df: pd.DataFrame, global_rate: float, lookup: str = "full",
                             cluster_col: str = "cluster_id"):
    """src.pgb_prior.build_leave_cluster_out_prior_vectors with a choice of table class: every
    training row gets its excipient prior from the training rows outside its own API cluster."""
    if lookup in (None, "full"):
        # the exact function run_grid.py calls, so `full` reproduces the grid bit for bit
        from src.pgb_prior import build_leave_cluster_out_prior_vectors
        return build_leave_cluster_out_prior_vectors(train_df, cluster_col=cluster_col,
                                                     global_rate=global_rate)
    cls = table_class(lookup)
    df = train_df.copy()
    df["exc_key"] = df["Excipient_Smiles"].astype(str).map(
        lambda x: _canonical_stereo_blind(x) if _canonical_stereo_blind(x) is not None else x)
    df["fam_key"] = df["Excipient_Smiles"].astype(str).map(_strip_hydrates_solvents)
    fps_cache = {}
    for key in df["exc_key"].unique():
        fps_cache[key] = _morgan512(df[df["exc_key"] == key]["Excipient_Smiles"].iloc[0])

    clusters = df[cluster_col].unique() if cluster_col in df.columns else [None]
    tables = {}
    for c in clusters:
        pt = cls(global_rate=global_rate)
        pt.fit(df, leave_out_cluster_id=c, _precomputed_fps=fps_cache)
        tables[c] = pt

    priors = []
    for _, row in train_df.iterrows():
        exc = row.get("Excipient_Smiles")
        res = tables[row.get(cluster_col)].lookup("" if pd.isna(exc) else exc)
        priors.append(np.array([res["p_exact"], res["p_delta"], res["p_family"], res["log_n"],
                                res["unseen"]], dtype=np.float32))
    return priors
