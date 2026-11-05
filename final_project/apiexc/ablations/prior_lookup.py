"""
Excipient prior tables for the ablation harness.

apiexc/features/excipient_prior.py fits a Laplace-smoothed incompatibility rate per excipient and looks an excipient up as
exact -> family -> nearest neighbour -> global. Its global rate (smoothing pseudocount, unseen fallback,
p_delta reference) is the train positive rate (276 / 2770 = 0.0996), saved in prior_table.json; tables saved
before that load with the old hard-coded 0.094.

The harness uses the train rate for every new run. Two data settings change the table:

    no_family     lookup exact -> nearest neighbour (Morgan-512 Tanimoto >= 0.6) -> global. The family key
                  strips water / solvents and keeps the longest SMILES fragment, so for salts it groups every
                  metal salt of the same anion (all phosphates, all stearates).
    old_global    the old global rate 0.094, to reproduce runs made before the fix.
"""

import numpy as np
import pandas as pd
from rdkit import DataStructs

from apiexc.features.excipient_prior import GLOBAL_RATE, ExcipientPriorTable, _canonical_stereo_blind, _morgan512, _strip_hydrates_solvents


class NoFamilyPriorTable(ExcipientPriorTable):
    """ExcipientPriorTable whose lookup skips the family fallback: exact -> nearest neighbour -> global.
    For an exact match p_family is set to p_exact. Fitting, saving and loading are unchanged."""

    def _result(self, p, p_family, log_n, unseen):
        return {"p_exact": p, "p_delta": p - self.global_rate, "p_family": p_family, "log_n": log_n,
                "unseen": unseen}

    def lookup(self, exc_smiles: str) -> dict:
        if not self._fitted:
            raise RuntimeError("Call fit() before lookup().")

        g = self.global_rate
        if pd.isna(exc_smiles) or not exc_smiles:
            return self._result(g, g, 0.0, 1.0)

        key = _canonical_stereo_blind(exc_smiles)
        if key is None:
            key = exc_smiles

        # Exact match
        if key in self._exact:
            p, n = self._exact[key]
            return self._result(p, p, float(np.log1p(n)), 0.0)

        # Nearest-neighbour fallback (Tanimoto >= 0.6), same rule as ExcipientPriorTable
        fp_query = _morgan512(exc_smiles)
        if fp_query is not None and self._fps:
            best_sim = 0.0
            best_p = g
            for stored_key, stored_fp in self._fps.items():
                sim = DataStructs.TanimotoSimilarity(fp_query, stored_fp)
                if sim > best_sim:
                    best_sim = sim
                    best_p, _ = self._exact[stored_key]
            if best_sim >= 0.6:
                return self._result(best_p, best_p, 0.0, 0.0)

        # Global fallback
        return self._result(g, g, 0.0, 1.0)


def prior_table_class(config):
    """ExcipientPriorTable, or NoFamilyPriorTable when config.prior_lookup == 'no_family'."""
    return NoFamilyPriorTable if getattr(config, "prior_lookup", "full") == "no_family" else ExcipientPriorTable


def prior_global_rate(config):
    """Global rate of a run: config.prior_global_rate_value (set at train time); 0.094 for runs made before."""
    return float(getattr(config, "prior_global_rate_value", GLOBAL_RATE))


def new_prior_table(config):
    """An empty prior table for this run (lookup and global rate from the config)."""
    return prior_table_class(config)(global_rate=prior_global_rate(config))


def load_prior_table(config, path):
    """Load a saved table with this run's lookup. The rate comes from the config, which also covers tables
    saved before the rate was written to prior_table.json."""
    table = prior_table_class(config).load(path)
    table.global_rate = prior_global_rate(config)
    return table


def build_leave_cluster_out_prior_vectors(train_df: pd.DataFrame, config, cluster_col="cluster_id"):
    """apiexc.features.excipient_prior.build_leave_cluster_out_prior_vectors with this run's table (lookup and global rate):
    every training row gets its excipient prior from all training rows outside its own API cluster."""
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
        pt = new_prior_table(config)
        pt.fit(df, leave_out_cluster_id=c, _precomputed_fps=fps_cache)
        tables[c] = pt

    priors = []
    for _, row in train_df.iterrows():
        exc_smi = row.get("Excipient_Smiles")
        if pd.isna(exc_smi):
            exc_smi = ""
        res = tables[row.get(cluster_col)].lookup(exc_smi)
        priors.append(np.array([res["p_exact"], res["p_delta"], res["p_family"], res["log_n"], res["unseen"]],
                               dtype=np.float32))
    return priors
