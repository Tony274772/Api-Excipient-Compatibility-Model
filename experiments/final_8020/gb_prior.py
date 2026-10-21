"""
Leave-own-API-cluster-out excipient priors for Gated Bilinear training rows.

experiments/gated_bilinear only subtracts the row's own API cluster from the exact-key count of the prior
block. The family prior and the logit-offset prior `p_exact` (logit = residual + s * logit(p_exact) + b)
always use all training rows, so each training row's prior contains its own label.

Here every prior quantity of a training row (prior block and p_exact) comes from the train rows outside its
API cluster: the exact-key and family-key counts of that cluster are subtracted, and keys left with no rows
are removed (so they also drop out of the nearest-neighbour fallback). This is the same rule as PGB's
build_leave_cluster_out_prior_vectors, which refits the prior table without the cluster.
Val and held-out rows are unchanged: they use the prior table of all training rows.
"""

import copy

import numpy as np
import pandas as pd
import torch

from experiments.gated_bilinear.dataset import GBCompatibilityDataset
from experiments.gated_bilinear.prior import _family_key, _stereo_blind_canonical


def _subtract(stats: dict, key: str, label: int):
    n_pos, n_total = stats[key]
    n_pos, n_total = n_pos - label, n_total - 1
    if n_total < 0 or n_pos < 0:
        raise RuntimeError(f"negative prior count for {key!r}")
    if n_total == 0:
        del stats[key]
    else:
        stats[key] = (n_pos, n_total)


def leave_cluster_out_table(full_table, exc_smiles, labels):
    """Copy of `full_table` with the given rows (one API cluster) removed from every count."""
    t = copy.copy(full_table)
    t.exact_stats = dict(full_table.exact_stats)
    t.family_stats = dict(full_table.family_stats)
    for smi, y in zip(exc_smiles, labels):
        _subtract(t.exact_stats, _stereo_blind_canonical(smi), int(y))
        _subtract(t.family_stats, _family_key(smi), int(y))
    t.nn_fps = {k: v for k, v in full_table.nn_fps.items() if k in t.exact_stats}
    t.cluster_stats = {}
    return t


def build_gb_train_priors(full_table, train_df: pd.DataFrame, cluster_col: str = "cluster_id"):
    """[(prior_block[5], p_exact)] per train row, from the train rows outside the row's API cluster."""
    if cluster_col not in train_df.columns or train_df[cluster_col].isna().any():
        raise RuntimeError(f"train rows need a complete {cluster_col!r} column for leave-own-cluster-out priors")
    exc = train_df["Excipient_Smiles"].fillna("").astype(str).tolist()
    labels = train_df["Outcome1"].astype(int).tolist()
    out = [None] * len(train_df)
    for _, idx in train_df.groupby(cluster_col).indices.items():
        t = leave_cluster_out_table(full_table, [exc[i] for i in idx], [labels[i] for i in idx])
        for i in idx:
            out[i] = (t.get_prior_block(exc[i], api_cluster=None, is_train=False).astype(np.float32),
                      float(t.get_logit_offset_p(exc[i])))
    return out


class GBLeaveClusterOutDataset(GBCompatibilityDataset):
    """GBCompatibilityDataset whose prior_block / p_exact come from precomputed leave-cluster-out priors."""

    def __init__(self, *args, row_priors, expected_df: pd.DataFrame, **kwargs):
        super().__init__(*args, **kwargs)
        same = (len(self.df) == len(expected_df)
                and (self.df["API_CID"].values == expected_df["API_CID"].values).all()
                and (self.df["Excipient_CID"].values == expected_df["Excipient_CID"].values).all())
        if not same or len(row_priors) != len(self.df):
            raise RuntimeError("row priors are not aligned with the training rows")
        self._row_priors = row_priors

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        prior_block, p_exact = self._row_priors[idx]
        item["prior_block"] = torch.tensor(prior_block, dtype=torch.float32)
        item["p_exact"] = torch.tensor(p_exact, dtype=torch.float32)
        return item
