"""PGB (Prior-Gated Bilinear) experiment package.

Implements the PGB architecture for all 9 model families:
- 4 fixed-vector families: maccs, morgan, pubchemfp, mol2vec
- 5 sequence/graph families: molformer, chemberta, gin, gat, dmpnn

This package keeps PGB isolated as its own architecture alongside experiments/gated_bilinear.
"""

from experiments.pgb.config import PGBConfig, FAMILIES
from experiments.pgb.model import MoleculeTower, PGBHead, PGBCompatibilityModel, pgb_model_from_config
from experiments.pgb.prior import ExcipientPriorTable, build_leave_cluster_out_prior_vectors
from experiments.pgb.dataset import PGBDataset, pgb_collate_fn, PGBNormStats

__all__ = [
    "PGBConfig",
    "FAMILIES",
    "MoleculeTower",
    "PGBHead",
    "PGBCompatibilityModel",
    "pgb_model_from_config",
    "ExcipientPriorTable",
    "build_leave_cluster_out_prior_vectors",
    "PGBDataset",
    "pgb_collate_fn",
    "PGBNormStats",
]
