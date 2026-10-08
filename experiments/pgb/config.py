"""Configuration for Prior-Gated Bilinear (PGB) architecture experiments."""

import os
from dataclasses import dataclass
from typing import Literal, Optional

from src.config import Config

FAMILIES = {
    "maccs":      {"encoder": "fixed_vector", "fixed_vector_source": "maccs",     "fusion": "concat", "pgb_max_epochs": 40},
    "morgan":     {"encoder": "fixed_vector", "fixed_vector_source": "morgan",    "fusion": "concat", "pgb_max_epochs": 40},
    "pubchemfp":  {"encoder": "fixed_vector", "fixed_vector_source": "pubchemfp", "fusion": "concat", "pgb_max_epochs": 40},
    "mol2vec":    {"encoder": "fixed_vector", "fixed_vector_source": "mol2vec",   "fusion": "concat", "pgb_max_epochs": 40},
    "molformer":  {"encoder": "molformer",  "fusion": "cross_attn", "pooling": "global_gated_attention", "pgb_max_epochs": 60},
    "chemberta":  {"encoder": "chemberta",  "fusion": "cross_attn", "pooling": "cls",                    "pgb_max_epochs": 60},
    "gin":        {"encoder": "pretrained_gin", "fusion": "cross_attn", "pooling": "global_gated_attention", "pgb_max_epochs": 60},
    "gat":        {"encoder": "pretrained_gat", "fusion": "cross_attn", "pooling": "global_gated_attention", "pgb_max_epochs": 60},
    "dmpnn":      {"encoder": "dmpnn_chemprop", "fusion": "cross_attn", "pooling": "cls",                "pgb_max_epochs": 60},
}


@dataclass
class PGBConfig(Config):
    """Configuration class for the PGB architecture."""

    use_pgb_head: bool = True
    split_type: str = "cluster"
    family: str = "maccs"
    pgb_metrics_root: str = "experiments/results/metrics_pgb"
    pgb_checkpoint_root: str = "checkpoints/pgb"

    def apply_family_spec(self, family_name: str):
        if family_name not in FAMILIES:
            raise ValueError(f"Unknown family '{family_name}'. Choices: {list(FAMILIES.keys())}")
        self.family = family_name
        spec = FAMILIES[family_name]
        for k, v in spec.items():
            setattr(self, k, v)
        self.checkpoint_dir = f"{self.pgb_checkpoint_root}/{family_name}"
        self.metrics_dir = f"{self.pgb_metrics_root}/{family_name}"
        self.resolve_paths()
