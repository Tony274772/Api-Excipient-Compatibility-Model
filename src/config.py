"""Full configuration schema – Section 10 of the spec."""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch


@dataclass
class Config:
    # --- Data paths ---
    data_dir: str = "data"
    train_csv: Optional[str] = None
    val_csv: Optional[str] = None
    test_csv: Optional[str] = None
    api_descriptors_path: str = "data/api_descriptors.csv"
    excipient_descriptors_path: str = "data/excipient_descriptors.csv"
    descriptor_norm_stats_path: str = "models/descriptor_norm_stats.json"
    reaction_lookup_path: str = "data/reaction_lookup.csv"

    # --- Axis A: encoder selection ---
    encoder: Literal["molformer", "pretrained_gin", "chemberta", "fixed_vector"] = "molformer"
    molformer_model_path: str = "models/pretrained/molformer" if os.path.isdir("models/pretrained/molformer") else "ibm/MoLFormer-XL-both-10pct"
    chemberta_model_path: str = "models/pretrained/chemberta" if os.path.isdir("models/pretrained/chemberta") else "DeepChem/ChemBERTa-77M-MTR"
    gin_pretrained_name: str = "models/pretrained/gin/gin_supervised_contextpred_pre_trained.pth" if os.path.isfile("models/pretrained/gin/gin_supervised_contextpred_pre_trained.pth") else "gin_supervised_contextpred"
    fixed_vector_source: Literal["mol2vec", "pubchemfp", "rdkit_descriptors"] = "mol2vec"
    fixed_vector_path: Optional[str] = None   # csv of precomputed vectors, keyed by CID
    encoder_output_dim: int = 768              # auto-set per encoder at model build time

    # --- Axis B: fusion ---
    fusion: Literal["cross_attn", "concat"] = "cross_attn"  # forced to "concat" if encoder is not sequence-capable
    pooling: Literal["global_gated_attention", "cls", "explicit_pairwise"] = "global_gated_attention"
    pool_hidden_dim: int = 128
    pair_pool_hidden_dim: int = 256
    proj_dim: int = 128
    num_heads: int = 8
    attn_dropout: float = 0.15
    proj_dropout: float = 0.15

    # --- Classifier head ---
    clf_dropout_1: float = 0.5
    clf_dropout_2: float = 0.4
    clf_hidden_dim: int = 128
    clf_hidden_dim_2: int = 64

    # --- Descriptors ---
    use_descriptors: bool = True
    num_descriptors: int = 21
    desc_proj_dim: int = 24
    desc_dropout: float = 0.15

    # --- Axis C: loss ---
    loss: Literal["bce", "weighted_bce", "focal", "asl"] = "asl"
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # --- Axis D: lookup-table augmentation (Phase 2, all off by default) ---
    use_weak_labels: bool = False
    weak_label_sample_weight: float = 0.3
    use_mechanism_features: bool = False
    use_aux_head: bool = False
    aux_head_weight: float = 0.3

    # --- Class imbalance ---
    positive_prior: float = 0.0  # computed fresh from train.csv at runtime
    use_balanced_sampler: bool = True

    # --- Training ---
    lr: float = 1.5e-4
    weight_decay: float = 8e-4
    grad_clip_norm: float = 1.0
    batch_size: int = 64
    max_epochs: int = 150
    early_stop_patience: int = 6
    early_stop_min_delta: float = 0.002
    lr_patience: int = 8
    lr_factor: float = 0.5

    # --- Evaluation ---
    threshold_step: float = 0.001

    # --- Reproducibility / system ---
    seed: int = 42
    device: str = "auto"
    split_type: Literal["cluster", "random"] = "cluster"
    checkpoint_dir: str = "checkpoints/molformer"   # set per-run
    metrics_dir: str = "metrics/molformer"           # set per-run

    def __post_init__(self):
        self.resolve_paths()

    def resolve_csv_paths(self):
        """Only fills in train/val/test CSV defaults. Safe to call any number of times."""
        data_path = f"{self.data_dir}/random_split" if self.split_type == "random" else self.data_dir
        if self.train_csv is None:
            self.train_csv = f"{data_path}/train.csv"
        if self.val_csv is None:
            self.val_csv = f"{data_path}/val.csv"
        if self.test_csv is None:
            self.test_csv = f"{data_path}/test.csv"

    def resolve_checkpoint_paths(self):
        """Derive checkpoint_dir and metrics_dir from encoder/fusion/loss/pooling."""
        encoder_name = self.encoder
        if self.encoder == "fixed_vector":
            encoder_name = f"fixed_vector_{self.fixed_vector_source}"

        if self.fusion == "cross_attn":
            if self.pooling == "explicit_pairwise":
                pool_str = "pairwise"
            elif self.pooling == "global_gated_attention":
                pool_str = "global_gated"
            else:
                pool_str = self.pooling
            combo = f"{encoder_name}_{self.fusion}_{pool_str}_{self.loss}"
        else:
            combo = f"{encoder_name}_{self.fusion}_{self.loss}"
        
        prefix = "random_split/" if self.split_type == "random" else ""
        self.checkpoint_dir = f"checkpoints/{prefix}{combo}"
        self.metrics_dir = f"metrics/{prefix}{combo}"

    def resolve_paths(self):
        """Back-compat wrapper: old callers that expect resolve_paths() to do both."""
        self.resolve_checkpoint_paths()
        self.resolve_csv_paths()
