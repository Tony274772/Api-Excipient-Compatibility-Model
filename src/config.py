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
    encoder: Literal[
        "molformer", "pretrained_gin", "chemberta", "fixed_vector",
        "dmpnn_chemprop", "dmpnn_scratch",
        "attentivefp", "gine", "gatv2", "pna", "pretrained_gat"
    ] = "molformer"
    molformer_model_path: str = "models/pretrained/molformer" if os.path.isdir("models/pretrained/molformer") else "ibm/MoLFormer-XL-both-10pct"
    chemberta_model_path: str = "models/pretrained/chemberta" if os.path.isdir("models/pretrained/chemberta") else "DeepChem/ChemBERTa-77M-MTR"
    gin_pretrained_name: str = "models/pretrained/gin/gin_supervised_contextpred_pre_trained.pth" if os.path.isfile("models/pretrained/gin/gin_supervised_contextpred_pre_trained.pth") else "gin_supervised_contextpred"
    fixed_vector_source: Literal["mol2vec", "pubchemfp", "rdkit_descriptors", "maccs", "morgan"] = "mol2vec"
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

    # --- PGB (Prior-Gated Bilinear) head ---
    use_pgb_head: bool = False              # enable PGB head and all associated features
    pgb_tower_dim: int = 96                 # output dimension of each molecule tower
    pgb_bilinear_rank: int = 16             # rank of the low-rank bilinear term
    pgb_head_hidden_1: int = 128            # MLP layer 1 width
    pgb_head_hidden_2: int = 48             # MLP layer 2 width
    pgb_head_dropout_1: float = 0.30        # dropout after layer 1
    pgb_head_dropout_2: float = 0.20        # dropout after layer 2
    pgb_prior_scale_init: float = 0.8       # initial value of learned prior scale s
    pgb_prior_offset: float = 0.2           # fraction of global log-odds added to bias init
    pgb_num_flags: int = 18                 # number of chemistry flags (do not change)
    pgb_num_mechanisms: int = 5             # number of SMARTS mechanism rules (do not change)
    pgb_num_desc: int = 12                  # number of PGB descriptors (do not change)
    pgb_prior_vec_dim: int = 5              # [p_exact, p_delta, p_family, log_n, unseen]

    # --- PGB training overrides (only active when use_pgb_head=True) ---
    pgb_lr: float = 1.2e-3                  # LR for new PGB layers (tower, head)
    pgb_encoder_lr: float = 3e-4            # LR for kept encoder layers (proj, cross-attn)
    pgb_weight_decay: float = 2e-3
    pgb_max_epochs: int = 60                # 40 for fixed-vector, 60 for seq. encoders
    pgb_early_stop_patience: int = 7
    pgb_lr_patience: int = 4
    pgb_batch_size: int = 64
    pgb_use_balanced_sampler: bool = False   # MUST stay False — no double weighting
    pgb_loss: str = "asym_focal"             # "asym_focal" is the only valid PGB loss
    pgb_focal_gamma_pos: float = 1.2
    pgb_focal_gamma_neg: float = 2.5
    pgb_tnr_floor: float = 0.97             # min true-negative-rate for threshold selection
    pgb_prior_table_path: str = ""          # set at runtime per fold

    # --- Axis C: loss ---
    loss: Literal["bce", "weighted_bce", "focal", "asl", "asym_focal"] = "asl"
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

    # --- New trainable molecular GNN settings ---
    gnn_hidden_dim: int = 300
    gnn_node_feature_dim: int = 39
    gnn_edge_feature_dim: int = 10
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.10

    # D-MPNN specific
    dmpnn_depth: int = 3
    dmpnn_bias: bool = False
    dmpnn_undirected: bool = False
    dmpnn_use_pretrained: bool = True
    dmpnn_pretrained_checkpoint: Optional[str] = "models/pretrained/chemeleon_mp.pt"
    dmpnn_encoder_lr: float = 1.0e-5  # Separate LR for CheMeleon pretrained encoder

    # AttentiveFP specific
    attentivefp_num_layers: int = 2
    attentivefp_num_timesteps: int = 2

    # GATv2 specific
    gatv2_heads: int = 4
    gatv2_negative_slope: float = 0.2

    # GINE specific
    gine_train_eps: bool = True
    gine_eps: float = 0.0

    # PNA specific
    pna_towers: int = 5
    pna_pre_layers: int = 1
    pna_post_layers: int = 1
    pna_divide_input: bool = False
    pna_aggregators: tuple = ("mean", "min", "max", "std")
    pna_scalers: tuple = ("identity", "amplification", "attenuation")
    pna_degree_hist: Optional[list] = None  # computed from training data at runtime

    # Feature schema version guard
    gnn_feature_schema: str = "attentivefp39_edge10_v1"

    # Pretrained GNN configuration
    gnn_pretrained_frozen: bool = True
    gnn_offline: bool = False

    # CheMeleon provenance tracking
    chemeleon_checkpoint_path: Optional[str] = None
    chemeleon_checkpoint_md5: Optional[str] = None
    chemeleon_source_url: str = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
    
    # Stanford GAT specific
    stanford_gat_checkpoint_path: str = "models/pretrained/stanford_gat/gat_contextpred.pth"
    stanford_gat_source_repo: str = "https://github.com/snap-stanford/pretrain-gnns"

    def __post_init__(self):
        self.resolve_paths()

    def resolve_csv_paths(self):
        """Only fills in train/val/test CSV defaults. Safe to call any number of times."""
        data_path = f"{self.data_dir}/random_split" if self.split_type == "random" else self.data_dir
        if self.train_csv is None or self.train_csv in [f"{self.data_dir}/train.csv", f"{self.data_dir}/random_split/train.csv"]:
            self.train_csv = f"{data_path}/train.csv"
        if self.val_csv is None or self.val_csv in [f"{self.data_dir}/val.csv", f"{self.data_dir}/random_split/val.csv"]:
            self.val_csv = f"{data_path}/val.csv"
        if self.test_csv is None or self.test_csv in [f"{self.data_dir}/test.csv", f"{self.data_dir}/random_split/test.csv"]:
            self.test_csv = f"{data_path}/test.csv"
        if self.fixed_vector_path is None and self.encoder == "fixed_vector":
            path_map = {
                "pubchemfp": f"{self.data_dir}/pubchem_fps.csv",
                "maccs": f"{self.data_dir}/maccs_keys.csv",
                "mol2vec": f"{self.data_dir}/mol2vec_embeddings.csv",
                "morgan": f"{self.data_dir}/morgan_fps.csv",
            }
            self.fixed_vector_path = path_map.get(self.fixed_vector_source)

    def resolve_checkpoint_paths(self):
        """Derive checkpoint_dir and metrics_dir from encoder/fusion/loss/pooling."""
        encoder_name = self.encoder
        if self.encoder == "fixed_vector":
            encoder_name = f"fixed_vector_{self.fixed_vector_source}"

        if self.pooling == "explicit_pairwise":
            pool_str = "pairwise"
        elif self.pooling == "global_gated_attention":
            pool_str = "global_gated"
        else:
            pool_str = self.pooling

        if self.encoder == "fixed_vector":
            combo = f"{encoder_name}_{self.fusion}_{self.loss}"
        else:
            combo = f"{encoder_name}_{self.fusion}_{pool_str}_{self.loss}"
        
        prefix = "random_split/" if self.split_type == "random" else ""
        self.checkpoint_dir = f"checkpoints/{prefix}{combo}"
        self.metrics_dir = f"metrics/{prefix}{combo}"

        # PGB runs get their own prefix so they don't overwrite the baseline runs
        if getattr(self, "use_pgb_head", False):
            self.checkpoint_dir = self.checkpoint_dir.replace("checkpoints/", "checkpoints/pgb_")
            self.metrics_dir = self.metrics_dir.replace("metrics/", "metrics/pgb_")

    def resolve_paths(self):
        """Back-compat wrapper: old callers that expect resolve_paths() to do both."""
        self.resolve_checkpoint_paths()
        self.resolve_csv_paths()
