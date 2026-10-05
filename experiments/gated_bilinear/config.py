"""Configuration for the Gated Bilinear (PGB) experiment.

Extends the base Config with PGB-specific settings while keeping
all existing fields intact so encoders and data loading still work.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from src.config import Config


@dataclass
class GBConfig(Config):
    """All settings for a Prior-Gated Bilinear run.

    Inherits every field from the base Config so that encoder construction,
    data-path resolution, and checkpoint naming continue to work unchanged.
    PGB-specific fields are added below.
    """

    # ── Family selection ──────────────────────────────────────────────────
    gb_family: Literal[
        "maccs", "morgan", "pubchem", "mol2vec",
        "molformer", "chemberta",
        "gin", "gat", "dmpnn",
    ] = "maccs"

    # ── Tower dimensions (shared across families) ────────────────────────
    tower_output_dim: int = 96         # h_API / h_EXC width
    bilinear_rank: int = 16            # rank of the bilinear interaction

    # ── PGB head ─────────────────────────────────────────────────────────
    pgb_mlp_dropout_1: float = 0.30    # after 426 → 128
    pgb_mlp_dropout_2: float = 0.20    # after 128 → 48
    pgb_mlp_hidden_1: int = 128
    pgb_mlp_hidden_2: int = 48
    prior_scale_init: float = 0.8      # s starts at 0.8
    base_rate: float = 0.094           # global incompatibility rate

    # ── Gate ─────────────────────────────────────────────────────────────
    gate_hidden_dim: int = 16          # gate intermediate width

    # ── Descriptors (PGB uses 12, not 21) ────────────────────────────────
    gb_num_descriptors: int = 12
    gb_desc_proj_dim: int = 24

    # ── Chemistry flags ──────────────────────────────────────────────────
    num_chem_flags: int = 18
    chem_flag_proj_dim: int = 16

    # ── Mechanism flags ──────────────────────────────────────────────────
    num_mechanism_flags: int = 5

    # ── Prior block ──────────────────────────────────────────────────────
    prior_block_dim: int = 5
    prior_laplace_alpha: int = 4
    prior_tanimoto_threshold: float = 0.6

    # ── Loss (asymmetric focal with doc parameters) ──────────────────────
    gb_loss: str = "asymmetric_focal"
    gb_afl_gamma_pos: float = 1.2
    gb_afl_gamma_neg: float = 2.5

    # ── Training recipe ──────────────────────────────────────────────────
    # Fixed-vector families
    gb_lr_fixed: float = 1.2e-3
    # Transformer / graph families: new layers
    gb_lr_new: float = 1.0e-3
    # Transformer / graph families: kept layers
    gb_lr_kept: float = 3.0e-4

    gb_weight_decay: float = 2.0e-3
    gb_grad_clip: float = 1.0
    gb_batch_size: int = 64

    gb_max_epochs_fixed: int = 40
    gb_max_epochs_deep: int = 60

    gb_lr_patience: int = 4       # halve after 4 epochs without gain
    gb_lr_factor: float = 0.5
    gb_early_stop_patience: int = 7
    gb_early_stop_min_delta: float = 0.002

    # ── Threshold ────────────────────────────────────────────────────────
    gb_tnr_floor: float = 0.97         # true-negative-rate floor for MCC threshold
    gb_screen_min_recall: float = 0.75 # screen cut: val recall ≥ 0.75

    # ── Output paths (override at runtime) ───────────────────────────────
    gb_checkpoint_dir: str = "checkpoints/gated_bilinear"
    gb_metrics_dir: str = "experiments/results/metrics_bilinear"
    gb_heldout_csv: str = "held_out_testset/held_out_test_set.csv"
    gb_heldout_predictions_csv: str = "held_out_testset/held_out_predictions_gatedbilinear.csv"

    def resolve_gb_paths(self):
        """Set checkpoint and metrics dirs based on family name."""
        self.gb_checkpoint_dir = f"checkpoints/gated_bilinear/{self.gb_family}"
        self.gb_metrics_dir = f"experiments/results/metrics_bilinear/{self.gb_family}"

    def __post_init__(self):
        super().__post_init__()
        # Disable balanced sampler for PGB runs
        self.use_balanced_sampler = False
