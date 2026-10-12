"""Configuration for PGB models using global_gated_attention pooling.

Specifically provides ChemBERTa and DMPNN with global_gated_attention pooling
(changing from the default 'cls' pooling) while keeping everything completely
separate from existing checkpoints and metrics.
"""

from dataclasses import dataclass
from experiments.pgb.config import PGBConfig

GATED_FAMILIES = {
    "chemberta": {
        "encoder": "chemberta",
        "fusion": "cross_attn",
        "pooling": "global_gated_attention",
        "pgb_max_epochs": 60,
    },
    "dmpnn": {
        "encoder": "dmpnn_chemprop",
        "fusion": "cross_attn",
        "pooling": "global_gated_attention",
        "pgb_max_epochs": 60,
    },
}


@dataclass
class PGBGlobalGatedConfig(PGBConfig):
    """Configuration class for ChemBERTa & DMPNN with global_gated_attention pooling."""

    use_pgb_head: bool = True
    split_type: str = "cluster"
    family: str = "chemberta"
    data_dir: str = "data"
    pgb_metrics_root: str = "experiments/results/metrics_pgb/global_gated"
    pgb_checkpoint_root: str = "checkpoints/pgb/global_gated"

    def apply_family_spec(self, family_name: str):
        if family_name not in GATED_FAMILIES:
            raise ValueError(f"Unknown family '{family_name}'. Choices: {list(GATED_FAMILIES.keys())}")
        self.family = family_name
        spec = GATED_FAMILIES[family_name]
        for k, v in spec.items():
            setattr(self, k, v)
        
        # Ensure exact same group-split data is loaded
        self.data_dir = "data"
        self.split_type = "cluster"
        self.train_csv = "data/train.csv"
        self.val_csv = "data/val.csv"
        self.test_csv = "data/test.csv"
        self.heldout_csv = "held_out_testset/held_out_test_set.csv"

        self.resolve_paths()
        self.checkpoint_dir = f"{self.pgb_checkpoint_root}/{family_name}"
        self.metrics_dir = f"{self.pgb_metrics_root}/{family_name}"
