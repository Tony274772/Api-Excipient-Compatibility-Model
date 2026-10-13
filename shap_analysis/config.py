"""
Configuration module for PGB Morgan SHAP Analysis.
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class SHAPConfig:
    """Configuration parameters for running SHAP analysis on PGB Morgan model."""

    # Model & Architecture
    family: str = "morgan"
    checkpoint_path: str = "checkpoints/pgb/group_split/morgan/best_model.pt"
    norm_stats_path: str = "checkpoints/pgb/group_split/morgan/pgb_norm_stats.npz"
    prior_table_path: str = "checkpoints/pgb/group_split/morgan/prior_table.json"

    # Datasets
    train_csv: str = "data/train.csv"
    val_csv: str = "data/val.csv"
    test_csv: str = "data/test.csv"
    held_out_csv: str = "held_out_testset/held_out_test_set.csv"

    # SHAP Computation Parameters
    explainer_type: Literal["gradient", "kernel"] = "gradient"
    background_samples: int = 100
    test_samples: Optional[int] = None  # None or 0 to use all samples in the dataset
    random_seed: int = 42
    batch_size: int = 32
    output_mode: Literal["logit", "probability"] = "logit"

    # Output paths
    results_dir: str = "shap_analysis/results"
    plots_dir: str = "shap_analysis/results/plots"
    npz_output_path: str = "shap_analysis/results/shap_values.npz"
    csv_importance_path: str = "shap_analysis/results/feature_importance.csv"
    summary_report_path: str = "shap_analysis/results/summary_report.md"

    # Hardware
    device: str = field(
        default_factory=lambda: "cuda" if __import__("torch").cuda.is_available() else "cpu"
    )

    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
