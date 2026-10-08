"""Training loop for PGB architecture."""

from src.train import (
    pgb_train,
    _pgb_val_pr_auc,
)

__all__ = [
    "pgb_train",
    "_pgb_val_pr_auc",
]
