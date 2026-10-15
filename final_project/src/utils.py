"""Utility functions: seeding, device helpers."""

import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to a torch.device.
    
    'auto' -> CUDA if available, else CPU.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
