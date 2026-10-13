"""
SHAP (SHapley Additive exPlanations) Analysis Module for PGB Morgan Compatibility Model.

This package provides tools for model interpretability, feature importance attribution,
and chemical insight discovery on the Prior-Gated Bilinear (PGB) architecture.
"""

from shap_analysis.config import SHAPConfig
from shap_analysis.model_wrapper import FlatPGBModelWrapper, TowerLevelPGBModelWrapper
from shap_analysis.feature_names import (
    FEATURE_NAMES,
    FEATURE_GROUPS,
    MACRO_GROUPS,
    get_feature_name,
    get_feature_group,
)

__all__ = [
    "SHAPConfig",
    "FlatPGBModelWrapper",
    "TowerLevelPGBModelWrapper",
    "FEATURE_NAMES",
    "FEATURE_GROUPS",
    "MACRO_GROUPS",
    "get_feature_name",
    "get_feature_group",
]
