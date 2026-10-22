"""
PyTorch Model Wrappers for SHAP Explanation on PGB Compatibility Models.
Provides standard differentiable interfaces for gradient-based and kernel-based SHAP explainers.
"""

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn

from shap_analysis.feature_names import FEATURE_GROUPS


class FlatPGBModelWrapper(nn.Module):
    """
    Wraps PGBCompatibilityModel to accept a flat 2D feature tensor [B, 2468]
    and output scalar logits or predicted probabilities [B, 1].
    """

    def __init__(
        self,
        model: nn.Module,
        output_mode: Literal["logit", "probability"] = "logit",
    ):
        super().__init__()
        self.model = model
        self.output_mode = output_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass unpacking flat feature tensor.
        Args:
            x: Tensor of shape [B, 2468] or [2468]
        Returns:
            Tensor of shape [B, 1] containing logit or probability
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        device = x.device
        B = x.shape[0]

        # Extract feature groups by predefined index ranges
        api_p_start, api_p_end = FEATURE_GROUPS["api_primary"]
        exc_p_start, exc_p_end = FEATURE_GROUPS["exc_primary"]
        api_s_start, api_s_end = FEATURE_GROUPS["api_secondary"]
        exc_s_start, exc_s_end = FEATURE_GROUPS["exc_secondary"]
        api_d_start, api_d_end = FEATURE_GROUPS["api_pgb_desc"]
        exc_d_start, exc_d_end = FEATURE_GROUPS["exc_pgb_desc"]
        api_salt_start, api_salt_end = FEATURE_GROUPS["api_salt"]
        exc_salt_start, exc_salt_end = FEATURE_GROUPS["exc_salt"]
        api_f_start, api_f_end = FEATURE_GROUPS["api_flags"]
        exc_f_start, exc_f_end = FEATURE_GROUPS["exc_flags"]
        prior_start, prior_end = FEATURE_GROUPS["prior_vec"]
        mech_start, mech_end = FEATURE_GROUPS["mech_flags"]

        api_primary = x[:, api_p_start:api_p_end]
        exc_primary = x[:, exc_p_start:exc_p_end]
        api_secondary = x[:, api_s_start:api_s_end]
        exc_secondary = x[:, exc_s_start:exc_s_end]
        api_pgb_desc = x[:, api_d_start:api_d_end]
        exc_pgb_desc = x[:, exc_d_start:exc_d_end]
        api_salt = x[:, api_salt_start:api_salt_end]
        exc_salt = x[:, exc_salt_start:exc_salt_end]
        api_flags = x[:, api_f_start:api_f_end]
        exc_flags = x[:, exc_f_start:exc_f_end]
        prior_vec = x[:, prior_start:prior_end]
        mech_flags = x[:, mech_start:mech_end]

        # Availability indicator: 1.0 for valid input
        avail = torch.ones(B, device=device)

        # Molecule Tower feature transformations (96-d each)
        h_api = self.model.api_tower(
            struct_vec=None,
            primary=api_primary,
            primary_avail=avail,
            secondary=api_secondary,
            pgb_desc=api_pgb_desc,
            salt=api_salt,
            flags=api_flags,
        )

        h_exc = self.model.exc_tower(
            struct_vec=None,
            primary=exc_primary,
            primary_avail=avail,
            secondary=exc_secondary,
            pgb_desc=exc_pgb_desc,
            salt=exc_salt,
            flags=exc_flags,
        )

        # Head execution
        logit = self.model.head(
            h_api=h_api,
            h_exc=h_exc,
            prior_vec=prior_vec,
            api_flags=api_flags,
            exc_flags=exc_flags,
            mech_flags=mech_flags,
        )

        if self.output_mode == "probability":
            output = torch.sigmoid(logit)
        else:
            output = logit

        return output.unsqueeze(-1)  # [B, 1]


class TowerLevelPGBModelWrapper(nn.Module):
    """
    Wraps PGBHead to accept intermediate tower embeddings and pairwise flags:
    Total input dimension = 96 (h_api) + 96 (h_exc) + 5 (prior) + 18 (api_flags) + 18 (exc_flags) + 5 (mech) = 238-d.
    Used for Analysis #9: API Tower vs. Excipient Tower attribution.
    """

    def __init__(
        self,
        model: nn.Module,
        output_mode: Literal["logit", "probability"] = "logit",
    ):
        super().__init__()
        self.model = model
        self.output_mode = output_mode

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Tensor of shape [B, 238]
        Returns:
            Tensor of shape [B, 1]
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        h_api = z[:, 0:96]
        h_exc = z[:, 96:192]
        prior_vec = z[:, 192:197]
        api_flags = z[:, 197:215]
        exc_flags = z[:, 215:233]
        mech_flags = z[:, 233:238]

        logit = self.model.head(
            h_api=h_api,
            h_exc=h_exc,
            prior_vec=prior_vec,
            api_flags=api_flags,
            exc_flags=exc_flags,
            mech_flags=mech_flags,
        )

        if self.output_mode == "probability":
            output = torch.sigmoid(logit)
        else:
            output = logit

        return output.unsqueeze(-1)
