"""Asymmetric Focal Loss for Prior-Gated Bilinear architecture.

Implements Section 3.5 of the instructions:
- Asymmetric focal loss with specific parameters.
- gamma+ = 1.2, gamma- = 2.5
- pos_weight = n_neg / n_pos (from training set)
- Replaces weighted BCE and standard focal loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Loss for highly imbalanced binary classification.
    
    Formula:
    L = -w_c * [y * (1 - p)^gamma_pos * log(p) + (1 - y) * p^gamma_neg * log(1 - p)]
    
    where w_c is the class weight (pos_weight for y=1, 1 for y=0).
    """

    def __init__(
        self,
        gamma_pos: float = 1.2,
        gamma_neg: float = 2.5,
        pos_weight: float = 1.0,
    ):
        """
        Args:
            gamma_pos: Focusing parameter for positive class (default 1.2).
            gamma_neg: Focusing parameter for negative class (default 2.5).
            pos_weight: Weight for positive class (n_neg / n_pos).
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B] Raw logits from the model.
            targets: [B] Binary targets (0 or 1).
            sample_weights: [B] Optional per-sample weights.
            
        Returns:
            Scalar loss.
        """
        # Ensure targets are float
        targets = targets.float()
        
        # Calculate probabilities
        probs = torch.sigmoid(logits)
        
        # Binary cross entropy term
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Calculate focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        focal_weight = (1 - p_t) ** gamma
        
        # Apply class weights
        class_weight = self.pos_weight * targets + 1.0 * (1 - targets)
        
        # Combine
        loss = class_weight * focal_weight * bce_loss
        
        # Apply sample weights if provided
        if sample_weights is not None:
            loss = loss * sample_weights
            
        # Return mean loss
        return loss.mean()
