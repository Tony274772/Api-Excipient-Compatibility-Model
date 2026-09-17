"""Loss functions – Section 8.

All operate on raw logits (apply sigmoid internally).
All accept an optional sample_weight tensor.
"""

import torch
import torch.nn.functional as F


def bce_loss(logits: torch.Tensor, targets: torch.Tensor,
             sample_weight: torch.Tensor = None, **kwargs) -> torch.Tensor:
    """Plain binary cross-entropy with logits, no class weighting."""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


def weighted_bce_loss(logits: torch.Tensor, targets: torch.Tensor,
                      sample_weight: torch.Tensor = None,
                      pos_weight: torch.Tensor = None, **kwargs) -> torch.Tensor:
    """BCE with pos_weight = num_negatives / num_positives."""
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               sample_weight: torch.Tensor = None,
               gamma: float = 2.0, alpha: float = 0.25, **kwargs) -> torch.Tensor:
    """Standard binary focal loss: L = -alpha * (1-p_t)^gamma * log(p_t)."""
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)

    # alpha weighting: alpha for positive, (1-alpha) for negative
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    focal_weight = alpha_t * (1 - p_t) ** gamma
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = focal_weight * bce

    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


def asl_loss(logits: torch.Tensor, targets: torch.Tensor,
             sample_weight: torch.Tensor = None,
             gamma_neg: float = 4.0, gamma_pos: float = 1.0,
             clip: float = 0.05, eps: float = 1e-8, **kwargs) -> torch.Tensor:
    """Asymmetric Loss (Ben-Baruch et al. 2020).
    
    loss_pos = -y * (1-p)^gamma_pos * log_sigmoid(logits)
    p_shift  = clamp(p - clip, min=0)
    loss_neg = -(1-y) * p_shift^gamma_neg * log(1 - p_shift + eps)
    loss     = mean(loss_pos + loss_neg)
    """
    p = torch.sigmoid(logits)

    # Positive part
    loss_pos = targets * (1 - p) ** gamma_pos * F.logsigmoid(logits)

    # Negative part with probability shifting
    p_shift = torch.clamp(p - clip, min=0)
    loss_neg = (1 - targets) * p_shift ** gamma_neg * torch.log(1 - p_shift + eps)

    loss = -(loss_pos + loss_neg)

    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


def get_loss_fn(config):
    """Return a loss function callable based on config.loss."""
    if config.loss == "bce":
        return lambda logits, targets, sw=None: bce_loss(logits, targets, sw)

    elif config.loss == "weighted_bce":
        # pos_weight is computed at training time from train labels
        return lambda logits, targets, sw=None, pw=None: weighted_bce_loss(
            logits, targets, sw, pos_weight=pw
        )

    elif config.loss == "focal":
        return lambda logits, targets, sw=None: focal_loss(
            logits, targets, sw,
            gamma=config.focal_gamma, alpha=config.focal_alpha,
        )

    elif config.loss == "asl":
        return lambda logits, targets, sw=None: asl_loss(
            logits, targets, sw,
            gamma_neg=config.asl_gamma_neg,
            gamma_pos=config.asl_gamma_pos,
            clip=config.asl_clip,
        )

    else:
        raise ValueError(f"Unknown loss: {config.loss}")
