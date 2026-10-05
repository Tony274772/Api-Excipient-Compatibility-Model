"""Prior-Gated Bilinear (PGB) head — shared across all 9 families.

Implements Section 3.4 of the instructions document:
- Bilinear rank-16 interaction: (h_API @ U) * (h_EXC @ V)
- Gate: [prior 5, EXC flags 18, mechanism 5] → 28 → 16 (GELU) → 16 (sigmoid)
- Gated bilinear: bilinear × gate, channel-wise
- Product: h_API * h_EXC [96]
- Difference: |h_API - h_EXC| [96]
- Concat [426] = h_API 96 + h_EXC 96 + product 96 + |diff| 96
                 + bilinear 16 + gated_bilinear 16 + prior 5 + mechanism 5
- MLP: 426 → 128 (GELU, Drop 0.30) → 48 (GELU, Drop 0.20) → 1
- Logit offset: residual + s * log(p / (1-p)) + b
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PriorGatedBilinearHead(nn.Module):
    """Shared PGB head, identical in every family.

    Args:
        tower_dim:    h_API / h_EXC width (default 96).
        bilinear_rank: rank of the factored bilinear term (default 16).
        prior_dim:    dimension of the excipient prior block (default 5).
        num_chem_flags: excipient chemistry flags (default 18).
        num_mechanism_flags: pair mechanism flags (default 5).
        mlp_dropout_1: dropout after first MLP layer (default 0.30).
        mlp_dropout_2: dropout after second MLP layer (default 0.20).
        mlp_hidden_1: first hidden dim (default 128).
        mlp_hidden_2: second hidden dim (default 48).
        prior_scale_init: initial value of learned scale s (default 0.8).
        base_rate: global incompatibility rate (default 0.094).
    """

    def __init__(
        self,
        tower_dim: int = 96,
        bilinear_rank: int = 16,
        prior_dim: int = 5,
        num_chem_flags: int = 18,
        num_mechanism_flags: int = 5,
        mlp_dropout_1: float = 0.30,
        mlp_dropout_2: float = 0.20,
        mlp_hidden_1: int = 128,
        mlp_hidden_2: int = 48,
        prior_scale_init: float = 0.8,
        base_rate: float = 0.094,
    ):
        super().__init__()

        self.tower_dim = tower_dim
        self.bilinear_rank = bilinear_rank

        # ── Bilinear rank-16 ─────────────────────────────────────────────
        # U maps h_API: [B, 96] → [B, 16], no bias
        self.bilinear_U = nn.Linear(tower_dim, bilinear_rank, bias=False)
        # V maps h_EXC: [B, 96] → [B, 16], no bias
        self.bilinear_V = nn.Linear(tower_dim, bilinear_rank, bias=False)

        # ── Gate ─────────────────────────────────────────────────────────
        gate_input_dim = prior_dim + num_chem_flags + num_mechanism_flags  # 5+18+5=28
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, bilinear_rank),   # 28 → 16
            nn.GELU(),
            nn.Linear(bilinear_rank, bilinear_rank),    # 16 → 16
            nn.Sigmoid(),
        )

        # ── MLP head ─────────────────────────────────────────────────────
        # Concat: h_API(96) + h_EXC(96) + product(96) + |diff|(96)
        #         + bilinear(16) + gated_bilinear(16) + prior(5) + mechanism(5)
        concat_dim = tower_dim * 4 + bilinear_rank * 2 + prior_dim + num_mechanism_flags
        # = 96*4 + 16*2 + 5 + 5 = 384 + 32 + 10 = 426

        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, mlp_hidden_1),   # 426 → 128
            nn.GELU(),
            nn.Dropout(mlp_dropout_1),
            nn.Linear(mlp_hidden_1, mlp_hidden_2), # 128 → 48
            nn.GELU(),
            nn.Dropout(mlp_dropout_2),
            nn.Linear(mlp_hidden_2, 1),            # 48 → 1
        )

        # ── Prior logit offset ───────────────────────────────────────────
        # logit = residual + s * log(p / (1-p)) + b
        self.prior_scale = nn.Parameter(torch.tensor(prior_scale_init))
        # b starts at 0.2 * log(base / (1 - base))
        b_init = 0.2 * math.log(base_rate / (1.0 - base_rate))
        self.prior_bias = nn.Parameter(torch.tensor(b_init))

    def forward(
        self,
        h_api: torch.Tensor,          # [B, 96]
        h_exc: torch.Tensor,          # [B, 96]
        prior_block: torch.Tensor,    # [B, 5]
        exc_chem_flags: torch.Tensor, # [B, 18]
        mechanism_flags: torch.Tensor, # [B, 5]
        p_exact: torch.Tensor,        # [B] — excipient prior for logit offset
    ) -> torch.Tensor:
        """Forward pass.

        Returns:
            logits: [B] raw logits (apply sigmoid externally).
        """
        # ── Interaction terms ────────────────────────────────────────────
        product = h_api * h_exc                          # [B, 96]
        diff = torch.abs(h_api - h_exc)                  # [B, 96]

        # ── Bilinear rank-16 ─────────────────────────────────────────────
        api_proj = self.bilinear_U(h_api)                # [B, 16]
        exc_proj = self.bilinear_V(h_exc)                # [B, 16]
        bilinear = api_proj * exc_proj                   # [B, 16]

        # ── Gate ─────────────────────────────────────────────────────────
        gate_input = torch.cat([prior_block, exc_chem_flags, mechanism_flags], dim=1)  # [B, 28]
        g = self.gate(gate_input)                        # [B, 16]

        # ── Gated bilinear ───────────────────────────────────────────────
        gated_bilinear = bilinear * g                    # [B, 16]

        # ── Concat [426] ─────────────────────────────────────────────────
        pair_vector = torch.cat([
            h_api,              # 96
            h_exc,              # 96
            product,            # 96
            diff,               # 96
            bilinear,           # 16
            gated_bilinear,     # 16
            prior_block,        # 5
            mechanism_flags,    # 5
        ], dim=1)               # = 426

        # ── MLP residual ─────────────────────────────────────────────────
        residual = self.mlp(pair_vector).squeeze(1)      # [B]

        # ── Prior logit offset ───────────────────────────────────────────
        # Clamp p_exact to avoid log(0) or log(inf)
        p_clamped = torch.clamp(p_exact, min=1e-6, max=1.0 - 1e-6)
        log_odds = torch.log(p_clamped / (1.0 - p_clamped))

        logits = residual + self.prior_scale * log_odds + self.prior_bias  # [B]

        return logits
