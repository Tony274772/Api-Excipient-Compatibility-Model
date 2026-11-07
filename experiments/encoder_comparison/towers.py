"""Molecule tower for the comparison: exactly one molecular representation per family.

Identical to src.model.MoleculeTower (same block sizes, same order, same activations) except:

  * the secondary fingerprint branch is never built — no family gets a second representation,
  * the primary fingerprint branch is optional, so a sequence / graph family can run on its
    learned embedding alone (primary_dim = 0),
  * no missing-primary embedding / availability flag: with one representation per family the
    flag is 1 for every molecule in this dataset, so it would only add a constant input.

Everything that is not a molecular representation is unchanged: 12 descriptors -> 24,
18 chemistry flags -> 16, 8 salt-context values passed through, mix -> 96.
"""

import torch
import torch.nn as nn


class ComparisonTower(nn.Module):
    """One molecule -> 96-d vector h, from exactly one representation + the shared features."""

    def __init__(self, struct_dim: int, primary_dim: int, pgb_dim: int = 96,
                 dropout: float = 0.15, use_salt_context: bool = True):
        super().__init__()
        if struct_dim == 0 and primary_dim == 0:
            raise ValueError("a tower needs one representation: struct_dim or primary_dim must be > 0")
        if struct_dim > 0 and primary_dim > 0:
            raise ValueError("a tower may have only one representation in this comparison "
                             f"(got struct_dim={struct_dim} and primary_dim={primary_dim})")

        self.has_struct = struct_dim > 0
        self.has_primary = primary_dim > 0
        self.use_salt_context = use_salt_context

        struct_out = struct_dim if self.has_struct else 0

        if self.has_primary:
            self.primary_block = nn.Sequential(
                nn.Linear(primary_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            primary_out = 64
        else:
            primary_out = 0

        self.desc_block = nn.Sequential(
            nn.Linear(12, 24),
            nn.LayerNorm(24),
            nn.GELU(),
        )
        self.flag_block = nn.Sequential(
            nn.Linear(18, 16),
            nn.GELU(),
        )

        mix_in = struct_out + primary_out + 24 + 16 + (8 if use_salt_context else 0)
        self.mix = nn.Sequential(
            nn.Linear(mix_in, pgb_dim),
            nn.LayerNorm(pgb_dim),
            nn.GELU(),
        )

    def forward(self, struct_vec, primary, primary_avail, secondary, pgb_desc, salt, flags):
        """Signature matches src.model.MoleculeTower so PGBCompatibilityModel can call it
        unchanged. `primary_avail` and `secondary` are accepted and ignored."""
        parts = []
        if self.has_struct:
            parts.append(struct_vec)
        if self.has_primary:
            parts.append(self.primary_block(primary))
        parts.append(self.desc_block(pgb_desc))
        parts.append(self.flag_block(flags))
        if self.use_salt_context and salt is not None:
            parts.append(salt)
        return self.mix(torch.cat(parts, dim=-1))
