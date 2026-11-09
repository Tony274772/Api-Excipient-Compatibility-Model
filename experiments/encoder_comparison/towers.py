"""Molecule tower for the grid: one optional primary + one optional secondary description.

Identical to src.model.MoleculeTower (same block sizes, same order, same activations) except that
BOTH representation branches are optional, so the grid can build:

    struct only          a deep method with cross-attention, no secondary        (the `none` column)
    primary only         a fingerprint method, no secondary                      (the `none` column)
    struct + secondary   a deep method plus a second description
    primary + secondary  a fingerprint method plus a second description
    neither              the FLOOR: no molecular description at all

Everything that is not a molecular description is unchanged and present in every cell:
12 descriptors -> 24, 18 chemistry flags -> 16, 8 salt-context values passed through, mix -> 96.

Widths never change with the input size: primary is always 64 wide, secondary always 32, the tower
output always 96. Only the INPUT width differs, because it is a property of the fingerprint or the
released checkpoint. See Appendix A of paper/encoder_grid_plan.md.

There is no missing-primary embedding / availability flag: every molecule in this dataset has every
representation (audit_inputs.py enforces that), so the flag would only add a constant input.
"""

import torch
import torch.nn as nn


class ComparisonTower(nn.Module):
    """One molecule -> 96-d vector h, from its description(s) + the shared features."""

    PRIMARY_OUT = 64
    SECONDARY_OUT = 32

    def __init__(self, struct_dim: int, primary_dim: int, secondary_dim: int = 0,
                 pgb_dim: int = 96, dropout: float = 0.15, use_salt_context: bool = True,
                 allow_no_representation: bool = False):
        super().__init__()
        if struct_dim > 0 and primary_dim > 0:
            raise ValueError(
                "a tower may have at most one PRIMARY representation: either a structural vector "
                f"from a cross-attended encoder or a fixed primary vector, not both "
                f"(got struct_dim={struct_dim} and primary_dim={primary_dim})")
        if struct_dim == 0 and primary_dim == 0 and not allow_no_representation:
            raise ValueError(
                "this tower has no molecular description. That is only valid for the FLOOR cell; "
                "pass allow_no_representation=True if you meant it.")
        if secondary_dim > 0 and struct_dim == 0 and primary_dim == 0:
            raise ValueError("a secondary description without a primary one is not a grid cell")

        self.has_struct = struct_dim > 0
        self.has_primary = primary_dim > 0
        self.has_secondary = secondary_dim > 0
        self.use_salt_context = use_salt_context

        struct_out = struct_dim if self.has_struct else 0

        if self.has_primary:
            self.primary_block = nn.Sequential(
                nn.Linear(primary_dim, self.PRIMARY_OUT),
                nn.LayerNorm(self.PRIMARY_OUT),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            primary_out = self.PRIMARY_OUT
        else:
            primary_out = 0

        # Secondary branch: same shape as src.model.MoleculeTower, including the halved dropout.
        if self.has_secondary:
            self.secondary_block = nn.Sequential(
                nn.Linear(secondary_dim, self.SECONDARY_OUT),
                nn.LayerNorm(self.SECONDARY_OUT),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
            secondary_out = self.SECONDARY_OUT
        else:
            secondary_out = 0

        self.desc_block = nn.Sequential(
            nn.Linear(12, 24),
            nn.LayerNorm(24),
            nn.GELU(),
        )
        self.flag_block = nn.Sequential(
            nn.Linear(18, 16),
            nn.GELU(),
        )

        mix_in = struct_out + primary_out + secondary_out + 24 + 16 + (8 if use_salt_context else 0)
        self.mix_in = mix_in
        self.mix = nn.Sequential(
            nn.Linear(mix_in, pgb_dim),
            nn.LayerNorm(pgb_dim),
            nn.GELU(),
        )

    def forward(self, struct_vec, primary, primary_avail, secondary, pgb_desc, salt, flags):
        """Signature matches src.model.MoleculeTower so PGBCompatibilityModel can call it
        unchanged. `primary_avail` is accepted and ignored (see the module docstring)."""
        parts = []
        if self.has_struct:
            parts.append(struct_vec)
        if self.has_primary:
            parts.append(self.primary_block(primary))
        if self.has_secondary:
            if secondary is None:
                raise RuntimeError(
                    "this tower was built with a secondary block but the batch carries no "
                    "`api_secondary`/`exc_secondary`. The dataset and the tower disagree about "
                    "the cell being run.")
            parts.append(self.secondary_block(secondary))
        parts.append(self.desc_block(pgb_desc))
        parts.append(self.flag_block(flags))
        if self.use_salt_context:
            parts.append(salt)
        return self.mix(torch.cat(parts, dim=-1))
