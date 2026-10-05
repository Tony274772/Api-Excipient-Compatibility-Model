"""Per-family tower modules for the PGB architecture.

Each tower takes raw per-molecule features and produces h [96].
API tower and EXC tower share the same architecture but have separate weights.

Tower variants (from instructions Sections 4–6):
- FixedVectorTower: MACCS, Morgan, PubChem, Mol2vec families
- TransformerTowerMix: MoLFormer, ChemBERTa (struct 128 + side branches)
- GNNTowerMix: GIN, GAT, D-MPNN (struct 128 + side branches)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# Reusable sub-blocks
# ═══════════════════════════════════════════════════════════════════════════════

class FingerprintBlock(nn.Module):
    """Linear → LN → GELU → Dropout (optional)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        layers = [
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DescriptorBlock(nn.Module):
    """Linear → LN → GELU (12 → 24)."""

    def __init__(self, in_dim: int = 12, out_dim: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChemFlagBlock(nn.Module):
    """Linear → GELU (18 → 16). No LN per the instructions."""

    def __init__(self, in_dim: int = 18, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixBlock(nn.Module):
    """Linear → LN → GELU (concat_dim → 96)."""

    def __init__(self, in_dim: int, out_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixed-vector towers (Sections 4.1–4.4)
# ═══════════════════════════════════════════════════════════════════════════════

class MACCSTower(nn.Module):
    """MACCS family tower (Section 4.1).

    MACCS 166→64, Morgan 512→64, Desc 12→24, Flags 18→16, Mix 168→96.
    """

    def __init__(self):
        super().__init__()
        self.maccs_block = FingerprintBlock(166, 64, dropout=0.15)
        self.morgan_block = FingerprintBlock(512, 64, dropout=0.15)
        self.desc_block = DescriptorBlock(12, 24)
        self.flag_block = ChemFlagBlock(18, 16)
        self.mix = MixBlock(64 + 64 + 24 + 16, 96)  # 168 → 96

    def forward(
        self,
        maccs: torch.Tensor,      # [B, 166]
        morgan: torch.Tensor,     # [B, 512]
        descriptors: torch.Tensor, # [B, 12]
        chem_flags: torch.Tensor,  # [B, 18]
    ) -> torch.Tensor:
        h_maccs = self.maccs_block(maccs)
        h_morgan = self.morgan_block(morgan)
        h_desc = self.desc_block(descriptors)
        h_flags = self.flag_block(chem_flags)
        concat = torch.cat([h_maccs, h_morgan, h_desc, h_flags], dim=1)
        return self.mix(concat)  # [B, 96]


class MorganTower(nn.Module):
    """Morgan family tower (Section 4.2).

    Morgan 1024→64 (Drop .25), MACCS 166→64, Desc 12→24, Flags 18→16, Mix 168→96.
    """

    def __init__(self):
        super().__init__()
        self.morgan_block = FingerprintBlock(1024, 64, dropout=0.25)
        self.maccs_block = FingerprintBlock(166, 64, dropout=0.15)
        self.desc_block = DescriptorBlock(12, 24)
        self.flag_block = ChemFlagBlock(18, 16)
        self.mix = MixBlock(64 + 64 + 24 + 16, 96)  # 168 → 96

    def forward(
        self,
        morgan: torch.Tensor,     # [B, 1024]
        maccs: torch.Tensor,      # [B, 166]
        descriptors: torch.Tensor,
        chem_flags: torch.Tensor,
    ) -> torch.Tensor:
        h_morgan = self.morgan_block(morgan)
        h_maccs = self.maccs_block(maccs)
        h_desc = self.desc_block(descriptors)
        h_flags = self.flag_block(chem_flags)
        concat = torch.cat([h_morgan, h_maccs, h_desc, h_flags], dim=1)
        return self.mix(concat)


class PubChemTower(nn.Module):
    """PubChem family tower (Section 4.3).

    PubChem 881→64 (Drop .25), Morgan 512→64, Desc 12→24, Flags 18→16, Mix 168→96.
    """

    def __init__(self):
        super().__init__()
        self.pubchem_block = FingerprintBlock(881, 64, dropout=0.25)
        self.morgan_block = FingerprintBlock(512, 64, dropout=0.15)
        self.desc_block = DescriptorBlock(12, 24)
        self.flag_block = ChemFlagBlock(18, 16)
        self.mix = MixBlock(64 + 64 + 24 + 16, 96)  # 168 → 96

    def forward(
        self,
        pubchem: torch.Tensor,    # [B, 881]
        morgan: torch.Tensor,     # [B, 512]
        descriptors: torch.Tensor,
        chem_flags: torch.Tensor,
    ) -> torch.Tensor:
        h_pubchem = self.pubchem_block(pubchem)
        h_morgan = self.morgan_block(morgan)
        h_desc = self.desc_block(descriptors)
        h_flags = self.flag_block(chem_flags)
        concat = torch.cat([h_pubchem, h_morgan, h_desc, h_flags], dim=1)
        return self.mix(concat)


class Mol2vecTower(nn.Module):
    """Mol2vec family tower (Section 4.4).

    Mol2vec 300→64, MACCS 166→32, Morgan 512→32, Desc 12→24, Flags 18→16,
    Mix 168→96.
    """

    def __init__(self):
        super().__init__()
        self.mol2vec_block = FingerprintBlock(300, 64, dropout=0.15)
        self.maccs_block = FingerprintBlock(166, 32, dropout=0.0)  # LN|GELU only
        self.morgan_block = FingerprintBlock(512, 32, dropout=0.0)  # LN|GELU only
        self.desc_block = DescriptorBlock(12, 24)
        self.flag_block = ChemFlagBlock(18, 16)
        self.mix = MixBlock(64 + 32 + 32 + 24 + 16, 96)  # 168 → 96

    def forward(
        self,
        mol2vec: torch.Tensor,    # [B, 300]
        maccs: torch.Tensor,      # [B, 166]
        morgan: torch.Tensor,     # [B, 512]
        descriptors: torch.Tensor,
        chem_flags: torch.Tensor,
    ) -> torch.Tensor:
        h_mol2vec = self.mol2vec_block(mol2vec)
        h_maccs = self.maccs_block(maccs)
        h_morgan = self.morgan_block(morgan)
        h_desc = self.desc_block(descriptors)
        h_flags = self.flag_block(chem_flags)
        concat = torch.cat([h_mol2vec, h_maccs, h_morgan, h_desc, h_flags], dim=1)
        return self.mix(concat)


# ═══════════════════════════════════════════════════════════════════════════════
# Transformer / GNN tower mix (Sections 5 & 6)
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerTowerMix(nn.Module):
    """Tower mix for MoLFormer and ChemBERTa (Sections 5.1, 5.2).

    struct 128 (from existing fusion), side branch, Desc 12→24, Flags 18→16,
    Mix 200→96.

    For MoLFormer: side branch = MACCS 166→32.
    For ChemBERTa: side branch = Morgan 512→32.
    """

    def __init__(self, side_branch_dim: int = 166, side_branch_out: int = 32):
        super().__init__()
        self.side_block = FingerprintBlock(side_branch_dim, side_branch_out, dropout=0.0)
        self.desc_block = DescriptorBlock(12, 24)
        self.flag_block = ChemFlagBlock(18, 16)
        self.mix = MixBlock(128 + side_branch_out + 24 + 16, 96)  # 200 → 96

    def forward(
        self,
        struct: torch.Tensor,       # [B, 128] from existing fusion
        side_features: torch.Tensor, # [B, 166] or [B, 512]
        descriptors: torch.Tensor,   # [B, 12]
        chem_flags: torch.Tensor,    # [B, 18]
    ) -> torch.Tensor:
        h_side = self.side_block(side_features)
        h_desc = self.desc_block(descriptors)
        h_flags = self.flag_block(chem_flags)
        concat = torch.cat([struct, h_side, h_desc, h_flags], dim=1)
        return self.mix(concat)


class GNNTowerMix(nn.Module):
    """Tower mix for GIN, GAT, D-MPNN (Sections 6.1, 6.2, 6.3).

    struct 128 (from existing fusion/pooling), MACCS 166→32, Desc 12→24,
    Flags 18→16, Mix 200→96.
    """

    def __init__(self):
        super().__init__()
        self.maccs_block = FingerprintBlock(166, 32, dropout=0.0)
        self.desc_block = DescriptorBlock(12, 24)
        self.flag_block = ChemFlagBlock(18, 16)
        self.mix = MixBlock(128 + 32 + 24 + 16, 96)  # 200 → 96

    def forward(
        self,
        struct: torch.Tensor,       # [B, 128] from fusion/pooling
        maccs: torch.Tensor,        # [B, 166]
        descriptors: torch.Tensor,  # [B, 12]
        chem_flags: torch.Tensor,   # [B, 18]
    ) -> torch.Tensor:
        h_maccs = self.maccs_block(maccs)
        h_desc = self.desc_block(descriptors)
        h_flags = self.flag_block(chem_flags)
        concat = torch.cat([struct, h_maccs, h_desc, h_flags], dim=1)
        return self.mix(concat)


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_tower(family: str) -> nn.Module:
    """Return a tower module for the given family name.

    The caller creates two instances (API and EXC) with separate weights.
    """
    family = family.lower()

    if family == "maccs":
        return MACCSTower()
    elif family == "morgan":
        return MorganTower()
    elif family == "pubchem":
        return PubChemTower()
    elif family == "mol2vec":
        return Mol2vecTower()
    elif family == "molformer":
        # Side branch: MACCS 166 → 32
        return TransformerTowerMix(side_branch_dim=166, side_branch_out=32)
    elif family == "chemberta":
        # Side branch: Morgan 512 → 32
        return TransformerTowerMix(side_branch_dim=512, side_branch_out=32)
    elif family in ("gin", "gat", "dmpnn"):
        return GNNTowerMix()
    else:
        raise ValueError(f"Unknown PGB family: {family}")
