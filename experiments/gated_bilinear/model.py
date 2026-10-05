"""Full Prior-Gated Bilinear model for all 9 families.

Wires encoder → projection → (cross-attn/pooling if applicable) → tower → PGB head.
Reuses existing encoder and fusion components from src/ without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model import (
    ProjectionHead,
    GatedAttentionPooling,
)
from experiments.gated_bilinear.towers import build_tower
from experiments.gated_bilinear.head import PriorGatedBilinearHead


# Families that use fixed-vector encoders (no sequence, no cross-attention)
FIXED_VECTOR_FAMILIES = {"maccs", "morgan", "pubchem", "mol2vec"}

# Families that use transformer encoders
TRANSFORMER_FAMILIES = {"molformer", "chemberta"}

# Families that use GNN encoders
GNN_FAMILIES = {"gin", "gat", "dmpnn"}


class GatedBilinearModel(nn.Module):
    """Prior-Gated Bilinear model.

    Args:
        config: GBConfig instance.
        encoder: The existing encoder module (from src.encoders).
        family: One of the 9 family names.
    """

    def __init__(self, config, encoder, family: str):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.family = family.lower()

        enc_dim = encoder.output_dim
        proj_dim = config.proj_dim  # 128
        tower_out = config.tower_output_dim  # 96

        # ── Encoder fusion (reuse existing architecture where KEEP) ──────
        if self.family in FIXED_VECTOR_FAMILIES:
            # Fixed-vector families: no cross-attention, no sequence
            # Projection is handled inside the tower
            self._has_fusion = False
        else:
            # Transformer / GNN families: keep existing projection + cross-attn + pooling
            self._has_fusion = True
            self._build_fusion(config, enc_dim, proj_dim)

        # ── Towers (separate weights for API and EXC) ────────────────────
        self.api_tower = build_tower(self.family)
        self.exc_tower = build_tower(self.family)

        # ── PGB head (identical in every family) ─────────────────────────
        self.head = PriorGatedBilinearHead(
            tower_dim=tower_out,
            bilinear_rank=config.bilinear_rank,
            prior_dim=config.prior_block_dim,
            num_chem_flags=config.num_chem_flags,
            num_mechanism_flags=config.num_mechanism_flags,
            mlp_dropout_1=config.pgb_mlp_dropout_1,
            mlp_dropout_2=config.pgb_mlp_dropout_2,
            mlp_hidden_1=config.pgb_mlp_hidden_1,
            mlp_hidden_2=config.pgb_mlp_hidden_2,
            prior_scale_init=config.prior_scale_init,
            base_rate=config.base_rate,
        )

    def _build_fusion(self, config, enc_dim: int, proj_dim: int):
        """Build projection, cross-attention, and pooling layers (KEPT layers).

        This replicates the existing pipeline's fusion architecture.
        """
        # ── Projection heads (separate weights per side) ─────────────────
        self.api_proj = ProjectionHead(enc_dim, proj_dim, config.proj_dropout)
        self.exc_proj = ProjectionHead(enc_dim, proj_dim, config.proj_dropout)

        # ── Determine pooling mode ───────────────────────────────────────
        # GIN uses late fusion (no cross-attention per Section 6.1)
        self._use_cross_attn = (self.family != "gin")

        if self._use_cross_attn:
            # Bidirectional cross-attention
            self.cross_attn_exc = nn.MultiheadAttention(
                embed_dim=proj_dim,
                num_heads=config.num_heads,
                dropout=config.attn_dropout,
                batch_first=True,
            )
            self.cross_attn_api = nn.MultiheadAttention(
                embed_dim=proj_dim,
                num_heads=config.num_heads,
                dropout=config.attn_dropout,
                batch_first=True,
            )
            self.ln_exc = nn.LayerNorm(proj_dim)
            self.ln_api = nn.LayerNorm(proj_dim)

            # Learned placeholders for missing excipient
            self.exc_placeholder = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.01)
            self.exc_global_placeholder = nn.Parameter(torch.randn(1, proj_dim) * 0.01)

        # ── Pooling mode per family ──────────────────────────────────────
        # MoLFormer, GAT: global gated attention pooling
        # ChemBERTa, D-MPNN: CLS pooling
        # GIN: gated attention pooling (no cross-attn)
        if self.family in ("molformer", "gat", "gin"):
            self._pooling_mode = "global_gated"
            pool_hidden_dim = getattr(config, "pool_hidden_dim", proj_dim)
            self.api_pool = GatedAttentionPooling(
                input_dim=proj_dim, hidden_dim=pool_hidden_dim
            )
            self.exc_pool = GatedAttentionPooling(
                input_dim=proj_dim, hidden_dim=pool_hidden_dim
            )
            # Fusion: concat global + pooled → Linear → 128
            self.api_pool_fusion = nn.Sequential(
                nn.Linear(proj_dim * 2, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(config.proj_dropout),
            )
            self.exc_pool_fusion = nn.Sequential(
                nn.Linear(proj_dim * 2, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(config.proj_dropout),
            )
        else:
            # CLS pooling (ChemBERTa, D-MPNN)
            self._pooling_mode = "cls"

    def _run_fusion(self, batch: dict, device: torch.device):
        """Run encoder → projection → cross-attention → pooling.

        Returns struct_api [B, 128] and struct_exc [B, 128].
        """
        api_smiles = batch["api_smiles"]
        exc_smiles = batch["exc_smiles"]
        exc_available = batch["exc_available"]

        # Encode
        api_tok, api_pool, api_mask = self.encoder.encode(api_smiles)
        exc_tok, exc_pool, exc_mask = self.encoder.encode(exc_smiles)

        api_tok = api_tok.to(device)
        api_pool = api_pool.to(device)
        api_mask = api_mask.to(device)
        exc_tok = exc_tok.to(device)
        exc_pool = exc_pool.to(device)
        exc_mask = exc_mask.to(device)

        B = api_tok.shape[0]
        proj_dim = self.config.proj_dim

        # Project
        api_tok_proj = self.api_proj(api_tok)
        api_pool_proj = self.api_proj(api_pool)
        exc_tok_proj = self.exc_proj(exc_tok)
        exc_pool_proj = self.exc_proj(exc_pool)

        if self.family == "gin":
            # GIN: no cross-attention (Section 6.1)
            # Gated attention pooling + projected mean
            api_pooled, _ = self.api_pool(api_tok_proj, api_mask)
            exc_pooled, _ = self.exc_pool(exc_tok_proj, exc_mask)

            # Missing excipient fallback
            missing_exc = (exc_available == 0.0)
            if missing_exc.any():
                exc_placeholder = self.exc_global_placeholder.expand(B, -1) if hasattr(self, 'exc_global_placeholder') else torch.zeros(B, proj_dim, device=device)
                exc_pooled = torch.where(missing_exc.unsqueeze(1), exc_placeholder, exc_pooled)

            api_combined = torch.cat([api_pool_proj, api_pooled], dim=-1)
            exc_combined = torch.cat([exc_pool_proj, exc_pooled], dim=-1)
            struct_api = self.api_pool_fusion(api_combined)
            struct_exc = self.exc_pool_fusion(exc_combined)
            return struct_api, struct_exc

        # ── Cross-attention path (MoLFormer, ChemBERTa, GAT, D-MPNN) ────
        # Prepend CLS token
        api_cls = api_pool_proj.unsqueeze(1)
        api_seq = torch.cat([api_cls, api_tok_proj], dim=1)
        api_mask_ext = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device),
            api_mask
        ], dim=1)

        exc_cls = exc_pool_proj.unsqueeze(1)
        exc_seq = torch.cat([exc_cls, exc_tok_proj], dim=1)
        exc_mask_ext = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device),
            exc_mask
        ], dim=1)

        # Handle missing excipients
        exc_avail_mask = exc_available.unsqueeze(1).unsqueeze(2)
        exc_avail_mask_1d = exc_available.unsqueeze(1)

        exc_seq = exc_seq * exc_avail_mask
        inv_mask = (1 - exc_available).unsqueeze(1).unsqueeze(2)
        global_placeholder = self.exc_global_placeholder.expand(B, -1).unsqueeze(1)
        placeholder_seq = self.exc_placeholder.expand(B, 1, -1)

        exc_seq_cls = exc_seq[:, :1, :] + inv_mask * global_placeholder
        if exc_seq.shape[1] > 1:
            exc_seq_tok1 = exc_seq[:, 1:2, :] + inv_mask * placeholder_seq
            exc_seq_rest = exc_seq[:, 2:, :]
            exc_seq = torch.cat([exc_seq_cls, exc_seq_tok1, exc_seq_rest], dim=1)
        else:
            exc_seq = exc_seq_cls

        exc_mask_ext = exc_mask_ext & (exc_avail_mask_1d.bool())
        for i in range(B):
            if exc_available[i].item() == 0.0:
                exc_mask_ext[i, :] = True
                exc_mask_ext[i, 0] = False

        # Bidirectional cross-attention
        refined_exc, _ = self.cross_attn_exc(
            query=exc_seq, key=api_seq, value=api_seq,
            key_padding_mask=api_mask_ext,
        )
        refined_exc = torch.nan_to_num(refined_exc)
        refined_exc = self.ln_exc(exc_seq + refined_exc)

        refined_api, _ = self.cross_attn_api(
            query=api_seq, key=exc_seq, value=exc_seq,
            key_padding_mask=exc_mask_ext,
        )
        refined_api = torch.nan_to_num(refined_api)
        refined_api = self.ln_api(api_seq + refined_api)

        # ── Extract structural representation ────────────────────────────
        if self._pooling_mode == "cls":
            struct_api = refined_api[:, 0, :]
            struct_exc = refined_exc[:, 0, :]
        else:
            # Global + Gated Attention Pooling
            api_global = refined_api[:, 0, :]
            exc_global = refined_exc[:, 0, :]
            api_tokens = refined_api[:, 1:, :]
            exc_tokens = refined_exc[:, 1:, :]
            api_token_mask = api_mask_ext[:, 1:]
            exc_token_mask = exc_mask_ext[:, 1:]

            api_token_pool, _ = self.api_pool(api_tokens, api_token_mask)
            exc_token_pool, _ = self.exc_pool(exc_tokens, exc_token_mask)

            # Missing excipient fallback
            missing_exc = (exc_available == 0.0)
            if missing_exc.any():
                exc_placeholder_pool = self.exc_global_placeholder.expand(B, -1)
                exc_token_pool = torch.where(
                    missing_exc.unsqueeze(1), exc_placeholder_pool, exc_token_pool
                )

            api_combined = torch.cat([api_global, api_token_pool], dim=-1)
            exc_combined = torch.cat([exc_global, exc_token_pool], dim=-1)
            struct_api = self.api_pool_fusion(api_combined)
            struct_exc = self.exc_pool_fusion(exc_combined)

        return struct_api, struct_exc

    def _run_fixed_vector_encoder(self, batch: dict, device: torch.device):
        """Run fixed-vector encoder for MACCS/Morgan/PubChem/Mol2vec families.

        Returns raw encoder vectors (the tower handles all projection).
        """
        api_smiles = batch["api_smiles"]
        exc_smiles = batch["exc_smiles"]

        if self.encoder.is_sequence_capable:
            _, api_pool, _ = self.encoder.encode(api_smiles)
            _, exc_pool, _ = self.encoder.encode(exc_smiles)
        else:
            api_pool = self.encoder.encode(api_smiles)
            exc_pool = self.encoder.encode(exc_smiles)

        return api_pool.to(device), exc_pool.to(device)

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: dict with keys:
                api_smiles, exc_smiles, exc_available,
                api_maccs, exc_maccs,           # [B, 166]
                api_morgan, exc_morgan,         # [B, 512 or 1024]
                api_descriptors, exc_descriptors, # [B, 12]
                api_chem_flags, exc_chem_flags, # [B, 18]
                mechanism_flags,                # [B, 5]
                prior_block,                    # [B, 5]
                p_exact,                        # [B]
                # For fixed-vector families that need raw vectors:
                api_pubchem, exc_pubchem,        # [B, 881] (PubChem only)
                api_mol2vec, exc_mol2vec,        # [B, 300] (Mol2vec only)

        Returns:
            logits: [B] raw logits.
        """
        device = batch["api_descriptors"].device

        # ── Step 1: Get structural vectors ───────────────────────────────
        if self.family in FIXED_VECTOR_FAMILIES:
            # Fixed-vector families: tower takes raw features directly
            h_api = self._run_fixed_vector_tower(batch, device, side="api")
            h_exc = self._run_fixed_vector_tower(batch, device, side="exc")
        else:
            # Transformer / GNN: fusion → struct 128 → tower mix → 96
            struct_api, struct_exc = self._run_fusion(batch, device)
            h_api = self._run_deep_tower(batch, device, struct_api, side="api")
            h_exc = self._run_deep_tower(batch, device, struct_exc, side="exc")

        # ── Step 2: PGB head ─────────────────────────────────────────────
        logits = self.head(
            h_api=h_api,
            h_exc=h_exc,
            prior_block=batch["prior_block"].to(device),
            exc_chem_flags=batch["exc_chem_flags"].to(device),
            mechanism_flags=batch["mechanism_flags"].to(device),
            p_exact=batch["p_exact"].to(device),
        )

        return logits  # [B]

    def _run_fixed_vector_tower(
        self, batch: dict, device: torch.device, side: str
    ) -> torch.Tensor:
        """Run the appropriate fixed-vector tower for one side (API or EXC)."""
        tower = self.api_tower if side == "api" else self.exc_tower
        prefix = f"{side}_"

        maccs = batch[f"{prefix}maccs"].to(device)
        morgan = batch[f"{prefix}morgan"].to(device)
        descs = batch[f"{prefix}descriptors"].to(device)
        flags = batch[f"{prefix}chem_flags"].to(device)

        if self.family == "maccs":
            return tower(maccs, morgan, descs, flags)
        elif self.family == "morgan":
            # Morgan family uses 1024-bit morgan (primary) and 166-bit MACCS (side)
            return tower(morgan, maccs, descs, flags)
        elif self.family == "pubchem":
            pubchem = batch[f"{prefix}pubchem"].to(device)
            return tower(pubchem, morgan, descs, flags)
        elif self.family == "mol2vec":
            mol2vec = batch[f"{prefix}mol2vec"].to(device)
            return tower(mol2vec, maccs, morgan, descs, flags)
        else:
            raise ValueError(f"Unknown fixed-vector family: {self.family}")

    def _run_deep_tower(
        self, batch: dict, device: torch.device,
        struct: torch.Tensor, side: str,
    ) -> torch.Tensor:
        """Run the tower mix for transformer / GNN families."""
        tower = self.api_tower if side == "api" else self.exc_tower
        prefix = f"{side}_"

        descs = batch[f"{prefix}descriptors"].to(device)
        flags = batch[f"{prefix}chem_flags"].to(device)

        if self.family == "molformer":
            # Side branch: MACCS 166
            side_features = batch[f"{prefix}maccs"].to(device)
        elif self.family == "chemberta":
            # Side branch: Morgan 512
            side_features = batch[f"{prefix}morgan"].to(device)
        elif self.family in GNN_FAMILIES:
            # MACCS 166
            side_features = batch[f"{prefix}maccs"].to(device)
        else:
            raise ValueError(f"Unknown deep family: {self.family}")

        if self.family in GNN_FAMILIES:
            return tower(struct, side_features, descs, flags)
        else:
            return tower(struct, side_features, descs, flags)
