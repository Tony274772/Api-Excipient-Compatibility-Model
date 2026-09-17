"""Fusion + classifier model – Sections 6 & 7.

Encoder-agnostic: reads encoder_output_dim from config, branches on
encoder.is_sequence_capable and config.fusion to select cross-attention
or concat-only fusion.
"""

import math

import numpy as np
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Linear(D_in -> 256) -> LN -> GELU -> Dropout -> Linear(256 -> proj_dim) -> LN -> GELU."""

    def __init__(self, input_dim: int, proj_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class DescriptorProjectionHead(nn.Module):
    """Linear(21 -> 32) -> LN -> GELU -> Dropout -> Linear(32 -> 24) -> LN -> GELU."""

    def __init__(self, input_dim: int = 21, desc_proj_dim: int = 24, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, desc_proj_dim),
            nn.LayerNorm(desc_proj_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class GatedAttentionPooling(nn.Module):
    """Gated attention pooling over a variable-length sequence.

    Args:
        input_dim: Dimension of input token vectors (default: 128).
        hidden_dim: Hidden dimension for gating projection (default: 128).

    Forward Args:
        x: Token embeddings [B, L, D]
        padding_mask: Bool tensor [B, L], True = padding / invalid, False = valid

    Returns:
        pooled: Pooled representation [B, D]
        weights: Attention weights [B, L]
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.tanh_proj = nn.Linear(input_dim, hidden_dim)
        self.sigmoid_proj = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        tanh_part = torch.tanh(self.tanh_proj(x))
        sigmoid_part = torch.sigmoid(self.sigmoid_proj(x))
        gated = tanh_part * sigmoid_part
        scores = self.score(gated).squeeze(-1)  # [B, L]

        if padding_mask is not None:
            # Truly safe all-masked handling:
            # Detect rows where all tokens are masked
            all_masked = padding_mask.all(dim=-1, keepdim=True)  # [B, 1]
            scores = scores.masked_fill(padding_mask, -1e9)
            weights = torch.softmax(scores, dim=-1)
            # If all tokens in a row were masked, zero out all weights
            weights = weights.masked_fill(all_masked, 0.0)
        else:
            weights = torch.softmax(scores, dim=-1)

        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        return pooled, weights


class CompatibilityModel(nn.Module):
    """Full API-Excipient compatibility prediction model.
    
    Args:
        config: Config dataclass
        encoder: An encoder module with .is_sequence_capable, .output_dim, .encode()
    """

    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder

        enc_dim = encoder.output_dim
        proj_dim = config.proj_dim  # 128

        # Determine fusion mode
        self.use_cross_attn = (
            config.fusion == "cross_attn" and encoder.is_sequence_capable
        )

        # Projection heads for API and Excipient (separate weights)
        self.api_proj = ProjectionHead(enc_dim, proj_dim, config.proj_dropout)
        self.exc_proj = ProjectionHead(enc_dim, proj_dim, config.proj_dropout)

        if self.use_cross_attn:
            # Learned placeholders for missing excipient
            self.exc_placeholder = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.01)
            self.exc_global_placeholder = nn.Parameter(torch.randn(1, proj_dim) * 0.01)

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

            # Gated attention pooling using config.pool_hidden_dim
            pool_hidden_dim = getattr(config, "pool_hidden_dim", proj_dim)
            self.api_pool = GatedAttentionPooling(
                input_dim=proj_dim,
                hidden_dim=pool_hidden_dim,
            )
            self.exc_pool = GatedAttentionPooling(
                input_dim=proj_dim,
                hidden_dim=pool_hidden_dim,
            )

            # Separate fusion heads: 256 (global + pooled) -> 128
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

        # Descriptor projection heads
        if config.use_descriptors:
            self.api_desc_proj = DescriptorProjectionHead(
                config.num_descriptors, config.desc_proj_dim, config.desc_dropout
            )
            self.exc_desc_proj = DescriptorProjectionHead(
                config.num_descriptors, config.desc_proj_dim, config.desc_dropout
            )

        # Compute pair_dim
        struct_dim = proj_dim  # 128
        if config.use_descriptors:
            h_api_dim = struct_dim + config.desc_proj_dim  # 128 + 24 = 152
            h_exc_dim = struct_dim + config.desc_proj_dim + 1  # 128 + 24 + 1 = 153
        else:
            h_api_dim = struct_dim  # 128
            h_exc_dim = struct_dim + 1  # 129

        # interaction and difference use h_api_dim
        pair_dim = h_api_dim + h_exc_dim + h_api_dim + h_api_dim  # 152+153+152+152=609

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, config.clf_hidden_dim),  # 609 -> 128
            nn.GELU(),
            nn.Dropout(config.clf_dropout_1),  # 0.5
            nn.Linear(config.clf_hidden_dim, config.clf_hidden_dim_2),  # 128 -> 64
            nn.GELU(),
            nn.Dropout(config.clf_dropout_2),  # 0.4
            nn.Linear(config.clf_hidden_dim_2, 1),  # 64 -> 1
        )

    def init_bias(self, positive_prior: float):
        """Initialize the final layer bias to log(prior / (1 - prior))."""
        if positive_prior > 0 and positive_prior < 1:
            bias_val = math.log(positive_prior / (1 - positive_prior))
            # The last layer is classifier[-1]
            self.classifier[-1].bias.data.fill_(bias_val)

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass.
        
        Args:
            batch: dict with api_smiles, exc_smiles, api_desc, exc_desc, exc_available
            
        Returns:
            logits: [B, 1] raw logits
        """
        api_smiles = batch["api_smiles"]
        exc_smiles = batch["exc_smiles"]
        api_desc = batch["api_desc"]      # [B, 21]
        exc_desc = batch["exc_desc"]      # [B, 21]
        exc_available = batch["exc_available"]  # [B]

        device = api_desc.device

        if self.use_cross_attn:
            # --- Sequence-capable encoder + cross-attention fusion ---
            api_tok, api_pool, api_mask = self.encoder.encode(api_smiles)
            exc_tok, exc_pool, exc_mask = self.encoder.encode(exc_smiles)

            # Move to device if needed
            api_tok = api_tok.to(device)
            api_pool = api_pool.to(device)
            api_mask = api_mask.to(device)
            exc_tok = exc_tok.to(device)
            exc_pool = exc_pool.to(device)
            exc_mask = exc_mask.to(device)

            B = api_tok.shape[0]

            # Project token embeddings and pooled embeddings
            api_tok_proj = self.api_proj(api_tok)       # [B, L_api, 128]
            api_pool_proj = self.api_proj(api_pool)     # [B, 128]
            exc_tok_proj = self.exc_proj(exc_tok)       # [B, L_exc, 128]
            exc_pool_proj = self.exc_proj(exc_pool)     # [B, 128]

            # Prepend CLS token (projected pooled embedding) to token sequences
            api_cls = api_pool_proj.unsqueeze(1)        # [B, 1, 128]
            api_seq = torch.cat([api_cls, api_tok_proj], dim=1)   # [B, L_api+1, 128]
            api_mask_ext = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=device),  # CLS not padded
                api_mask
            ], dim=1)  # [B, L_api+1]

            exc_cls = exc_pool_proj.unsqueeze(1)        # [B, 1, 128]
            exc_seq = torch.cat([exc_cls, exc_tok_proj], dim=1)   # [B, L_exc+1, 128]
            exc_mask_ext = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=device),
                exc_mask
            ], dim=1)  # [B, L_exc+1]

            # Handle missing excipients: substitute placeholders when exc_available == 0
            exc_avail_mask = exc_available.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            exc_avail_mask_1d = exc_available.unsqueeze(1)  # [B, 1]

            # Zero out real excipient and splice in placeholders where unavailable
            exc_seq = exc_seq * exc_avail_mask
            # Add placeholder at CLS position (index 0) and first token position (index 1)
            placeholder_seq = self.exc_placeholder.expand(B, 1, -1)  # [B, 1, 128]
            global_placeholder = self.exc_global_placeholder.expand(B, -1).unsqueeze(1)  # [B, 1, 128]

            # When exc is unavailable, set CLS = global_placeholder, first token = placeholder
            inv_mask = (1 - exc_available).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            # Only modify CLS (index 0) and first token (index 1) for unavailable
            exc_seq_cls = exc_seq[:, :1, :] + inv_mask * global_placeholder
            if exc_seq.shape[1] > 1:
                exc_seq_tok1 = exc_seq[:, 1:2, :] + inv_mask * placeholder_seq
                exc_seq_rest = exc_seq[:, 2:, :]
                exc_seq = torch.cat([exc_seq_cls, exc_seq_tok1, exc_seq_rest], dim=1)
            else:
                exc_seq = exc_seq_cls

            # When unavailable, unmask the CLS position so attention can use placeholder
            exc_mask_ext = exc_mask_ext & (exc_avail_mask_1d.bool())
            # For unavailable excipients, only CLS is valid
            for i in range(B):
                if exc_available[i].item() == 0.0:
                    exc_mask_ext[i, :] = True
                    exc_mask_ext[i, 0] = False  # CLS is valid

            # Bidirectional cross-attention
            # Exc queries attend to API keys/values
            refined_exc, _ = self.cross_attn_exc(
                query=exc_seq,
                key=api_seq,
                value=api_seq,
                key_padding_mask=api_mask_ext,
            )
            refined_exc = torch.nan_to_num(refined_exc)
            refined_exc = self.ln_exc(exc_seq + refined_exc)  # Residual + LN

            # API queries attend to Exc keys/values
            refined_api, _ = self.cross_attn_api(
                query=api_seq,
                key=exc_seq,
                value=exc_seq,
                key_padding_mask=exc_mask_ext,
            )
            refined_api = torch.nan_to_num(refined_api)
            refined_api = self.ln_api(api_seq + refined_api)  # Residual + LN

            # Structural representation extraction
            pooling_mode = getattr(self.config, "pooling", "global_gated_attention")
            if pooling_mode == "cls":
                # Legacy CLS-only structural extraction
                h_api_struct = refined_api[:, 0, :]   # [B, 128]
                h_exc_struct = refined_exc[:, 0, :]   # [B, 128]
            else:
                # Global + Gated Attention Pooling
                # refined_api / refined_exc shape: [B, L+1, 128]
                # Position 0: prepended global token
                api_global = refined_api[:, 0, :]       # [B, 128]
                exc_global = refined_exc[:, 0, :]       # [B, 128]

                # Molecular tokens: positions 1..L
                # [:, 1:] explicitly excludes the prepended global token (index 0)
                # and reuses the existing masks (excluding position 0).
                api_tokens = refined_api[:, 1:, :]      # [B, L_api, 128]
                exc_tokens = refined_exc[:, 1:, :]      # [B, L_exc, 128]

                # Reusing existing padding masks for molecular tokens (excluding position 0)
                api_token_mask = api_mask_ext[:, 1:]    # [B, L_api]
                exc_token_mask = exc_mask_ext[:, 1:]    # [B, L_exc]

                # Gated attention pooling over valid molecular tokens
                api_token_pool, api_token_weights = self.api_pool(
                    api_tokens,
                    api_token_mask,
                )
                exc_token_pool, exc_token_weights = self.exc_pool(
                    exc_tokens,
                    exc_token_mask,
                )

                # Missing-excipient fallback:
                # Explicitly preserve current missing-excipient behavior:
                # Where exc_available == 0, only position 0 (global) is valid,
                # and the token-pooling branch falls back to the learned placeholder.
                missing_exc = (exc_available == 0.0)    # [B]
                exc_placeholder_pool = self.exc_global_placeholder.expand(
                    exc_tokens.size(0),
                    -1,
                )                                        # [B, 128]
                exc_token_pool = torch.where(
                    missing_exc.unsqueeze(1),
                    exc_placeholder_pool,
                    exc_token_pool,
                )

                # Concatenate global + pooled: [B, 256]
                api_combined = torch.cat([api_global, api_token_pool], dim=-1)  # [B, 256]
                exc_combined = torch.cat([exc_global, exc_token_pool], dim=-1)  # [B, 256]

                # Fusion 256 -> 128
                h_api_struct = self.api_pool_fusion(api_combined)  # [B, 128]
                h_exc_struct = self.exc_pool_fusion(exc_combined)  # [B, 128]

                # Store detached attention weights internally for inspection/visualization
                self.api_token_weights = api_token_weights.detach()
                self.exc_token_weights = exc_token_weights.detach()

        else:
            # --- Concat-only fusion (no cross-attention) ---
            if self.encoder.is_sequence_capable:
                _, api_pool, _ = self.encoder.encode(api_smiles)
                _, exc_pool, _ = self.encoder.encode(exc_smiles)
                api_pool = api_pool.to(device)
                exc_pool = exc_pool.to(device)
            else:
                api_pool = self.encoder.encode(api_smiles).to(device)
                exc_pool = self.encoder.encode(exc_smiles).to(device)

            h_api_struct = self.api_proj(api_pool)   # [B, 128]
            h_exc_struct = self.exc_proj(exc_pool)   # [B, 128]

        # Descriptor fusion (Section 6.3)
        if self.config.use_descriptors:
            d_api = self.api_desc_proj(api_desc)    # [B, 24]
            d_exc = self.exc_desc_proj(exc_desc)    # [B, 24]

            h_api = torch.cat([h_api_struct, d_api], dim=1)   # [B, 152]
            h_exc = torch.cat([h_exc_struct, d_exc, exc_available.unsqueeze(1)], dim=1)  # [B, 153]
        else:
            h_api = h_api_struct   # [B, 128]
            h_exc = torch.cat([h_exc_struct, exc_available.unsqueeze(1)], dim=1)  # [B, 129]

        # Interaction terms
        h_exc_core = h_exc[:, :h_api.shape[1]]   # truncate exc_available scalar
        interaction = h_api * h_exc_core           # element-wise product
        difference = torch.abs(h_api - h_exc_core)  # element-wise abs difference

        # Pair vector
        pair_vector = torch.cat([h_api, h_exc, interaction, difference], dim=1)  # [B, 609]

        # Classifier
        logits = self.classifier(pair_vector)  # [B, 1]
        return logits.squeeze(1)  # [B]
