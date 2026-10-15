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


class PairwiseAttentionPooling(nn.Module):
    """Learned attention pooling over flattened token-pair features with masking.

    Args:
        input_dim: Dimension of pair features (default: 512).
        hidden_dim: Hidden dimension for scoring net (default: 256).

    Forward Args:
        x: [B, N_pairs, 512]
        padding_mask: Bool tensor [B, N_pairs], True = invalid / padding, False = valid

    Returns:
        pooled: [B, 512]
        weights: [B, N_pairs] normalized attention weights over valid pairs
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        scores = self.score_net(x).squeeze(-1)  # [B, N_pairs]

        if padding_mask is None:
            valid_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            valid_mask = ~padding_mask

        # Detect rows where all pairs are invalid (e.g. missing excipient)
        all_invalid = (~valid_mask).all(dim=-1, keepdim=True)  # [B, 1]

        scores = scores.masked_fill(~valid_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)

        # Explicitly zero invalid locations
        weights = weights * valid_mask.to(weights.dtype)

        # Re-normalize valid weights safely
        denom = weights.sum(dim=-1, keepdim=True)
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        weights = weights / safe_denom

        # If all pairs were invalid, zero out weights strictly
        weights = weights.masked_fill(all_invalid, 0.0)

        # Use bmm to compute weighted sum directly without allocating [B, N_pairs, 512] tensor
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        pooled = torch.where(all_invalid, torch.zeros_like(pooled), pooled)

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
        self.is_explicit_pairwise = (
            self.use_cross_attn and getattr(config, "pooling", "global_gated_attention") == "explicit_pairwise"
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

            if self.is_explicit_pairwise:
                # Explicit Pairwise Token Interaction pooling (pre_pairwise.md)
                pair_pool_hidden_dim = getattr(config, "pair_pool_hidden_dim", 256)
                self.pairwise_pool = PairwiseAttentionPooling(
                    input_dim=512,
                    hidden_dim=pair_pool_hidden_dim,
                )
            elif getattr(config, "pooling", "global_gated_attention") == "global_gated_attention":
                # Gated attention pooling using config.pool_hidden_dim (default)
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

        # Classifier head configuration
        if self.is_explicit_pairwise:
            # Pairwise architecture: 512 (pair) + 128 (api_global) + 128 (exc_global) = 768
            # Plus descriptors: 24 (api) + 24 (exc) + 1 (exc_available) = 49 -> 817 total
            clf_in_dim = 768 + (config.desc_proj_dim * 2 + 1 if config.use_descriptors else 1)
            self.classifier = nn.Sequential(
                nn.Linear(clf_in_dim, 256),
                nn.GELU(),
                nn.Dropout(config.clf_dropout_1),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Dropout(config.clf_dropout_2),
                nn.Linear(64, 1),
            )
        else:
            # Standard classifier head (609 -> 128 -> 64 -> 1)
            struct_dim = proj_dim  # 128
            if config.use_descriptors:
                h_api_dim = struct_dim + config.desc_proj_dim  # 128 + 24 = 152
                h_exc_dim = struct_dim + config.desc_proj_dim + 1  # 128 + 24 + 1 = 153
            else:
                h_api_dim = struct_dim  # 128
                h_exc_dim = struct_dim + 1  # 129

            pair_dim = h_api_dim + h_exc_dim + h_api_dim + h_api_dim  # 609
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
            if pooling_mode == "explicit_pairwise":
                # --- Explicit Pairwise Token Interaction (pre_pairwise.md) ---
                api_global = refined_api[:, 0, :]   # [B, 128]
                exc_global = refined_exc[:, 0, :]   # [B, 128]

                # Molecular tokens: positions 1..L
                # [:, 1:] explicitly excludes the prepended global token (index 0)
                # and reuses the existing masks (excluding position 0).
                api_tokens = refined_api[:, 1:, :]  # [B, L_api, 128]
                exc_tokens = refined_exc[:, 1:, :]  # [B, L_exc, 128]

                # Reusing existing padding masks for molecular tokens (excluding position 0)
                api_token_mask = api_mask_ext[:, 1:]  # [B, L_api]
                exc_token_mask = exc_mask_ext[:, 1:]  # [B, L_exc]

                L_a = api_tokens.size(1)
                L_e = exc_tokens.size(1)

                # Memory-conscious vectorized chunking over batch dimension B
                # to prevent CUDA OOM on GPUs with <= 6GB VRAM for large L_a * L_e.
                max_pairs_per_chunk = 32768
                chunk_size = max(1, min(B, max_pairs_per_chunk // max(1, L_a * L_e)))

                pair_pooled_list = []
                pair_weights_list = []

                for start_idx in range(0, B, chunk_size):
                    end_idx = min(start_idx + chunk_size, B)
                    api_chunk = api_tokens[start_idx:end_idx]          # [b, L_a, 128]
                    exc_chunk = exc_tokens[start_idx:end_idx]          # [b, L_e, 128]
                    api_mask_chunk = api_token_mask[start_idx:end_idx]  # [b, L_a]
                    exc_mask_chunk = exc_token_mask[start_idx:end_idx]  # [b, L_e]
                    b = end_idx - start_idx

                    api_exp = api_chunk.unsqueeze(2)  # [b, L_a, 1, 128]
                    exc_exp = exc_chunk.unsqueeze(1)  # [b, 1, L_e, 128]

                    prod = api_exp * exc_exp
                    diff = torch.abs(api_exp - exc_exp)

                    pair_tensor_chunk = torch.cat(
                        [
                            api_exp.expand(-1, -1, L_e, -1),
                            exc_exp.expand(-1, L_a, -1, -1),
                            prod,
                            diff,
                        ],
                        dim=-1,
                    )  # [b, L_a, L_e, 512]

                    # Pair padding mask: invalid if either token is invalid
                    # True = invalid pair, False = valid pair
                    pair_mask_chunk = api_mask_chunk.unsqueeze(2) | exc_mask_chunk.unsqueeze(1)  # [b, L_a, L_e]

                    pair_flat_chunk = pair_tensor_chunk.view(b, L_a * L_e, 512)
                    pair_mask_flat_chunk = pair_mask_chunk.view(b, L_a * L_e)

                    # Pairwise attention pooling
                    pooled_chunk, weights_chunk = self.pairwise_pool(pair_flat_chunk, pair_mask_flat_chunk)
                    pair_pooled_list.append(pooled_chunk)
                    pair_weights_list.append(weights_chunk)

                pair_pooled = torch.cat(pair_pooled_list, dim=0)    # [B, 512]
                pair_weights = torch.cat(pair_weights_list, dim=0)  # [B, L_a*L_e]

                # Store detached pair attention weights internally for inspection
                self.pair_token_weights = pair_weights.detach()

                # Retain cross-attended global API and Excipient vectors separately at pairwise end
                pair_core = torch.cat([pair_pooled, api_global, exc_global], dim=1)  # [B, 512 + 128 + 128] = [B, 768]

                # Descriptor fusion
                if self.config.use_descriptors:
                    d_api = self.api_desc_proj(api_desc)  # [B, 24]
                    d_exc = self.exc_desc_proj(exc_desc)  # [B, 24]
                    pair_vector = torch.cat(
                        [pair_core, d_api, d_exc, exc_available.unsqueeze(1)],
                        dim=1,
                    )  # [B, 768 + 24 + 24 + 1] = [B, 817]
                else:
                    pair_vector = torch.cat(
                        [pair_core, exc_available.unsqueeze(1)],
                        dim=1,
                    )  # [B, 769]

                logits = self.classifier(pair_vector)  # [B, 1]
                return logits.squeeze(1)  # [B]

            elif pooling_mode == "cls":
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


# ═══════════════════════════════════════════════════════════════════════════
# Prior-Gated Bilinear (PGB) Architecture
# ═══════════════════════════════════════════════════════════════════════════


class MoleculeTower(nn.Module):
    """
    Turns a single molecule's features into a 96-d vector h.
    Input: struct_vec [B, struct_dim], maccs [B,167], morgan [B,512],
           pgb_desc [B,12], flags [B,18], salt_context [B,8].
    """

    def __init__(
        self,
        struct_dim: int,
        primary_dim: int,
        secondary_dim: int,
        pgb_dim: int = 96,
        dropout: float = 0.15,
        use_missing_embedding: bool = False,
        use_salt_context: bool = True,
    ):
        super().__init__()
        self.has_struct = struct_dim > 0
        self.has_secondary = secondary_dim > 0
        self.use_missing_embedding = use_missing_embedding
        self.use_salt_context = use_salt_context

        # Structural branch
        if self.has_struct:
            self.struct_block = nn.Identity()
            struct_out = struct_dim
        else:
            struct_out = 0

        # Primary fingerprint branch
        self.primary_block = nn.Sequential(
            nn.Linear(primary_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if self.use_missing_embedding:
            self.missing_primary = nn.Parameter(torch.zeros(64))

        # Secondary fingerprint branch
        if self.has_secondary:
            secondary_out = 32
            self.secondary_block = nn.Sequential(
                nn.Linear(secondary_dim, secondary_out),
                nn.LayerNorm(secondary_out),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
        else:
            secondary_out = 0

        # Descriptor block: 12 → 24
        self.desc_block = nn.Sequential(
            nn.Linear(12, 24),
            nn.LayerNorm(24),
            nn.GELU(),
        )

        # Flag block: 18 → 16
        self.flag_block = nn.Sequential(
            nn.Linear(18, 16),
            nn.GELU(),
        )

        # Mix: concat all branches → 96
        mix_in = struct_out + 64 + secondary_out + 24 + 16
        if self.use_missing_embedding:
            mix_in += 1  # avail flag
        if self.use_salt_context:
            mix_in += 8  # salt features

        self.mix = nn.Sequential(
            nn.Linear(mix_in, pgb_dim),
            nn.LayerNorm(pgb_dim),
            nn.GELU(),
        )

    def forward(
        self,
        struct_vec,     # [B, struct_dim] or None
        primary,        # [B, primary_dim]
        primary_avail,  # [B] or None
        secondary,      # [B, secondary_dim] or None
        pgb_desc,       # [B, 12]
        salt,           # [B, 8] or None
        flags,          # [B, 18]
    ):
        parts = []
        if self.has_struct and struct_vec is not None:
            parts.append(struct_vec)
            
        prim_proj = self.primary_block(primary)
        if self.use_missing_embedding and primary_avail is not None:
            # Replace unavailable with missing embedding
            missing_mask = (primary_avail == 0.0).unsqueeze(1)
            prim_proj = torch.where(missing_mask, self.missing_primary, prim_proj)
            parts.append(prim_proj)
            parts.append(primary_avail.unsqueeze(1))
        else:
            parts.append(prim_proj)

        if self.has_secondary and secondary is not None:
            parts.append(self.secondary_block(secondary))
            
        parts.append(self.desc_block(pgb_desc))
        parts.append(self.flag_block(flags))
        
        if self.use_salt_context and salt is not None:
            parts.append(salt)

        return self.mix(torch.cat(parts, dim=-1))


class PGBHead(nn.Module):
    """
    Prior-Gated Bilinear head.
    Input: h_api [B,96], h_exc [B,96], prior_vec [B,5], mech_flags [B,5], exc_flags [B,18]
    Output: logit [B]
    """

    def __init__(self, config):
        super().__init__()
        d = config.pgb_tower_dim       # 96
        r = config.pgb_bilinear_rank   # 16

        # Bilinear maps (no bias — each is a thin projection)
        self.U = nn.Linear(d, r, bias=False)
        self.V = nn.Linear(d, r, bias=False)

        # Gate network: prior_vec(5) + api_flags(18) + exc_flags(18) + mech_flags(5) = 46
        gate_in = config.pgb_prior_vec_dim + (config.pgb_num_flags * 2) + config.pgb_num_mechanisms
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 32),
            nn.GELU(),
            nn.Linear(32, r),
            nn.Sigmoid(),
        )

        # MLP: [h_api(d) + h_exc(d) + product(d) + diff(d) + gated(r) + prior(5) + mech(5)]
        concat_dim = d + d + d + d + r + config.pgb_prior_vec_dim + config.pgb_num_mechanisms  # 410
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, config.pgb_head_hidden_1),
            nn.GELU(),
            nn.Dropout(config.pgb_head_dropout_1),
            nn.Linear(config.pgb_head_hidden_1, config.pgb_head_hidden_2),
            nn.GELU(),
            nn.Dropout(config.pgb_head_dropout_2),
            nn.Linear(config.pgb_head_hidden_2, 1),
        )
        
        # Adaptive prior trust
        self.prior_trust_mlp = nn.Sequential(
            nn.Linear(config.pgb_prior_vec_dim, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.prior_bias = nn.Parameter(torch.zeros(1))

    def forward(self, h_api, h_exc, prior_vec, api_flags, exc_flags, mech_flags):
        """
        h_api, h_exc : [B, 96]
        prior_vec    : [B, 5]  — [p_exact, p_delta, p_family, log_n, unseen]
        api_flags    : [B, 18]
        exc_flags    : [B, 18]
        mech_flags   : [B, 5]
        Returns      : logit [B]
        """
        product = h_api * h_exc
        diff    = torch.abs(h_api - h_exc)

        bilinear = self.U(h_api) * self.V(h_exc)              # [B, r]

        gate_input = torch.cat([prior_vec, api_flags, exc_flags, mech_flags], dim=-1)  # [B, 46]
        gate       = self.gate(gate_input)                     # [B, r]
        gated      = bilinear * gate                           # [B, r]

        concat = torch.cat(
            [h_api, h_exc, product, diff, gated, prior_vec, mech_flags],
            dim=-1,
        )  # [B, 410]

        residual = self.mlp(concat).squeeze(-1)  # [B]

        # Adaptive Prior weight
        prior_weight = self.prior_trust_mlp(prior_vec).squeeze(-1)

        # Prior logit offset: w * log(p / (1 - p)) + b
        p = prior_vec[:, 0].clamp(1e-6, 1 - 1e-6)   # p_exact
        logit_prior = torch.log(p / (1 - p))
        logit = residual + prior_weight * logit_prior + self.prior_bias

        return logit   # [B]

    def init_bias(self, global_rate: float, pgb_prior_offset: float = 0.2):
        """
        Set the prior_bias so that at zero residual and a never-seen excipient
        (p_exact = global_rate) the output is (1 + pgb_prior_offset) * log(p/(1-p)).
        """
        if 0 < global_rate < 1:
            log_odds = np.log(global_rate / (1 - global_rate))
            self.prior_bias.data.fill_(pgb_prior_offset * log_odds)


class PGBCompatibilityModel(nn.Module):
    """
    Full Prior-Gated Bilinear model for one encoder family.

    Keeps the existing encoder (frozen or trainable) and the existing
    cross-attention/pooling layers from CompatibilityModel.
    Replaces the 609-d pair head with MoleculeTower + PGBHead.

    Instantiate by calling pgb_model_from_config(config, encoder) — do not
    call this class directly unless you know exactly what tower_spec to pass.
    """

    def __init__(self, config, encoder, api_tower: MoleculeTower, exc_tower: MoleculeTower):
        super().__init__()
        self.config  = config
        self.encoder = encoder
        self.api_tower = api_tower
        self.exc_tower = exc_tower
        self.head = PGBHead(config)

        enc_dim  = encoder.output_dim
        proj_dim = config.proj_dim   # 128 — kept from existing config

        self.use_cross_attn = (
            config.fusion == "cross_attn" and encoder.is_sequence_capable
        )

        if self.use_cross_attn:
            self.api_proj = ProjectionHead(enc_dim, proj_dim, config.proj_dropout)
            self.exc_proj = ProjectionHead(enc_dim, proj_dim, config.proj_dropout)
            self.exc_placeholder = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.01)
            self.exc_global_placeholder = nn.Parameter(torch.randn(1, proj_dim) * 0.01)
            self.cross_attn_exc = nn.MultiheadAttention(
                proj_dim, config.num_heads, config.attn_dropout, batch_first=True
            )
            self.cross_attn_api = nn.MultiheadAttention(
                proj_dim, config.num_heads, config.attn_dropout, batch_first=True
            )
            self.ln_exc = nn.LayerNorm(proj_dim)
            self.ln_api = nn.LayerNorm(proj_dim)

            pooling = getattr(config, "pooling", "global_gated_attention")
            self.pooling = pooling
            if pooling == "global_gated_attention":
                pool_hidden = getattr(config, "pool_hidden_dim", proj_dim)
                self.api_pool = GatedAttentionPooling(proj_dim, pool_hidden)
                self.exc_pool = GatedAttentionPooling(proj_dim, pool_hidden)
                self.api_pool_fusion = nn.Sequential(
                    nn.Linear(proj_dim * 2, proj_dim), nn.LayerNorm(proj_dim),
                    nn.GELU(), nn.Dropout(config.proj_dropout),
                )
                self.exc_pool_fusion = nn.Sequential(
                    nn.Linear(proj_dim * 2, proj_dim), nn.LayerNorm(proj_dim),
                    nn.GELU(), nn.Dropout(config.proj_dropout),
                )
        else:
            self.struct_dim = 0   # fixed-vector: no structural branch

    def _encode_struct(self, batch, device):
        """
        Returns h_api_struct [B,128] and h_exc_struct [B,128] for seq. encoders,
        or None, None for fixed-vector encoders (struct_dim == 0).
        Mirrors CompatibilityModel.forward logic for cross-attn + gated pooling / CLS.
        """
        if not self.use_cross_attn:
            return None, None

        api_smiles = batch["api_smiles"]
        exc_smiles = batch["exc_smiles"]
        exc_available = batch["exc_available"]

        api_tok, api_pool, api_mask = self.encoder.encode(api_smiles)
        exc_tok, exc_pool, exc_mask = self.encoder.encode(exc_smiles)

        api_tok  = api_tok.to(device);  api_pool  = api_pool.to(device);  api_mask  = api_mask.to(device)
        exc_tok  = exc_tok.to(device);  exc_pool  = exc_pool.to(device);  exc_mask  = exc_mask.to(device)

        B = api_tok.shape[0]

        api_tok_p  = self.api_proj(api_tok)
        api_pool_p = self.api_proj(api_pool)
        exc_tok_p  = self.exc_proj(exc_tok)
        exc_pool_p = self.exc_proj(exc_pool)

        api_cls = api_pool_p.unsqueeze(1)
        api_seq = torch.cat([api_cls, api_tok_p], dim=1)
        api_mask_ext = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=device), api_mask], dim=1)

        exc_cls = exc_pool_p.unsqueeze(1)
        exc_seq = torch.cat([exc_cls, exc_tok_p], dim=1)
        exc_mask_ext = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=device), exc_mask], dim=1)

        # Missing-excipient placeholder logic (same as CompatibilityModel)
        exc_avail_mask = exc_available.unsqueeze(1).unsqueeze(2)
        exc_seq = exc_seq * exc_avail_mask
        inv_mask = (1 - exc_available).unsqueeze(1).unsqueeze(2)
        exc_seq_cls  = exc_seq[:, :1, :] + inv_mask * self.exc_global_placeholder.expand(B,-1).unsqueeze(1)
        if exc_seq.shape[1] > 1:
            exc_seq_tok1 = exc_seq[:, 1:2, :] + inv_mask * self.exc_placeholder.expand(B,1,-1)
            exc_seq = torch.cat([exc_seq_cls, exc_seq_tok1, exc_seq[:, 2:, :]], dim=1)
        else:
            exc_seq = exc_seq_cls

        exc_mask_ext = exc_mask_ext & (exc_available.unsqueeze(1).bool())
        for i in range(B):
            if exc_available[i].item() == 0.0:
                exc_mask_ext[i, :] = True;  exc_mask_ext[i, 0] = False

        refined_exc, _ = self.cross_attn_exc(query=exc_seq, key=api_seq, value=api_seq,
                                              key_padding_mask=api_mask_ext)
        refined_exc = torch.nan_to_num(refined_exc)
        refined_exc = self.ln_exc(exc_seq + refined_exc)

        refined_api, _ = self.cross_attn_api(query=api_seq, key=exc_seq, value=exc_seq,
                                              key_padding_mask=exc_mask_ext)
        refined_api = torch.nan_to_num(refined_api)
        refined_api = self.ln_api(api_seq + refined_api)

        pooling = getattr(self, "pooling", "global_gated_attention")
        if pooling == "global_gated_attention":
            api_global  = refined_api[:, 0, :]
            exc_global  = refined_exc[:, 0, :]
            api_tokens  = refined_api[:, 1:, :]
            exc_tokens  = refined_exc[:, 1:, :]
            api_token_mask = api_mask_ext[:, 1:]
            exc_token_mask = exc_mask_ext[:, 1:]

            api_token_pool, _ = self.api_pool(api_tokens, api_token_mask)
            exc_token_pool, _ = self.exc_pool(exc_tokens, exc_token_mask)

            missing_exc = (exc_available == 0.0)
            exc_token_pool = torch.where(
                missing_exc.unsqueeze(1),
                self.exc_global_placeholder.expand(exc_tokens.size(0), -1),
                exc_token_pool,
            )
            api_struct = self.api_pool_fusion(torch.cat([api_global, api_token_pool], dim=-1))
            exc_struct = self.exc_pool_fusion(torch.cat([exc_global, exc_token_pool], dim=-1))
        else:
            # CLS pooling (ChemBERTa, D-MPNN)
            api_struct = refined_api[:, 0, :]
            exc_struct = refined_exc[:, 0, :]

        return api_struct, exc_struct

    def forward(self, batch: dict) -> torch.Tensor:
        device = batch["api_flags"].device

        # Structural vectors (from encoder + cross-attention)
        api_struct, exc_struct = self._encode_struct(batch, device)

        # On-the-fly feature tensors (pre-computed by PGBDataset collate)
        api_primary   = batch["api_primary"].to(device)    # [B, primary_dim]
        exc_primary   = batch["exc_primary"].to(device)
        api_prim_avail= batch["api_prim_avail"].to(device)
        exc_prim_avail= batch["exc_prim_avail"].to(device)
        api_secondary = batch.get("api_secondary")
        exc_secondary = batch.get("exc_secondary")
        if api_secondary is not None:
            api_secondary = api_secondary.to(device)
            exc_secondary = exc_secondary.to(device)
        api_pgb_desc  = batch["api_pgb_desc"].to(device)   # [B, 12]
        exc_pgb_desc  = batch["exc_pgb_desc"].to(device)
        api_salt      = batch["api_salt"].to(device)
        exc_salt      = batch["exc_salt"].to(device)
        api_flags     = batch["api_flags"].to(device)      # [B, 18]
        exc_flags     = batch["exc_flags"].to(device)
        prior_vec     = batch["prior_vec"].to(device)      # [B, 5]
        mech_flags    = batch["mech_flags"].to(device)     # [B, 5]

        # Normalized PGB descriptors
        api_pgb_desc_norm = batch["api_pgb_desc_norm"].to(device)
        exc_pgb_desc_norm = batch["exc_pgb_desc_norm"].to(device)

        h_api = self.api_tower(api_struct, api_primary, api_prim_avail, api_secondary, api_pgb_desc_norm, api_salt, api_flags)
        h_exc = self.exc_tower(exc_struct, exc_primary, exc_prim_avail, exc_secondary, exc_pgb_desc_norm, exc_salt, exc_flags)

        return self.head(h_api, h_exc, prior_vec, api_flags, exc_flags, mech_flags)


def pgb_model_from_config(config, encoder) -> "PGBCompatibilityModel":
    """
    Build a PGBCompatibilityModel appropriate for `config.encoder`.

    Tower specs per family:
      maccs      : primary=167 MACCS, secondary=512 Morgan
      morgan     : primary=1024 Morgan, secondary=167 MACCS
      pubchemfp  : primary=881 PubChem, secondary=512 Morgan
      mol2vec    : primary=300 Mol2vec, secondary=512 Morgan
      molformer  : struct=128, primary=167 MACCS, secondary=0
      chemberta  : struct=128, primary=512 Morgan, secondary=0
      pretrained_gin : struct=128, primary=167 MACCS, secondary=0
      pretrained_gat : struct=128, primary=167 MACCS, secondary=0
      dmpnn_chemprop : struct=128, primary=167 MACCS, secondary=0
    """
    enc = config.encoder
    fvs = getattr(config, "fixed_vector_source", "mol2vec")

    _tower_specs = {
        # (struct_dim, primary_dim, secondary_dim)
        "fixed_vector": {
            "maccs":     (0, 167, 512),
            "morgan":    (0, 1024, 167),
            "pubchemfp": (0, 881,  512),
            "mol2vec":   (0, 300,  512),
        },
        "seq": (128, 167, 0),
    }

    # Special case: ChemBERTa uses Morgan as primary instead of MACCS
    _chemberta_spec = (128, 512, 0)

    if enc == "fixed_vector":
        sd, pd_, sec = _tower_specs["fixed_vector"][fvs]
    elif enc == "chemberta":
        sd, pd_, sec = _chemberta_spec
    else:
        sd, pd_, sec = _tower_specs["seq"]

    use_missing = getattr(config, "pgb_use_missing_embedding", True)
    use_salt = getattr(config, "pgb_use_salt_context", True)

    api_tower = MoleculeTower(sd, pd_, sec, config.pgb_tower_dim, config.proj_dropout, use_missing_embedding=use_missing, use_salt_context=use_salt)
    exc_tower = MoleculeTower(sd, pd_, sec, config.pgb_tower_dim, config.proj_dropout, use_missing_embedding=use_missing, use_salt_context=use_salt)

    model = PGBCompatibilityModel(config, encoder, api_tower, exc_tower)
    model.head.init_bias(
        global_rate=getattr(config, "positive_prior", 0.094),
        pgb_prior_offset=config.pgb_prior_offset,
    )
    return model
