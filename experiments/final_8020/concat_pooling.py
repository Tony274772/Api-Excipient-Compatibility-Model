"""
Concat-fusion CompatibilityModel with gated attention pooling over the encoder's atom tokens.

src.model.CompatibilityModel ignores `pooling` when fusion == "concat": it always uses the encoder's
pooled readout (the mean over atom embeddings for D-MPNN / GAT). That made every *_concat_global_gated_*
run the same network as its *_concat_cls_* twin. This class gives the global-gated variant its own
architecture, the concat counterpart of the cross-attention global-gated path (src/model.py):

    tokens  = proj(atom embeddings)                     [B, L, 128]
    global  = proj(encoder readout)                     [B, 128]
    pooled  = GatedAttentionPooling(tokens, mask)       [B, 128]
    struct  = pool_fusion([global, pooled])             [B, 128]   (Linear 256->128, LN, GELU, Dropout)

followed by the unchanged descriptor fusion, interaction terms and classifier of CompatibilityModel.
No cross-attention: API and excipient are still encoded independently, as in every concat model.
The *_concat_cls_* models keep using CompatibilityModel (struct = proj(encoder readout)).
"""

import torch
import torch.nn as nn

from src.model import CompatibilityModel, GatedAttentionPooling


class ConcatGatedPoolingModel(CompatibilityModel):
    def __init__(self, config, encoder):
        if config.fusion != "concat" or not encoder.is_sequence_capable:
            raise ValueError("ConcatGatedPoolingModel needs fusion='concat' and a sequence-capable (token) encoder, "
                             f"got fusion={config.fusion!r}, is_sequence_capable={encoder.is_sequence_capable}")
        super().__init__(config, encoder)
        proj_dim = config.proj_dim
        pool_hidden_dim = getattr(config, "pool_hidden_dim", proj_dim)
        self.api_pool = GatedAttentionPooling(input_dim=proj_dim, hidden_dim=pool_hidden_dim)
        self.exc_pool = GatedAttentionPooling(input_dim=proj_dim, hidden_dim=pool_hidden_dim)
        self.api_pool_fusion = nn.Sequential(nn.Linear(proj_dim * 2, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
                                             nn.Dropout(config.proj_dropout))
        self.exc_pool_fusion = nn.Sequential(nn.Linear(proj_dim * 2, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
                                             nn.Dropout(config.proj_dropout))

    def _struct(self, smiles, proj, pool, fusion, device):
        tok, readout, mask = self.encoder.encode(smiles)
        tok, readout, mask = tok.to(device), readout.to(device), mask.to(device)
        if bool(mask.all(dim=1).any()):
            raise ValueError("a molecule has no valid atom tokens; gated pooling would return zeros")
        pooled, _ = pool(proj(tok), mask)
        return fusion(torch.cat([proj(readout), pooled], dim=-1))

    def forward(self, batch: dict) -> torch.Tensor:
        api_desc, exc_desc, exc_available = batch["api_desc"], batch["exc_desc"], batch["exc_available"]
        if bool((exc_available == 0).any()):
            raise ValueError("missing excipient in batch: ConcatGatedPoolingModel has no missing-excipient path")
        device = api_desc.device
        h_api_struct = self._struct(batch["api_smiles"], self.api_proj, self.api_pool, self.api_pool_fusion, device)
        h_exc_struct = self._struct(batch["exc_smiles"], self.exc_proj, self.exc_pool, self.exc_pool_fusion, device)

        # descriptor fusion, interaction terms and classifier: identical to CompatibilityModel.forward
        if self.config.use_descriptors:
            h_api = torch.cat([h_api_struct, self.api_desc_proj(api_desc)], dim=1)
            h_exc = torch.cat([h_exc_struct, self.exc_desc_proj(exc_desc), exc_available.unsqueeze(1)], dim=1)
        else:
            h_api = h_api_struct
            h_exc = torch.cat([h_exc_struct, exc_available.unsqueeze(1)], dim=1)
        h_exc_core = h_exc[:, :h_api.shape[1]]
        pair_vector = torch.cat([h_api, h_exc, h_api * h_exc_core, torch.abs(h_api - h_exc_core)], dim=1)
        return self.classifier(pair_vector).squeeze(1)
