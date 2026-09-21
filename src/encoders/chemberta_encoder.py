"""ChemBERTa encoder – Section 5.4 (optional).

Frozen HuggingFace SMILES transformer.
Returns (token_embeddings, pooled_embedding, token_mask).
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ChemBERTaEncoder(nn.Module):
    """Frozen ChemBERTa encoder with SMILES cache."""

    is_sequence_capable = True

    def __init__(self, model_path: str = "DeepChem/ChemBERTa-77M-MTR", device: str = "cpu"):
        super().__init__()
        self.device = device
        local_dir = "models/pretrained/chemberta"
        if (model_path == "DeepChem/ChemBERTa-77M-MTR" or not os.path.exists(model_path)) and os.path.isdir(local_dir):
            model_path = local_dir

        is_local = os.path.isdir(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=is_local,
        )
        # Force safetensors to bypass torch.load() CVE check in torch < 2.6
        self.model = AutoModel.from_pretrained(
            model_path,
            use_safetensors=True,
            local_files_only=is_local,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        # Set output_dim from actual model config
        self.output_dim = self.model.config.hidden_size

        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def to(self, device, *args, **kwargs):
        self.device = str(device)
        self._cache.clear()
        return super().to(device, *args, **kwargs)

    @torch.no_grad()
    def encode(self, smiles_batch: list[str]):
        """Encode a batch of SMILES.
        
        Returns:
            token_embeddings: [B, L, D]
            pooled_embedding: [B, D] (mean-pooled, not from a pooler head)
            token_mask: [B, L] bool, True = padding
        """
        uncached = [s for s in smiles_batch if s not in self._cache]

        if uncached:
            tokens = self.tokenizer(
                uncached,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            outputs = self.model(**tokens)
            last_hidden = outputs.last_hidden_state  # [B, L, D]
            attn_mask = tokens["attention_mask"]       # [B, L]

            # Mean-pool over non-padding tokens (ChemBERTa may not have pooler_output)
            mask_expanded = attn_mask.unsqueeze(-1).float()  # [B, L, 1]
            sum_hidden = (last_hidden * mask_expanded).sum(dim=1)  # [B, D]
            count = mask_expanded.sum(dim=1).clamp(min=1)            # [B, 1]
            pooled = sum_hidden / count  # [B, D]

            token_mask = attn_mask == 0  # True = padding

            for i, smi in enumerate(uncached):
                seq_len = attn_mask[i].sum().item()
                self._cache[smi] = (
                    last_hidden[i, :seq_len].detach(),
                    pooled[i].detach(),
                    token_mask[i, :seq_len].detach(),
                )

        # Reassemble from cache
        token_embeds_list = []
        pooled_list = []
        mask_list = []
        max_len = 0

        for smi in smiles_batch:
            tok_emb, pool_emb, tmask = self._cache[smi]
            token_embeds_list.append(tok_emb)
            pooled_list.append(pool_emb)
            mask_list.append(tmask)
            max_len = max(max_len, tok_emb.shape[0])

        B = len(smiles_batch)
        D = self.output_dim
        token_embeddings = torch.zeros(B, max_len, D, device=self.device)
        token_mask_out = torch.ones(B, max_len, dtype=torch.bool, device=self.device)

        for i, (tok_emb, tmask) in enumerate(zip(token_embeds_list, mask_list)):
            L_i = tok_emb.shape[0]
            token_embeddings[i, :L_i] = tok_emb
            token_mask_out[i, :L_i] = tmask

        pooled_embedding = torch.stack(pooled_list)

        return token_embeddings, pooled_embedding, token_mask_out
