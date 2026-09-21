"""MoLFormer-XL encoder – Section 5.2.

Frozen HuggingFace transformer, shared for both API and Excipient.
Returns (token_embeddings, pooled_embedding, token_mask).
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class MoLFormerEncoder(nn.Module):
    """Frozen MoLFormer-XL encoder with in-memory SMILES cache."""

    is_sequence_capable = True
    output_dim = 768

    def __init__(self, model_path: str = "ibm/MoLFormer-XL-both-10pct", device: str = "cpu"):
        super().__init__()
        self.device = device
        local_dir = "models/pretrained/molformer"
        if (model_path == "ibm/MoLFormer-XL-both-10pct" or not os.path.exists(model_path)) and os.path.isdir(local_dir):
            model_path = local_dir

        is_local = os.path.isdir(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=is_local,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            deterministic_eval=True,
            trust_remote_code=True,
            local_files_only=is_local,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        # In-memory embedding cache keyed by SMILES string
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def to(self, device, *args, **kwargs):
        self.device = str(device)
        # Clear cache when device changes
        self._cache.clear()
        return super().to(device, *args, **kwargs)

    @torch.no_grad()
    def encode(self, smiles_batch: list[str]):
        """Encode a batch of SMILES strings.
        
        Returns:
            token_embeddings: [B, L, 768]
            pooled_embedding: [B, 768]
            token_mask: [B, L] bool, True = padding (to be excluded)
        """
        # Check which SMILES need computation
        uncached = [s for s in smiles_batch if s not in self._cache]

        if uncached:
            # Tokenize and encode uncached SMILES
            tokens = self.tokenizer(
                uncached,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            outputs = self.model(**tokens)

            # last_hidden_state: [B_uncached, L_full, 768]
            # pooler_output: [B_uncached, 768]
            last_hidden = outputs.last_hidden_state
            pooler = outputs.pooler_output

            attn_mask = tokens["attention_mask"]  # [B, L_full], 1=real, 0=pad

            # Exclude the special tokens (first and last positions)
            # token_embeddings = all positions except [CLS] and [SEP]
            # But MoLFormer uses different special tokens – let's just take all
            # non-special positions. The safest approach: keep all hidden states
            # but mask with the attention mask.
            # token_mask: True where padded (PyTorch MHA convention)
            token_mask = attn_mask == 0  # True = padding

            # Cache per-SMILES results (detached, on current device)
            for i, smi in enumerate(uncached):
                # Find the actual length (non-padded) for this molecule
                seq_len = attn_mask[i].sum().item()
                self._cache[smi] = (
                    last_hidden[i, :seq_len].detach(),   # [L_i, 768]
                    pooler[i].detach(),                   # [768]
                    token_mask[i, :seq_len].detach(),     # [L_i]
                )

        # Reassemble from cache, padding to max length in this batch
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

        # Pad to max_len
        B = len(smiles_batch)
        D = self.output_dim
        token_embeddings = torch.zeros(B, max_len, D, device=self.device)
        token_mask = torch.ones(B, max_len, dtype=torch.bool, device=self.device)  # True = pad

        for i, (tok_emb, tmask) in enumerate(zip(token_embeds_list, mask_list)):
            L_i = tok_emb.shape[0]
            token_embeddings[i, :L_i] = tok_emb
            token_mask[i, :L_i] = tmask

        pooled_embedding = torch.stack(pooled_list)  # [B, 768]

        return token_embeddings, pooled_embedding, token_mask
