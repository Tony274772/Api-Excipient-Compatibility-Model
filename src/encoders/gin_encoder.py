"""Pretrained GIN encoder – Section 5.3.

5-layer GIN from Hu et al. (ICLR 2020), loaded via dgllife.
Frozen, shared for both API and Excipient.
Returns (token_embeddings, pooled_embedding, token_mask).
"""

import torch
import torch.nn as nn
import dgl
from rdkit import Chem

from dgllife.model import load_pretrained
from dgllife.utils import (
    mol_to_bigraph,
    PretrainAtomFeaturizer,
    PretrainBondFeaturizer,
)


class PretrainedGINEncoder(nn.Module):
    """Frozen pretrained GIN encoder with SMILES cache."""

    is_sequence_capable = True
    output_dim = 300

    def __init__(self, pretrained_name: str = "gin_supervised_contextpred", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.atom_featurizer = PretrainAtomFeaturizer()
        self.bond_featurizer = PretrainBondFeaturizer()

        self.model = load_pretrained(pretrained_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # In-memory embedding cache keyed by SMILES string
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def to(self, device, *args, **kwargs):
        self.device = str(device)
        self._cache.clear()
        return super().to(device, *args, **kwargs)

    def _smiles_to_graph(self, smiles: str) -> dgl.DGLGraph:
        """Convert SMILES to a DGL graph with atom/bond features."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit cannot parse SMILES: {smiles}")

        g = mol_to_bigraph(
            mol,
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer,
            add_self_loop=False,
        )
        if g is None:
            raise ValueError(f"Could not create graph from SMILES: {smiles}")
        return g

    @torch.no_grad()
    def _encode_single(self, smiles: str):
        """Encode a single SMILES, returning (node_embeddings, pooled)."""
        if smiles in self._cache:
            return self._cache[smiles]

        g = self._smiles_to_graph(smiles)
        g = g.to(self.device)

        # PretrainAtomFeaturizer stores: 'atomic_number' and 'chirality_type'
        # PretrainBondFeaturizer stores: 'bond_type' and 'bond_direction_type'
        # The GIN model's forward expects:
        #   forward(g, categorical_node_feats, categorical_edge_feats)
        # where each is a list of 1D LongTensors (one per categorical feature).
        categorical_node_feats = [
            g.ndata['atomic_number'].to(self.device),
            g.ndata['chirality_type'].to(self.device),
        ]

        if g.num_edges() > 0:
            categorical_edge_feats = [
                g.edata['bond_type'].to(self.device),
                g.edata['bond_direction_type'].to(self.device),
            ]
        else:
            # Single-atom molecules have no edges
            categorical_edge_feats = [
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.long, device=self.device),
            ]

        # Forward pass through the GIN model
        node_embeddings = self.model(g, categorical_node_feats, categorical_edge_feats)
        # node_embeddings: [num_atoms, 300]

        # Pool: mean over all atoms
        pooled = node_embeddings.mean(dim=0)  # [300]

        self._cache[smiles] = (node_embeddings.detach(), pooled.detach())
        return node_embeddings.detach(), pooled.detach()

    @torch.no_grad()
    def encode(self, smiles_batch: list[str]):
        """Encode a batch of SMILES.
        
        Returns:
            token_embeddings: [B, L, 300] padded
            pooled_embedding: [B, 300]
            token_mask: [B, L] bool, True = padding
        """
        all_node_embeds = []
        all_pooled = []
        max_atoms = 0

        for smi in smiles_batch:
            node_emb, pooled = self._encode_single(smi)
            all_node_embeds.append(node_emb)
            all_pooled.append(pooled)
            max_atoms = max(max_atoms, node_emb.shape[0])

        B = len(smiles_batch)
        D = self.output_dim

        # Pad to max_atoms
        token_embeddings = torch.zeros(B, max_atoms, D, device=self.device)
        token_mask = torch.ones(B, max_atoms, dtype=torch.bool, device=self.device)  # True = pad

        for i, node_emb in enumerate(all_node_embeds):
            n_atoms = node_emb.shape[0]
            token_embeddings[i, :n_atoms] = node_emb
            token_mask[i, :n_atoms] = False  # real atoms are not padding

        pooled_embedding = torch.stack(all_pooled)  # [B, 300]

        return token_embeddings, pooled_embedding, token_mask
