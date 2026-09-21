"""Pretrained GIN encoder – Section 5.3.

5-layer GIN from Hu et al. (ICLR 2020), loaded via dgllife.
Frozen, shared for both API and Excipient.
Returns (token_embeddings, pooled_embedding, token_mask).
"""

import os
import sys
import unittest.mock
# Bypass graphbolt C++ library requirement on Windows / PyTorch >= 2.4
if 'dgl.graphbolt' not in sys.modules:
    sys.modules['dgl.graphbolt'] = unittest.mock.MagicMock()

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
        self.bond_featurizer = PretrainBondFeaturizer(self_loop=True)

        local_pth = "models/pretrained/gin/gin_supervised_contextpred_pre_trained.pth"
        if os.path.isfile(pretrained_name):
            pth_path = pretrained_name
        elif os.path.isfile(local_pth):
            pth_path = local_pth
        else:
            pth_path = None

        if pth_path is not None:
            from dgllife.model.pretrain import create_property_model
            model = create_property_model("gin_supervised_contextpred")
            ckpt = torch.load(pth_path, map_location="cpu")
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
        else:
            model = load_pretrained(pretrained_name)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        
        # Hide the model in a standard Python list so PyTorch's `to()` 
        # doesn't recursively find it and move it to CUDA.
        self._model_list = [model]

        # In-memory embedding cache keyed by SMILES string
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def model(self):
        return self._model_list[0]

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
            add_self_loop=True,
        )
        if g is None:
            raise ValueError(f"Could not create graph from SMILES: {smiles}")
        return g

    @torch.no_grad()
    def _encode_single(self, smiles: str):
        """Encode a single SMILES, returning (node_embeddings, pooled).

        DGL CPU-only wheel cannot move graphs to CUDA. Since the GIN is
        frozen (no gradients), we run the entire forward pass on CPU and
        then move only the output float tensors to the target device.
        """
        if smiles in self._cache:
            return self._cache[smiles]

        # Build graph on CPU (DGL CPU wheel cannot handle g.to('cuda'))
        g = self._smiles_to_graph(smiles)  # stays on CPU

        # Feature tensors on CPU for the GIN forward pass
        categorical_node_feats = [
            g.ndata['atomic_number'],   # CPU LongTensor
            g.ndata['chirality_type'],  # CPU LongTensor
        ]

        if g.num_edges() > 0:
            categorical_edge_feats = [
                g.edata['bond_type'],          # CPU LongTensor
                g.edata['bond_direction_type'],# CPU LongTensor
            ]
        else:
            # Single-atom molecules with only self-loops have no bonds
            categorical_edge_feats = [
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            ]

        # GIN forward on CPU, then move float outputs to target device
        node_embeddings = self.model(g, categorical_node_feats, categorical_edge_feats)
        # node_embeddings: [num_atoms, 300]  (CPU float)
        node_embeddings = node_embeddings.to(self.device)

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
