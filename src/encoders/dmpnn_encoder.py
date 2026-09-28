"""D-MPNN encoder — Chemprop-style directed bond message passing.

Follows Section 8 of the GNN specification.

Two modes:
    1. "chemeleon" (default for dmpnn_chemprop):
       Load CheMeleon pretrained weights via native Chemprop architecture.
       Uses Chemprop's own featurizer — NOT the shared 39/10 PyG schema.
       Native hidden width is preserved (not forced to 300).

    2. "scratch" (dmpnn_scratch):
       Local D-MPNN implementation using the shared 39/10 feature schema.
       Hidden dim = 300, depth = 3.

The CheMeleon checkpoint must be downloaded once and stored locally.
Normal training/inference MUST load from the local filesystem.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.gnn_common import (
    PyGGNNEncoderBase,
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    smiles_to_pyg_data,
)


# ─── Constants ────────────────────────────────────────────────────────────────

CHEMELEON_MD5 = "6a80b54fdb7de37ef0374d302f01e8ce"
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHEMELEON_DEFAULT_PATH = "models/pretrained/chemeleon_mp.pt"


# ─── Scratch D-MPNN (local 39/10 implementation) ─────────────────────────────


class ScratchDMPNNNetwork(nn.Module):
    """Directed bond message-passing network (scratch implementation).

    Uses the shared 39/10 feature schema. Follows Chemprop D-MPNN semantics:
    - Messages live on directed edges
    - Each edge message is updated from incoming edges excluding the reverse edge
    - Final atom states aggregate incoming edge messages
    """

    def __init__(
        self,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden_dim: int = 300,
        depth: int = 3,
        dropout: float = 0.10,
        bias: bool = False,
    ):
        super().__init__()
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout_rate = dropout

        # Initial edge message: W_i([x_v || e_vw])
        self.W_i = nn.Linear(atom_dim + bond_dim, hidden_dim, bias=bias)
        # Message update: W_h(m_vw)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # Final atom output: W_o([x_v || m_v])
        self.W_o = nn.Linear(atom_dim + hidden_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch=None):
        """Run D-MPNN message passing.

        Args:
            x:          [N, atom_dim] atom features
            edge_index: [2, E] directed edges (u->v for each direction)
            edge_attr:  [E, bond_dim] bond features
            batch:      [N] batch assignment (unused)

        Returns:
            node_embeddings: [N, hidden_dim]
        """
        num_atoms = x.size(0)
        num_edges = edge_index.size(1)
        src, dst = edge_index  # src[e] -> dst[e]

        if num_edges == 0:
            # No bonds: use only atom features
            m_v = torch.zeros(num_atoms, self.hidden_dim, device=x.device)
            h_v = F.relu(self.W_o(torch.cat([x, m_v], dim=-1)))
            return h_v

        # Build reverse-edge mapping:
        # For edge (u->v), find the edge (v->u)
        reverse_edge = torch.full((num_edges,), -1, dtype=torch.long, device=x.device)
        edge_dict = {}
        for e in range(num_edges):
            u, v = src[e].item(), dst[e].item()
            edge_dict.setdefault((u, v), []).append(e)

        for e in range(num_edges):
            u, v = src[e].item(), dst[e].item()
            rev_edges = edge_dict.get((v, u), [])
            if rev_edges:
                reverse_edge[e] = rev_edges[0]

        # Initialize edge hidden states: h_vw^0 = ReLU(W_i([x_v || e_vw]))
        x_src = x[src]  # [E, atom_dim]
        input_features = torch.cat([x_src, edge_attr], dim=-1)  # [E, atom_dim + bond_dim]
        h_edge = F.relu(self.W_i(input_features))  # [E, hidden_dim]
        h_edge_0 = h_edge.clone()

        # Message passing iterations
        for _ in range(self.depth - 1):
            # For each edge v->w, aggregate incoming edges u->v (excluding w->v)
            # m_vw = sum_{u in N(v) \ {w}} h_uv
            m_edge = torch.zeros_like(h_edge)

            # For each destination node v, collect all incoming edges u->v
            for e in range(num_edges):
                v = src[e].item()  # source of this edge is the "from" node
                # Find all edges whose dst is v (incoming to v)
                pass

            # Vectorized aggregation approach:
            # For each edge (v->w) at index e:
            #   Need sum of h_uv for all edges u->v, EXCLUDING w->v
            #   = (sum of all h[edges ending at v]) - h[edge w->v]
            # First compute sum of h for all edges ending at each node
            node_incoming_sum = torch.zeros(num_atoms, self.hidden_dim, device=x.device)
            node_incoming_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(h_edge), h_edge)

            # For edge e: v->w, the "sum of incoming to v" = node_incoming_sum[v]
            # We need to subtract h[reverse(e)] = h[w->v]
            m_edge = node_incoming_sum[src]  # sum of all edges ending at src[e]

            # Subtract the reverse edge's contribution
            has_reverse = (reverse_edge >= 0)
            if has_reverse.any():
                reverse_h = torch.zeros_like(h_edge)
                valid_rev = reverse_edge[has_reverse]
                reverse_h[has_reverse] = h_edge[valid_rev]
                m_edge = m_edge - reverse_h

            h_edge = F.relu(h_edge_0 + self.W_h(m_edge))
            h_edge = self.dropout(h_edge)

        # Final atom embeddings: aggregate incoming edge messages
        # m_v = sum_{w in N(v)} h_wv (all edges ending at v)
        m_v = torch.zeros(num_atoms, self.hidden_dim, device=x.device)
        m_v.scatter_add_(0, dst.unsqueeze(-1).expand_as(h_edge), h_edge)

        # h_v = ReLU(W_o([x_v || m_v]))
        h_v = F.relu(self.W_o(torch.cat([x, m_v], dim=-1)))
        h_v = self.dropout(h_v)

        return h_v


# ─── CheMeleon / Chemprop D-MPNN ─────────────────────────────────────────────


def _verify_chemeleon_checkpoint(path: str) -> None:
    """Verify the local CheMeleon checkpoint exists and has correct MD5."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"CheMeleon checkpoint not found at: {path}\n"
            f"Run the one-time download:\n"
            f"  python scripts/download_gnn_resources.py --resource chemeleon "
            f"--output {path}\n"
            f"Source: {CHEMELEON_URL}"
        )

    # Verify MD5
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    actual_md5 = md5.hexdigest()

    if actual_md5 != CHEMELEON_MD5:
        raise ValueError(
            f"CheMeleon checkpoint MD5 mismatch!\n"
            f"  Expected: {CHEMELEON_MD5}\n"
            f"  Actual:   {actual_md5}\n"
            f"  Path:     {path}\n"
            f"The file may be corrupted. Re-download with:\n"
            f"  python scripts/download_gnn_resources.py --resource chemeleon "
            f"--output {path} --force"
        )


class CheMeleonDMPNNEncoder(nn.Module):
    """CheMeleon-pretrained D-MPNN using native Chemprop architecture.

    This uses Chemprop's own BondMessagePassing and featurization,
    preserving the exact pretrained representation semantics.

    The native hidden width is preserved (NOT forced to 300).
    """

    is_sequence_capable = True

    def __init__(self, config, device: str = "cpu"):
        super().__init__()
        self.device_str = device
        self.frozen = getattr(config, "gnn_pretrained_frozen", True)

        checkpoint_path = getattr(config, "dmpnn_pretrained_checkpoint", CHEMELEON_DEFAULT_PATH)
        if checkpoint_path is None:
            checkpoint_path = CHEMELEON_DEFAULT_PATH

        # Verify the checkpoint
        _verify_chemeleon_checkpoint(checkpoint_path)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Try to import chemprop
        try:
            import chemprop
        except ImportError:
            raise ImportError(
                "chemprop is required for CheMeleon D-MPNN. "
                "Install it: pip install chemprop==2.3.1"
            )

        # Extract the message passing model from the checkpoint
        # CheMeleon checkpoints contain the BondMessagePassing state dict
        self._build_from_checkpoint(checkpoint, config)

        # Store provenance
        self._checkpoint_path = checkpoint_path
        self._checkpoint_md5 = CHEMELEON_MD5

        # Freeze if requested
        if self.frozen:
            for p in self.mp.parameters():
                p.requires_grad_(False)
            self.mp.eval()

        # Graph cache for Chemprop featurization (CPU-side)
        self._graph_cache: dict[str, object] = {}

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.mp.eval()

    def _build_from_checkpoint(self, checkpoint, config):
        """Build the Chemprop message-passing model from a CheMeleon checkpoint."""
        import chemprop
        from chemprop.nn.message_passing import BondMessagePassing
        from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

        # The CheMeleon checkpoint contains the model state dict
        # Detect the structure to determine the hidden size
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            # May be nested under various keys
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

        # Determine hidden size from the checkpoint weights
        # Look for the weight matrix key patterns in BondMessagePassing
        hidden_size = None
        depth = None

        # Try to infer from state dict keys
        for key, tensor in state_dict.items():
            if "W_h.weight" in key or "message_passing.W_h.weight" in key:
                hidden_size = tensor.shape[0]
                break
            if "W_i.weight" in key or "message_passing.W_i.weight" in key:
                hidden_size = tensor.shape[0]
                break

        if hidden_size is None:
            # Try common CheMeleon size (usually 300 or 1600)
            # Inspect any weight tensor to guess
            for key, tensor in state_dict.items():
                if tensor.ndim == 2 and tensor.shape[0] > 100:
                    hidden_size = tensor.shape[0]
                    break

        if hidden_size is None:
            hidden_size = 300  # fallback

        # Build the BondMessagePassing with matching architecture
        try:
            self.mp = BondMessagePassing(d_h=hidden_size)
        except TypeError:
            # Older chemprop API
            self.mp = BondMessagePassing(hidden_size=hidden_size)

        # Load the state dict
        # Try to match keys — CheMeleon may store with or without prefix
        try:
            self.mp.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # Try stripping common prefixes
            cleaned = {}
            for key, val in state_dict.items():
                # Remove common prefixes like "message_passing." or "model."
                new_key = key
                for prefix in ["message_passing.", "model.", "encoder.", "mp."]:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                cleaned[new_key] = val
            try:
                self.mp.load_state_dict(cleaned, strict=True)
            except RuntimeError:
                # Try non-strict loading as last resort
                missing, unexpected = self.mp.load_state_dict(cleaned, strict=False)
                if missing:
                    print(f"WARNING: CheMeleon checkpoint missing keys: {missing}")
                if unexpected:
                    print(f"WARNING: CheMeleon checkpoint unexpected keys: {unexpected}")

        self._output_dim = hidden_size
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _featurize_smiles(self, smiles: str):
        """Convert SMILES to Chemprop's native graph representation."""
        if smiles in self._graph_cache:
            return self._graph_cache[smiles]

        from chemprop.data import MoleculeDatapoint

        dp = MoleculeDatapoint.from_smi(smiles)
        mg = self.featurizer(dp.mol)
        self._graph_cache[smiles] = mg
        return mg

    def encode(self, smiles_batch: list[str]):
        """Encode a batch of SMILES using Chemprop's native featurization.

        Returns:
            token_embeddings: [B, L_max, D]
            pooled_embedding: [B, D]
            token_mask:       [B, L_max] bool, True = padding
        """
        if self.frozen:
            self.mp.eval()

        from chemprop.data import (
            MoleculeDatapoint, MoleculeDataset, build_dataloader,
            BatchMolGraph,
        )

        device = next(self.parameters()).device
        B = len(smiles_batch)
        D = self._output_dim

        # Handle empty strings
        valid_indices = []
        empty_indices = []
        datapoints = []

        for i, smi in enumerate(smiles_batch):
            if smi == "":
                empty_indices.append(i)
            else:
                valid_indices.append(i)
                datapoints.append(MoleculeDatapoint.from_smi(smi))

        all_node_embeds = []
        all_pooled = []
        max_atoms = 1

        if datapoints:
            # Use Chemprop's native batching
            dataset = MoleculeDataset(datapoints)

            # Build batch mol graph
            mol_graphs = [self.featurizer(dp.mol) for dp in datapoints]
            bmg = BatchMolGraph(mol_graphs)
            bmg.to(device)

            # Run message passing
            with torch.set_grad_enabled(not self.frozen and self.training):
                encoding = self.mp(bmg)

            # Check if we get atom-level or graph-level output
            if encoding.dim() == 2 and encoding.shape[0] == len(datapoints):
                # Graph-level output — we need to get node-level
                # Try to get node representations from the MP layer
                # For cross-attention we need per-atom embeddings
                # Use the graph-level as both pooled and single-token representation
                for idx, dp in enumerate(datapoints):
                    num_atoms = dp.mol.GetNumAtoms()
                    # Repeat the graph embedding for each atom as a fallback
                    node_emb = encoding[idx].unsqueeze(0).expand(num_atoms, -1)
                    all_node_embeds.append(node_emb)
                    all_pooled.append(encoding[idx])
                    max_atoms = max(max_atoms, num_atoms)
            else:
                # Node-level output
                # Parse node counts from mol graphs
                offset = 0
                for idx, dp in enumerate(datapoints):
                    num_atoms = dp.mol.GetNumAtoms()
                    node_emb = encoding[offset:offset + num_atoms]
                    all_node_embeds.append(node_emb)
                    all_pooled.append(node_emb.mean(dim=0))
                    max_atoms = max(max_atoms, num_atoms)
                    offset += num_atoms

        # Assemble full batch
        L_max = max(max_atoms, 1)
        token_embeddings = torch.zeros(B, L_max, D, device=device)
        token_mask = torch.ones(B, L_max, dtype=torch.bool, device=device)
        pooled_embedding = torch.zeros(B, D, device=device)

        for out_idx, batch_idx in enumerate(valid_indices):
            node_emb = all_node_embeds[out_idx]
            n_atoms = node_emb.size(0)
            token_embeddings[batch_idx, :n_atoms] = node_emb
            token_mask[batch_idx, :n_atoms] = False
            pooled_embedding[batch_idx] = all_pooled[out_idx]

        return token_embeddings, pooled_embedding, token_mask


# ─── Scratch D-MPNN Encoder ──────────────────────────────────────────────────


class ScratchDMPNNEncoder(PyGGNNEncoderBase):
    """Scratch D-MPNN using the shared 39/10 PyG feature schema.

    config.encoder = "dmpnn_scratch"
    This is a diagnostic/ablation encoder, NOT the primary D-MPNN.
    """

    def __init__(self, config, device: str = "cpu"):
        hidden_dim = getattr(config, "gnn_hidden_dim", 300)
        super().__init__(output_dim=hidden_dim)

        self.gnn = ScratchDMPNNNetwork(
            atom_dim=getattr(config, "gnn_node_feature_dim", ATOM_FEATURE_DIM),
            bond_dim=getattr(config, "gnn_edge_feature_dim", BOND_FEATURE_DIM),
            hidden_dim=hidden_dim,
            depth=getattr(config, "dmpnn_depth", 3),
            dropout=getattr(config, "gnn_dropout", 0.10),
            bias=getattr(config, "dmpnn_bias", False),
        )

    def _run_gnn(self, batch) -> torch.Tensor:
        return self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


# ─── Unified D-MPNN factory ──────────────────────────────────────────────────


class DMPNNEncoder(nn.Module):
    """Factory wrapper that dispatches to CheMeleon or Scratch D-MPNN.

    config.encoder = "dmpnn_chemprop" -> CheMeleon (default)
    config.encoder = "dmpnn_scratch"  -> Scratch

    For "dmpnn_chemprop":
        - Loads local CheMeleon checkpoint
        - Uses native Chemprop featurization
        - Preserves native hidden width
        - Never downloads during training/inference

    For "dmpnn_scratch":
        - Uses shared 39/10 PyG feature schema
        - Hidden dim = 300
    """

    is_sequence_capable = True

    def __init__(self, config, device: str = "cpu"):
        super().__init__()

        use_pretrained = getattr(config, "dmpnn_use_pretrained", True)
        encoder_name = getattr(config, "encoder", "dmpnn_chemprop")

        if encoder_name == "dmpnn_scratch" or not use_pretrained:
            self._impl = ScratchDMPNNEncoder(config, device)
        else:
            # CheMeleon pretrained
            try:
                self._impl = CheMeleonDMPNNEncoder(config, device)
            except (ImportError, FileNotFoundError, ValueError) as e:
                # Do NOT silently fall back to scratch
                raise RuntimeError(
                    f"Failed to load CheMeleon pretrained D-MPNN: {e}\n"
                    f"If you intended to use scratch D-MPNN, set "
                    f"--encoder dmpnn_scratch instead.\n"
                    f"DO NOT silently substitute scratch for CheMeleon."
                ) from e

    @property
    def output_dim(self) -> int:
        return self._impl.output_dim

    def encode(self, smiles_batch: list[str]):
        return self._impl.encode(smiles_batch)

    def forward(self, *args, **kwargs):
        return self._impl(*args, **kwargs)
