"""Shared molecular graph featurization and GNN encoder base class.

Provides:
- 39-dimensional atom features
- 10-dimensional bond features
- Directed edges (u->v, v->u) for each bond
- PyG Data construction with graph caching
- Dense batch padding with correct mask semantics (True = padding)
- Mean pooling
- Missing/empty SMILES handling
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from rdkit import Chem

# Lazy imports for PyG — avoid breaking non-GNN environments
_torch_geometric_available: Optional[bool] = None


def _check_pyg():
    global _torch_geometric_available
    if _torch_geometric_available is None:
        try:
            import torch_geometric  # noqa: F401
            _torch_geometric_available = True
        except ImportError:
            _torch_geometric_available = False
    if not _torch_geometric_available:
        raise ImportError(
            "torch_geometric is required for GNN encoders. "
            "Install it: pip install torch-geometric"
        )


# ─── Atom feature constants ───────────────────────────────────────────────────

ELEMENT_LIST = [
    "B", "C", "N", "O", "F", "Si", "P", "S",
    "Cl", "As", "Se", "Br", "Te", "I", "At", "other",
]

DEGREE_LIST = [0, 1, 2, 3, 4, 5]

HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    "other",
]

TOTAL_H_LIST = [0, 1, 2, 3, 4]

STEREO_LIST = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]

ATOM_FEATURE_DIM = 39
BOND_FEATURE_DIM = 10


# ─── Feature functions ────────────────────────────────────────────────────────


def _one_hot(value, allowable_set: list, allow_other: bool = True) -> list[float]:
    """One-hot encode a value. Unknown values go to the last bucket if allow_other."""
    encoding = [0.0] * len(allowable_set)
    if value in allowable_set:
        encoding[allowable_set.index(value)] = 1.0
    elif allow_other:
        encoding[-1] = 1.0
    return encoding


def atom_features(atom: Chem.rdchem.Atom) -> list[float]:
    """Compute 39-dimensional atom feature vector.

    Layout:
        element symbol one-hot:  16 dims
        atom degree one-hot:      6 dims
        formal charge:            1 dim
        radical electrons:        1 dim
        hybridization one-hot:    6 dims
        aromatic flag:            1 dim
        total H one-hot:          5 dims
        chirality possible:       1 dim
        R/S chirality one-hot:    2 dims
        Total:                   39 dims
    """
    symbol = atom.GetSymbol()
    features = _one_hot(symbol, ELEMENT_LIST, allow_other=True)  # 16

    degree = atom.GetTotalDegree()
    if degree > 5:
        degree = 5  # clamp to last valid bucket
    features += _one_hot(degree, DEGREE_LIST, allow_other=False)  # 6

    features.append(float(atom.GetFormalCharge()))  # 1
    features.append(float(atom.GetNumRadicalElectrons()))  # 1

    hybridization = atom.GetHybridization()
    features += _one_hot(hybridization, HYBRIDIZATION_LIST, allow_other=True)  # 6

    features.append(1.0 if atom.GetIsAromatic() else 0.0)  # 1

    total_h = atom.GetTotalNumHs()
    if total_h > 4:
        total_h = 4  # clamp
    features += _one_hot(total_h, TOTAL_H_LIST, allow_other=False)  # 5

    # Chirality
    has_chiral = 1.0 if atom.HasProp("_ChiralityPossible") else 0.0
    features.append(has_chiral)  # 1

    try:
        cip = atom.GetProp("_CIPCode")
        features.append(1.0 if cip == "R" else 0.0)
        features.append(1.0 if cip == "S" else 0.0)
    except KeyError:
        features.extend([0.0, 0.0])  # 2

    assert len(features) == ATOM_FEATURE_DIM, (
        f"Expected {ATOM_FEATURE_DIM} atom features, got {len(features)}"
    )
    return features


def bond_features(bond: Chem.rdchem.Bond) -> list[float]:
    """Compute 10-dimensional bond feature vector.

    Layout:
        single bond:     1 dim
        double bond:     1 dim
        triple bond:     1 dim
        aromatic bond:   1 dim
        conjugated flag: 1 dim
        ring flag:       1 dim
        stereo one-hot:  4 dims
        Total:          10 dims
    """
    bt = bond.GetBondType()
    features = [
        1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0,
    ]
    features += _one_hot(bond.GetStereo(), STEREO_LIST, allow_other=False)  # 4

    assert len(features) == BOND_FEATURE_DIM, (
        f"Expected {BOND_FEATURE_DIM} bond features, got {len(features)}"
    )
    return features


def smiles_to_pyg_data(smiles: str):
    """Convert a SMILES string to a PyG Data object.

    Returns a Data with:
        x:          [N, 39]  atom features
        edge_index: [2, E]   directed edges (u->v, v->u for each bond)
        edge_attr:  [E, 10]  bond features

    For molecules with no bonds:
        edge_index: [2, 0]
        edge_attr:  [0, 10]

    Multi-fragment molecules produce a single graph with disconnected components.

    Raises ValueError for unparseable SMILES.
    """
    _check_pyg()
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            f"RDKit cannot parse SMILES for GNN encoder: {smiles!r}"
        )

    # Assign stereochemistry info for chirality features
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # Atom features
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))
    x = torch.tensor(atom_feats, dtype=torch.float32)

    # Bond features (directed: u->v and v->u)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # u -> v
        edge_indices.append([u, v])
        edge_attrs.append(bf)
        # v -> u
        edge_indices.append([v, u])
        edge_attrs.append(bf)

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, BOND_FEATURE_DIM, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class PyGGNNEncoderBase(nn.Module):
    """Base class for all PyG-based GNN encoders.

    Handles:
    - Graph caching (CPU-side, keyed by SMILES)
    - Missing/empty SMILES handling
    - Batch construction via PyG Batch
    - Dense padding with to_dense_batch
    - Mean pooling
    - Correct mask semantics (True = padding)
    - Batch ordering preservation

    Subclasses must implement:
        _run_gnn(batch) -> node_embeddings [N_total, D]
    """

    is_sequence_capable = True

    def __init__(self, output_dim: int = 300):
        super().__init__()
        self._output_dim = output_dim
        self._graph_cache: dict[str, object] = {}  # SMILES -> PyG Data (CPU)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _get_graph(self, smiles: str):
        """Get or create a cached PyG Data object for a SMILES string."""
        if smiles not in self._graph_cache:
            data = smiles_to_pyg_data(smiles)
            self._graph_cache[smiles] = data
        return self._graph_cache[smiles]

    def _run_gnn(self, batch) -> torch.Tensor:
        """Run the GNN on a PyG Batch and return node embeddings [N_total, D].

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def encode(self, smiles_batch: list[str]):
        """Encode a batch of SMILES strings.

        Returns:
            token_embeddings: [B, L_max, D] padded node embeddings
            pooled_embedding: [B, D] mean-pooled graph embeddings
            token_mask:       [B, L_max] bool, True = padding/invalid
        """
        _check_pyg()
        from torch_geometric.data import Batch, Data
        from torch_geometric.utils import to_dense_batch
        from torch_geometric.nn import global_mean_pool

        B = len(smiles_batch)
        D = self._output_dim

        # Determine device from model parameters
        device = next(self.parameters()).device

        # Separate valid vs empty SMILES
        valid_indices = []
        empty_indices = []
        graph_list = []

        for i, smi in enumerate(smiles_batch):
            if smi == "":
                empty_indices.append(i)
            else:
                valid_indices.append(i)
                g = self._get_graph(smi)
                graph_list.append(g)

        if len(graph_list) > 0:
            # Batch the valid graphs
            pyg_batch = Batch.from_data_list(graph_list)
            pyg_batch = pyg_batch.to(device)

            # Run GNN to get node embeddings
            node_embeddings = self._run_gnn(pyg_batch)  # [N_total, D]

            # Dense padding: to_dense_batch returns (dense, valid_mask)
            # valid_mask: True = real node, False = padding
            dense_nodes, valid_mask = to_dense_batch(
                node_embeddings, pyg_batch.batch
            )  # [B_valid, L_max_valid, D], [B_valid, L_max_valid]

            # Mean pooling over valid nodes
            pooled_valid = global_mean_pool(
                node_embeddings, pyg_batch.batch
            )  # [B_valid, D]

            # Project token_mask: True = padding (inverted from to_dense_batch)
            token_mask_valid = ~valid_mask  # [B_valid, L_max_valid]
        else:
            dense_nodes = torch.zeros(0, 1, D, device=device)
            token_mask_valid = torch.ones(0, 1, dtype=torch.bool, device=device)
            pooled_valid = torch.zeros(0, D, device=device)

        # Now assemble the full batch (respecting original order)
        L_max_valid = dense_nodes.shape[1] if dense_nodes.shape[0] > 0 else 1
        L_max = max(L_max_valid, 1)  # at least 1 for empty SMILES

        token_embeddings = torch.zeros(B, L_max, D, device=device)
        token_mask = torch.ones(B, L_max, dtype=torch.bool, device=device)
        pooled_embedding = torch.zeros(B, D, device=device)

        # Fill valid entries in original order
        for out_idx, batch_idx in enumerate(valid_indices):
            n_tokens = dense_nodes.shape[1]
            token_embeddings[batch_idx, :n_tokens] = dense_nodes[out_idx]
            token_mask[batch_idx, :n_tokens] = token_mask_valid[out_idx]
            pooled_embedding[batch_idx] = pooled_valid[out_idx]

        # Empty entries remain zeros with all-True mask (already initialized)

        return token_embeddings, pooled_embedding, token_mask
