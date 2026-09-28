import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import Data, Batch

# ─── Stanford Feature Schema (from loader.py) ───────────────────────────────

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

num_atom_type = 120
num_chirality_tag = 3
num_bond_type = 6
num_bond_direction = 3


def smiles_to_stanford_data(smiles: str) -> Data:
    """Convert SMILES to PyG Data object using the Stanford chemistry schema."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [
            allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(atom.GetChiralTag())
        ]
        atom_features_list.append(atom_feature)
    
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    if x.ndim == 1:
        x = x.unsqueeze(0)  # Handle empty molecule case if any

    # bonds
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                allowable_features['possible_bonds'].index(bond.GetBondType()),
                allowable_features['possible_bond_dirs'].index(bond.GetBondDir())
            ]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ─── Stanford Architecture (from model.py) ──────────────────────────────────

class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__(node_dim=0)
        self.aggr = aggr
        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, size=None)

    def message(self, edge_index_i, edge_index_j, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j = x_j + edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias
        return aggr_out


class StanfordGNN(torch.nn.Module):
    def __init__(self, num_layer=5, emb_dim=300, JK="last", drop_ratio=0):
        super(StanfordGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.gnns.append(GATConv(emb_dim))

        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr):
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


# ─── Project Adapter ────────────────────────────────────────────────────────

class StanfordPretrainedGATEncoder(nn.Module):
    is_sequence_capable = True

    def __init__(self, config, device="cpu"):
        super().__init__()
        self.device = device
        self.output_dim = 300
        self.frozen = getattr(config, "gnn_pretrained_frozen", True)
        
        # Determine checkpoint path
        ckpt_path = getattr(config, "stanford_gat_checkpoint_path", "models/pretrained/stanford_gat/gat_contextpred.pth")
        
        # Fallback offline check
        is_offline = getattr(config, "gnn_offline", False)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"\nRequired local pretrained model not found: {ckpt_path}\n"
                f"Run the one-time GNN resource download/setup command:\n"
                f"  python scripts/download_gnn_resources.py --resource stanford_gat\n"
            )

        # Build architecture (using default 5 layers, 300 dim, last JK based on checkpoint)
        self.gnn = StanfordGNN(num_layer=5, emb_dim=300, JK="last", drop_ratio=0.0)

        # Load weights
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # In the snap repo, the model was often saved as a GNN_graphpred which has a self.gnn
        # If the keys have "gnn.", strip it.
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("gnn."):
                new_state_dict[k[4:]] = v
            else:
                new_state_dict[k] = v

        try:
            self.gnn.load_state_dict(new_state_dict, strict=True)
        except RuntimeError as e:
            # Maybe the checkpoint is just the GNN
            try:
                self.gnn.load_state_dict(state_dict, strict=True)
            except RuntimeError as e2:
                raise RuntimeError(f"Stanford GAT checkpoint keys mismatch: {e2}")

        self.gnn.to(self.device)

        if self.frozen:
            for p in self.gnn.parameters():
                p.requires_grad_(False)
            self.gnn.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            # Force eval mode for frozen modules (e.g. BatchNorm)
            self.gnn.eval()

    def encode(self, smiles_batch: list[str]):
        if self.frozen:
            self.gnn.eval()

        data_list = []
        valid_indices = []
        for i, s in enumerate(smiles_batch):
            if not s:
                continue
            try:
                d = smiles_to_stanford_data(s)
                data_list.append(d)
                valid_indices.append(i)
            except Exception:
                continue
        
        B = len(smiles_batch)
        if not data_list:
            return (
                torch.zeros(B, 1, self.output_dim, device=self.device),
                torch.zeros(B, self.output_dim, device=self.device),
                torch.ones(B, 1, dtype=torch.bool, device=self.device)
            )

        batch = Batch.from_data_list(data_list).to(self.device)
        
        # Ensure we don't compute gradients if frozen
        with torch.set_grad_enabled(not self.frozen and self.training):
            node_embeds = self.gnn(batch.x, batch.edge_index, batch.edge_attr)

        # Map back to dense batch
        from torch_geometric.utils import to_dense_batch
        dense_x, valid_mask = to_dense_batch(node_embeds, batch.batch)
        
        L_max = dense_x.shape[1]
        
        token_embeddings = torch.zeros(B, L_max, self.output_dim, device=self.device)
        pooled_embedding = torch.zeros(B, self.output_dim, device=self.device)
        token_mask = torch.ones(B, L_max, dtype=torch.bool, device=self.device)

        for dense_idx, batch_idx in enumerate(valid_indices):
            token_embeddings[batch_idx] = dense_x[dense_idx]
            token_mask[batch_idx] = ~valid_mask[dense_idx]  # Invert to True=padding
            
            # Mean pool over valid atoms
            real_atoms = valid_mask[dense_idx].sum().item()
            if real_atoms > 0:
                pooled_embedding[batch_idx] = dense_x[dense_idx, :real_atoms].mean(dim=0)

        return token_embeddings, pooled_embedding, token_mask
