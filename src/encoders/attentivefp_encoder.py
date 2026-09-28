"""AttentiveFP encoder — node-returning implementation.

Uses the AttentiveFP message-passing architecture from PyG but exposes
node-level hidden representations (not just graph-level readout) so that
the existing cross-attention path can treat atoms as tokens.

Follows Section 9 of the GNN specification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.gnn_common import PyGGNNEncoderBase, ATOM_FEATURE_DIM, BOND_FEATURE_DIM


class _GATEConv(nn.Module):
    """Edge-aware gated graph convolution used in AttentiveFP.

    Adapted from the PyG AttentiveFP source to preserve the
    characteristic gated recurrent node-update mechanism.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout

        self.lin_src = nn.Linear(in_channels, out_channels)
        self.lin_dst = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_dim, out_channels)
        self.att = nn.Parameter(torch.empty(1, out_channels))
        nn.init.xavier_uniform_(self.att.unsqueeze(0))

    def forward(self, x, edge_index, edge_attr):
        """Returns updated node features and attention weights."""
        src, dst = edge_index

        # Compute attention coefficients
        x_src = self.lin_src(x[src])
        x_dst = self.lin_dst(x[dst])
        e = self.lin_edge(edge_attr)

        alpha = (x_src + x_dst + e) * self.att
        alpha = alpha.sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        # Softmax per destination node
        from torch_geometric.utils import softmax
        alpha = softmax(alpha, dst, num_nodes=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Aggregate
        msg = x_src * alpha.unsqueeze(-1)

        from torch_geometric.utils import scatter
        out = scatter(msg, dst, dim=0, dim_size=x.size(0), reduce='sum')

        return out


class AttentiveFPNodeEncoder(nn.Module):
    """AttentiveFP message-passing that returns node-level representations.

    Architecture follows the original paper and PyG implementation:
    1. Initial atom embedding projection
    2. Edge-aware gated graph convolution layers with GRU recurrence
    3. Additional atom-level attention timesteps

    The graph-level readout is NOT applied — this returns atom embeddings.
    """

    def __init__(
        self,
        in_channels: int = ATOM_FEATURE_DIM,
        hidden_channels: int = 300,
        out_channels: int = 300,
        edge_dim: int = BOND_FEATURE_DIM,
        num_layers: int = 2,
        num_timesteps: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        # Initial node projection
        self.lin_initial = nn.Linear(in_channels, hidden_channels)

        # Edge-aware gated convolution layers
        self.convs = nn.ModuleList()
        self.grus = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                _GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
            )
            self.grus.append(nn.GRUCell(hidden_channels, hidden_channels))

        # Final projection if needed
        if hidden_channels != out_channels:
            self.lin_out = nn.Linear(hidden_channels, out_channels)
        else:
            self.lin_out = None

    def forward(self, x, edge_index, edge_attr, batch=None):
        """Run AttentiveFP message passing.

        Args:
            x:          [N, in_channels] atom features
            edge_index: [2, E] directed edge indices
            edge_attr:  [E, edge_dim] bond features
            batch:      [N] batch assignment (unused here, for API compat)

        Returns:
            node_embeddings: [N, out_channels]
        """
        # Initial projection
        x = F.leaky_relu(self.lin_initial(x), negative_slope=0.2)

        # Message passing layers with GRU
        for conv, gru in zip(self.convs, self.grus):
            h = F.relu(conv(x, edge_index, edge_attr))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x)

        # Additional timestep refinement using the last layer
        for _ in range(self.num_timesteps - 1):
            h = F.relu(self.convs[-1](x, edge_index, edge_attr))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = self.grus[-1](h, x)

        if self.lin_out is not None:
            x = self.lin_out(x)

        return x


class AttentiveFPEncoder(PyGGNNEncoderBase):
    """AttentiveFP GNN encoder for the compatibility model.

    config.encoder = "attentivefp"

    Hyperparameters (from config):
        hidden_channels = gnn_hidden_dim (300)
        num_layers      = attentivefp_num_layers (2)
        num_timesteps   = attentivefp_num_timesteps (2)
        dropout         = gnn_dropout (0.10)
    """

    def __init__(self, config, device: str = "cpu"):
        hidden_dim = getattr(config, "gnn_hidden_dim", 300)
        super().__init__(output_dim=hidden_dim)

        self.gnn = AttentiveFPNodeEncoder(
            in_channels=getattr(config, "gnn_node_feature_dim", ATOM_FEATURE_DIM),
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=getattr(config, "gnn_edge_feature_dim", BOND_FEATURE_DIM),
            num_layers=getattr(config, "attentivefp_num_layers", 2),
            num_timesteps=getattr(config, "attentivefp_num_timesteps", 2),
            dropout=getattr(config, "gnn_dropout", 0.10),
        )

    def _run_gnn(self, batch) -> torch.Tensor:
        return self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
