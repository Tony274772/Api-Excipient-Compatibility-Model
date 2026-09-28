"""GATv2 encoder — dynamic multi-head attention with edge features.

Follows Section 11 of the GNN specification.
Uses PyG GATv2Conv with edge_dim, self-loops, and residual connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_geometric.nn import GATv2Conv

from src.encoders.gnn_common import PyGGNNEncoderBase, ATOM_FEATURE_DIM, BOND_FEATURE_DIM


class GATv2Network(nn.Module):
    """GATv2 message-passing network with multi-head attention.

    Each layer:
        GATv2Conv (edge-aware, multi-head, residual, self-loops)
        ReLU
        LayerNorm
    """

    def __init__(
        self,
        in_channels: int = ATOM_FEATURE_DIM,
        hidden_channels: int = 300,
        num_layers: int = 3,
        heads: int = 4,
        edge_dim: int = BOND_FEATURE_DIM,
        dropout: float = 0.10,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        fill_value: str = "mean",
        residual: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout

        assert hidden_channels % heads == 0, (
            f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
        )
        out_per_head = hidden_channels // heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels

            self.convs.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_per_head,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=add_self_loops,
                    fill_value=fill_value,
                    residual=(residual and i > 0),  # residual only after first layer (dim mismatch)
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

    def forward(self, x, edge_index, edge_attr, batch=None):
        """Run GATv2 message passing.

        Args:
            x:          [N, in_channels] atom features
            edge_index: [2, E] directed edges
            edge_attr:  [E, edge_dim] bond features
            batch:      [N] batch assignment (unused, for API compat)

        Returns:
            node_embeddings: [N, hidden_channels]
        """
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x)
            x = norm(x)
        return x


class GATv2Encoder(PyGGNNEncoderBase):
    """GATv2 GNN encoder for the compatibility model.

    config.encoder = "gatv2"

    Hyperparameters (from config):
        hidden_channels  = gnn_hidden_dim (300)
        num_layers       = gnn_num_layers (3)
        heads            = gatv2_heads (4)
        dropout          = gnn_dropout (0.10)
        negative_slope   = gatv2_negative_slope (0.2)
    """

    def __init__(self, config, device: str = "cpu"):
        hidden_dim = getattr(config, "gnn_hidden_dim", 300)
        super().__init__(output_dim=hidden_dim)

        self.gnn = GATv2Network(
            in_channels=getattr(config, "gnn_node_feature_dim", ATOM_FEATURE_DIM),
            hidden_channels=hidden_dim,
            num_layers=getattr(config, "gnn_num_layers", 3),
            heads=getattr(config, "gatv2_heads", 4),
            edge_dim=getattr(config, "gnn_edge_feature_dim", BOND_FEATURE_DIM),
            dropout=getattr(config, "gnn_dropout", 0.10),
            negative_slope=getattr(config, "gatv2_negative_slope", 0.2),
        )

    def _run_gnn(self, batch) -> torch.Tensor:
        return self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
