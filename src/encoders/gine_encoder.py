"""GINE encoder — edge-aware GIN variant.

Follows Section 10 of the GNN specification.
Uses PyG GINEConv with 2-layer MLPs, LayerNorm, ReLU, and dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_geometric.nn import GINEConv

from src.encoders.gnn_common import PyGGNNEncoderBase, ATOM_FEATURE_DIM, BOND_FEATURE_DIM


class GINENetwork(nn.Module):
    """GINE message-passing network with per-layer MLPs.

    Each layer:
        GINEConv(mlp, eps, train_eps, edge_dim)
        LayerNorm
        ReLU
        Dropout
    """

    def __init__(
        self,
        in_channels: int = ATOM_FEATURE_DIM,
        hidden_channels: int = 300,
        num_layers: int = 3,
        edge_dim: int = BOND_FEATURE_DIM,
        dropout: float = 0.10,
        train_eps: bool = True,
        initial_eps: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels

            # 2-layer MLP as specified
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )

            self.convs.append(
                GINEConv(
                    nn=mlp,
                    eps=initial_eps,
                    train_eps=train_eps,
                    edge_dim=edge_dim,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

    def forward(self, x, edge_index, edge_attr, batch=None):
        """Run GINE message passing.

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
            x = norm(x)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


class GINEEncoder(PyGGNNEncoderBase):
    """GINE GNN encoder for the compatibility model.

    config.encoder = "gine"

    Hyperparameters (from config):
        hidden_channels = gnn_hidden_dim (300)
        num_layers      = gnn_num_layers (3)
        dropout         = gnn_dropout (0.10)
        train_eps       = gine_train_eps (True)
        initial_eps     = gine_eps (0.0)
    """

    def __init__(self, config, device: str = "cpu"):
        hidden_dim = getattr(config, "gnn_hidden_dim", 300)
        super().__init__(output_dim=hidden_dim)

        self.gnn = GINENetwork(
            in_channels=getattr(config, "gnn_node_feature_dim", ATOM_FEATURE_DIM),
            hidden_channels=hidden_dim,
            num_layers=getattr(config, "gnn_num_layers", 3),
            edge_dim=getattr(config, "gnn_edge_feature_dim", BOND_FEATURE_DIM),
            dropout=getattr(config, "gnn_dropout", 0.10),
            train_eps=getattr(config, "gine_train_eps", True),
            initial_eps=getattr(config, "gine_eps", 0.0),
        )

    def _run_gnn(self, batch) -> torch.Tensor:
        return self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
