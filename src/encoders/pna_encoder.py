"""PNA encoder — Principal Neighbourhood Aggregation.

Follows Section 12 of the GNN specification.
Uses PyG PNAConv with multiple aggregators and scalers,
requiring a training-derived degree histogram.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_geometric.nn import PNAConv

from src.encoders.gnn_common import PyGGNNEncoderBase, ATOM_FEATURE_DIM, BOND_FEATURE_DIM


class PNANetwork(nn.Module):
    """PNA message-passing network with degree-aware aggregation.

    Each layer:
        PNAConv (multi-aggregator, multi-scaler, degree-normalised)
        Residual (where dimensions match)
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
        deg: torch.Tensor = None,
        aggregators: list[str] = None,
        scalers: list[str] = None,
        towers: int = 5,
        pre_layers: int = 1,
        post_layers: int = 1,
        divide_input: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        if aggregators is None:
            aggregators = ["mean", "min", "max", "std"]
        if scalers is None:
            scalers = ["identity", "amplification", "attenuation"]
        if deg is None:
            raise ValueError(
                "PNA requires a degree histogram (deg). "
                "Compute it from training data before constructing PNAEncoder."
            )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels

            self.convs.append(
                PNAConv(
                    in_channels=in_dim,
                    out_channels=hidden_channels,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg,
                    edge_dim=edge_dim,
                    towers=towers,
                    pre_layers=pre_layers,
                    post_layers=post_layers,
                    divide_input=divide_input,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Linear projection for residual skip when input dim != hidden dim
        self.lin_skip = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def forward(self, x, edge_index, edge_attr, batch=None):
        """Run PNA message passing.

        Args:
            x:          [N, in_channels] atom features
            edge_index: [2, E] directed edges
            edge_attr:  [E, edge_dim] bond features
            batch:      [N] batch assignment (unused, for API compat)

        Returns:
            node_embeddings: [N, hidden_channels]
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_in = x
            x = conv(x, edge_index, edge_attr)

            # Residual skip where dimensions match
            if i > 0:
                x = x + x_in
            elif self.lin_skip is not None and i == 0:
                # First layer: project input for residual
                pass  # skip residual on first layer (dim mismatch)

            x = norm(x)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        return x


class PNAEncoder(PyGGNNEncoderBase):
    """PNA GNN encoder for the compatibility model.

    config.encoder = "pna"

    IMPORTANT: config.pna_degree_hist must be set before constructing this encoder.
    It should be computed from training-split molecules only.

    Hyperparameters (from config):
        hidden_channels = gnn_hidden_dim (300)
        num_layers      = gnn_num_layers (3)
        dropout         = gnn_dropout (0.10)
        towers          = pna_towers (5)
        aggregators     = pna_aggregators (mean, min, max, std)
        scalers         = pna_scalers (identity, amplification, attenuation)
    """

    def __init__(self, config, device: str = "cpu"):
        hidden_dim = getattr(config, "gnn_hidden_dim", 300)
        super().__init__(output_dim=hidden_dim)

        # Get degree histogram
        deg_hist = getattr(config, "pna_degree_hist", None)
        if deg_hist is None:
            raise ValueError(
                "PNA encoder requires config.pna_degree_hist to be set. "
                "Compute it from training SMILES before constructing PNAEncoder. "
                "Use: from src.encoders.pna_utils import compute_degree_histogram"
            )

        if isinstance(deg_hist, list):
            deg_hist = torch.tensor(deg_hist, dtype=torch.long)
        elif isinstance(deg_hist, torch.Tensor):
            deg_hist = deg_hist.long()

        aggregators = list(getattr(config, "pna_aggregators", ("mean", "min", "max", "std")))
        scalers = list(getattr(config, "pna_scalers", ("identity", "amplification", "attenuation")))

        self.gnn = PNANetwork(
            in_channels=getattr(config, "gnn_node_feature_dim", ATOM_FEATURE_DIM),
            hidden_channels=hidden_dim,
            num_layers=getattr(config, "gnn_num_layers", 3),
            edge_dim=getattr(config, "gnn_edge_feature_dim", BOND_FEATURE_DIM),
            dropout=getattr(config, "gnn_dropout", 0.10),
            deg=deg_hist,
            aggregators=aggregators,
            scalers=scalers,
            towers=getattr(config, "pna_towers", 5),
            pre_layers=getattr(config, "pna_pre_layers", 1),
            post_layers=getattr(config, "pna_post_layers", 1),
            divide_input=getattr(config, "pna_divide_input", False),
        )

    def _run_gnn(self, batch) -> torch.Tensor:
        return self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
