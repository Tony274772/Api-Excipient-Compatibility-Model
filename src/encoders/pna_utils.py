"""PNA degree histogram utility.

Computes the in-degree histogram from training-split SMILES only.
Must be recomputed independently for each CV fold.
"""

from __future__ import annotations

import torch

from src.encoders.gnn_common import smiles_to_pyg_data


def compute_degree_histogram(smiles_iterable) -> torch.Tensor:
    """Compute the in-degree histogram from a collection of SMILES strings.

    This histogram is required by PNAConv for its degree scalers.

    IMPORTANT: Only pass training-split SMILES. Never include validation or test.

    Args:
        smiles_iterable: Iterable of SMILES strings (training molecules only)

    Returns:
        degree_histogram: 1-D LongTensor where hist[d] = count of nodes with in-degree d
    """
    all_degrees = []

    for smiles in smiles_iterable:
        if not smiles or smiles.strip() == "":
            continue

        try:
            data = smiles_to_pyg_data(smiles)
        except ValueError:
            continue

        if data.edge_index.numel() == 0:
            # Single-atom molecule with no bonds: all nodes have degree 0
            all_degrees.append(torch.zeros(data.x.size(0), dtype=torch.long))
        else:
            # Count in-degrees from edge_index[1] (destination nodes)
            dst = data.edge_index[1]
            degrees = torch.zeros(data.x.size(0), dtype=torch.long)
            for d in dst:
                degrees[d.item()] += 1
            all_degrees.append(degrees)

    if not all_degrees:
        return torch.tensor([1], dtype=torch.long)

    all_degrees = torch.cat(all_degrees)
    max_degree = int(all_degrees.max().item())
    histogram = torch.bincount(all_degrees, minlength=max_degree + 1)

    # Ensure at least one element
    if histogram.numel() == 0:
        return torch.tensor([1], dtype=torch.long)

    return histogram.long()


def degree_histogram_from_dataframe(df, smiles_column: str = "API_Smiles") -> torch.Tensor:
    """Convenience: compute degree histogram from a pandas DataFrame.

    Collects unique SMILES from the specified column. Optionally also
    includes Excipient_Smiles if present.

    Args:
        df: pandas DataFrame with SMILES columns
        smiles_column: Primary SMILES column (default: API_Smiles)

    Returns:
        degree_histogram: 1-D LongTensor
    """
    smiles_set = set()

    if smiles_column in df.columns:
        smiles_set.update(df[smiles_column].dropna().unique().tolist())

    # Also include excipient SMILES if available
    if "Excipient_Smiles" in df.columns:
        smiles_set.update(
            s for s in df["Excipient_Smiles"].dropna().unique().tolist()
            if s and str(s).strip()
        )

    return compute_degree_histogram(smiles_set)
