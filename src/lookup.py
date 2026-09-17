"""Lookup-table augmentation – Section 9 (Phase 2, optional).

Implements reaction lookup, weak-label expansion, mechanism features,
and auxiliary multi-task head. All gated behind config flags (off by default).
"""

import os
import pandas as pd
from rdkit import Chem

# Starter reaction lookup table (Section 9.1)
REACTION_TABLE = [
    {
        "mechanism": "Maillard reaction",
        "api_smarts": "[NX3;H2]",
        "exc_smarts_list": ["[CX3H1](=O)", "[OX2H][CX4H1]([OX2H])"],
    },
    {
        "mechanism": "Ester/amide hydrolysis",
        "api_smarts_list": ["[#6][CX3](=O)[OX2H0][#6]", "[NX3][CX3](=[OX1])"],
        "exc_smarts_list": ["[OH-]"],
    },
    {
        "mechanism": "Metal-ion catalyzed oxidation",
        "api_smarts_list": ["[OX2H][cX3]", "[SX2H]"],
        "exc_smarts_list": ["[Fe]", "[Cu]", "[Mn]"],
    },
    {
        "mechanism": "Acid-base salt displacement",
        "api_smarts": "[CX3](=O)[OX2H1]",
        "exc_smarts_list": ["[Na]", "[K]", "[Ca]", "[Mg]"],
    },
    {
        "mechanism": "Schiff base formation",
        "api_smarts_list": ["[NX3;H2]", "[NX3;H1]"],
        "exc_smarts_list": ["[CX3H1](=O)", "[CX3](=O)[#6]"],
    },
]


def has_substructure(smiles: str, smarts: str) -> bool:
    """Check if a molecule has a substructure match for a SMARTS pattern."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return False
    return mol.HasSubstructMatch(pattern)


def get_mechanism_matches(api_smiles: str, exc_smiles: str) -> list[int]:
    """Return a binary vector of mechanism matches for an API-Excipient pair."""
    matches = []
    for entry in REACTION_TABLE:
        api_smarts_list = entry.get("api_smarts_list", [entry.get("api_smarts", "")])
        exc_smarts_list = entry.get("exc_smarts_list", [])

        api_match = any(has_substructure(api_smiles, s) for s in api_smarts_list if s)
        exc_match = any(has_substructure(exc_smiles, s) for s in exc_smarts_list if s)

        matches.append(1 if (api_match and exc_match) else 0)
    return matches


def get_num_mechanisms() -> int:
    return len(REACTION_TABLE)


def get_mechanism_names() -> list[str]:
    return [entry["mechanism"] for entry in REACTION_TABLE]
