"""Per-molecule and per-pair feature computation for the PGB architecture.

Implements:
- 18 chemistry flags (salt-aware)
- 12 RDKit descriptors (z-scored on train)
- MACCS keys (166 bits, salt-aware OR)
- Morgan fingerprints (512 or 1024 bits, salt-aware OR)
- 5 mechanism flags per pair (from src/lookup.py rules)
- Salt-aware parsing for all fingerprints / flags

All functions work on raw SMILES strings so they can be called at
dataset construction time or on the fly.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    MACCSkeys,
    rdMolDescriptors,
)
from rdkit.DataStructs import ConvertToNumpyArray

from src.lookup import REACTION_TABLE, has_substructure


# ═══════════════════════════════════════════════════════════════════════════════
# Salt-aware helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_fragments(smiles: str) -> list[Chem.Mol]:
    """Split a SMILES on '.' and return valid RDKit Mol objects.

    If the SMILES contains no dot, returns a single-element list.
    Fragments that fail parsing are silently dropped.
    """
    if not smiles or smiles == "nan":
        return []
    parts = smiles.split(".")
    mols = []
    for part in parts:
        part = part.strip()
        if part:
            mol = Chem.MolFromSmiles(part)
            if mol is not None:
                mols.append(mol)
    # Also try the full SMILES as-is (covers salts like [Mg+2].[Cl-].[Cl-])
    full = Chem.MolFromSmiles(smiles)
    if full is not None and not mols:
        mols = [full]
    return mols


def _or_bit_arrays(*arrays: np.ndarray) -> np.ndarray:
    """Bitwise OR of multiple binary arrays."""
    result = np.zeros_like(arrays[0])
    for a in arrays:
        result = np.maximum(result, a)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MACCS keys (166 bits, bit 0 dropped → 166 features)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_maccs(smiles: str) -> np.ndarray:
    """166-bit MACCS keys, salt-aware (OR over fragments), bit 0 dropped."""
    mols = _parse_fragments(smiles)
    if not mols:
        return np.zeros(166, dtype=np.float32)

    fps = []
    for mol in mols:
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(167, dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        fps.append(arr[1:])  # drop bit 0 to get 166

    return _or_bit_arrays(*fps) if fps else np.zeros(166, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Morgan fingerprints (radius 2, configurable length)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_morgan(smiles: str, n_bits: int = 512) -> np.ndarray:
    """Morgan fingerprint, salt-aware (OR over fragments).

    Args:
        smiles: SMILES string (may contain '.' for salts).
        n_bits: 512 for side branches, 1024 for Morgan family primary.
    """
    mols = _parse_fragments(smiles)
    if not mols:
        return np.zeros(n_bits, dtype=np.float32)

    fps = []
    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        fps.append(arr)

    return _or_bit_arrays(*fps) if fps else np.zeros(n_bits, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 12 RDKit descriptors (replaces the 21 from the base pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

GB_DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumHeteroatoms",   # new (replaces fr_* counts)
    "qed",              # new (replaces fr_* counts)
]


def _compute_gb_descriptors_single(mol: Chem.Mol) -> list[float]:
    """Compute 12-dim descriptor vector for a single Mol object."""
    values = []
    for name in GB_DESCRIPTOR_NAMES:
        if name == "FractionCSP3":
            values.append(rdMolDescriptors.CalcFractionCSP3(mol))
        elif name == "NumHeteroatoms":
            values.append(float(Descriptors.NumHeteroatoms(mol)))
        elif name == "qed":
            from rdkit.Chem.QED import qed
            values.append(qed(mol))
        else:
            func = getattr(Descriptors, name, None)
            if func is None:
                values.append(0.0)
            else:
                values.append(float(func(mol)))
    return values


def compute_gb_descriptors(smiles: str) -> np.ndarray:
    """12-dim RDKit descriptors, using the first valid fragment."""
    mols = _parse_fragments(smiles)
    if not mols:
        return np.zeros(len(GB_DESCRIPTOR_NAMES), dtype=np.float32)

    # Use the largest fragment by atom count (the "main" molecule)
    main_mol = max(mols, key=lambda m: m.GetNumAtoms())
    vals = _compute_gb_descriptors_single(main_mol)
    return np.array(vals, dtype=np.float32)


def compute_gb_descriptor_norm_stats(train_smiles: list[str]) -> dict:
    """Compute mean/std of the 12 descriptors from training SMILES only.

    Returns dict with keys 'mean' and 'std', each a list of 12 floats.
    """
    all_descs = []
    for smi in train_smiles:
        d = compute_gb_descriptors(smi)
        if not np.any(np.isnan(d)):
            all_descs.append(d)

    if not all_descs:
        return {
            "mean": [0.0] * len(GB_DESCRIPTOR_NAMES),
            "std": [1.0] * len(GB_DESCRIPTOR_NAMES),
        }

    arr = np.stack(all_descs)
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
    }


def normalize_gb_descriptors(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Normalize: (x - mean) / (std + 1e-8)."""
    return (values - mean) / (std + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# 18 Chemistry flags (per-molecule, salt-aware)
# ═══════════════════════════════════════════════════════════════════════════════

# SMARTS patterns for the 18 flags
_CHEM_FLAG_DEFS = [
    ("no_carbons",      None),                                   # special logic
    ("oxidizer",        "[O-][Cl](=O)(=O)=O"),                   # perchlorate-like
    ("strong_acid",     "[H]O[S](=O)(=O)[!H]"),                  # sulphuric acid
    ("strong_base",     "[OH-]"),                                 # hydroxide
    ("metal",           "[#3,#11,#12,#13,#19,#20,#22,#24,#25,#26,#29,#30]"),
    ("Mg",              "[Mg]"),
    ("Ca",              "[Ca]"),
    ("Al",              "[Al]"),
    ("Si",              "[Si]"),
    ("Ti_Fe",           "[Ti,Fe]"),
    ("Mn_Cr",           "[Mn,Cr]"),
    ("is_salt",         None),                                   # special: has '.'
    ("very_short",      None),                                   # special: ≤ 5 chars
    ("many_OH",         "[OX2H1]"),                              # count ≥ 3
    ("amine",           "[NX3;H2,H1;!$([NX3]C=O)]"),
    ("carboxylic_acid", "[CX3](=O)[OX2H1]"),
    ("ester",           "[#6][CX3](=O)[OX2H0][#6]"),
    ("phenol",          "[OX2H][cX3]"),
]

CHEM_FLAG_NAMES = [name for name, _ in _CHEM_FLAG_DEFS]


def compute_chem_flags(smiles: str) -> np.ndarray:
    """18-dim binary chemistry flags, salt-aware.

    For bit-pattern flags, fragments are OR-ed.
    Special flags (no_carbons, is_salt, very_short, many_OH) use custom logic.
    """
    flags = np.zeros(len(_CHEM_FLAG_DEFS), dtype=np.float32)
    mols = _parse_fragments(smiles)

    if not mols and not smiles:
        return flags

    for idx, (name, smarts) in enumerate(_CHEM_FLAG_DEFS):
        if name == "no_carbons":
            # True if no fragment contains carbon
            has_carbon = False
            for mol in mols:
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() == 6:
                        has_carbon = True
                        break
                if has_carbon:
                    break
            flags[idx] = 0.0 if has_carbon else 1.0

        elif name == "is_salt":
            flags[idx] = 1.0 if "." in smiles else 0.0

        elif name == "very_short":
            # Very short SMILES string (≤ 5 non-whitespace chars)
            flags[idx] = 1.0 if len(smiles.strip()) <= 5 else 0.0

        elif name == "many_OH":
            # Count OH groups across all fragments; flag if ≥ 3
            total_oh = 0
            pattern = Chem.MolFromSmarts("[OX2H1]")
            if pattern is not None:
                for mol in mols:
                    total_oh += len(mol.GetSubstructMatches(pattern))
            flags[idx] = 1.0 if total_oh >= 3 else 0.0

        else:
            # Standard SMARTS flag — OR across fragments
            if smarts is None:
                continue
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            for mol in mols:
                if mol.HasSubstructMatch(pattern):
                    flags[idx] = 1.0
                    break

    return flags


# ═══════════════════════════════════════════════════════════════════════════════
# 5 Mechanism flags (per-pair, reusing src/lookup.py rules)
# ═══════════════════════════════════════════════════════════════════════════════

MECHANISM_NAMES = [entry["mechanism"] for entry in REACTION_TABLE]


def compute_mechanism_flags(api_smiles: str, exc_smiles: str) -> np.ndarray:
    """5-dim binary mechanism flags for an API–excipient pair.

    Uses the SMARTS rules already defined in src/lookup.py.
    Salt-aware: checks each fragment of both API and excipient.
    """
    flags = np.zeros(len(REACTION_TABLE), dtype=np.float32)

    api_frags = api_smiles.split(".") if api_smiles else []
    exc_frags = exc_smiles.split(".") if exc_smiles else []

    for idx, entry in enumerate(REACTION_TABLE):
        api_smarts_list = entry.get("api_smarts_list", [])
        if not api_smarts_list:
            single = entry.get("api_smarts", "")
            if single:
                api_smarts_list = [single]

        exc_smarts_list = entry.get("exc_smarts_list", [])

        # Check API side (any fragment matches any API SMARTS)
        api_match = False
        for frag in api_frags:
            frag = frag.strip()
            if not frag:
                continue
            for smarts in api_smarts_list:
                if smarts and has_substructure(frag, smarts):
                    api_match = True
                    break
            if api_match:
                break

        # Check excipient side
        exc_match = False
        for frag in exc_frags:
            frag = frag.strip()
            if not frag:
                continue
            for smarts in exc_smarts_list:
                if smarts and has_substructure(frag, smarts):
                    exc_match = True
                    break
            if exc_match:
                break

        flags[idx] = 1.0 if (api_match and exc_match) else 0.0

    return flags
