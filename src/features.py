"""
src/features.py
On-the-fly molecular feature extraction for the PGB head.
All functions accept a single SMILES string and return numpy arrays.
"""

from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
from rdkit.Chem.rdchem import Mol

# ── helpers ────────────────────────────────────────────────────────────────

def _parse_salt_aware(smiles: str) -> list[Mol]:
    """
    Split on '.' and return a list of valid RDKit Mol objects.
    This ensures that salts are treated as multiple fragments.
    """
    frags = smiles.split(".")
    mols = [Chem.MolFromSmiles(f) for f in frags]
    valid_mols = [m for m in mols if m is not None]
    
    if valid_mols:
        return valid_mols
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return [mol]
    return []


def _or_bits(arrs: list[np.ndarray]) -> np.ndarray:
    """Element-wise OR over a list of uint8 arrays. Returns zeros if list is empty."""
    if not arrs:
        return np.array([], dtype=np.uint8)
    result = arrs[0].copy()
    for a in arrs[1:]:
        result = result | a
    return result


# ── public API ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=10000)
def maccs_bits(smiles: str) -> np.ndarray:
    """
    167-bit MACCS key vector, salt-aware (bits are OR-ed across fragments).
    Returns float32 array of shape (167,).
    Bit 0 is kept to match RDKit's indexing (indices 0-166).
    """
    mols = _parse_salt_aware(smiles)
    if not mols:
        return np.zeros(167, dtype=np.float32)
    arrays = [np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.uint8) for m in mols]
    return _or_bits(arrays).astype(np.float32)


@lru_cache(maxsize=10000)
def morgan_bits(smiles: str, radius: int = 2, n_bits: int = 512) -> np.ndarray:
    """
    Morgan fingerprint, salt-aware OR-fold. Returns float32 (n_bits,).
    """
    mols = _parse_salt_aware(smiles)
    if not mols:
        return np.zeros(n_bits, dtype=np.float32)
    arrays = []
    gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for mol in mols:
        fp = gen.GetFingerprint(mol)
        arrays.append(np.array(list(fp), dtype=np.uint8))
    return _or_bits(arrays).astype(np.float32)


# Chemistry flag SMARTS — 18 flags total, in this exact order
_FLAG_SMARTS = [
    ("no_carbons",        None),          # handled specially: len(mol.GetAtoms() with atomic num 6) == 0
    ("oxidizer",          "[#8-,#8+0;!$([OH])][#6,#7,#16]"),   # peroxide-like or O attached to heteroatom
    ("strong_acid",       "[SX4](=O)(=O)[OH]"),                 # sulfonic acid
    ("strong_base",       "[OH-,O-;!$(OC=O)]"),                 # free oxide/hydroxide ion
    ("has_metal",         "[#3,#11,#12,#13,#19,#20,#22,#23,#25,#26,#27,#28,#29,#30]"),
    ("has_Mg",            "[#12]"),
    ("has_Ca",            "[#20]"),
    ("has_Al",            "[#13]"),
    ("has_Si",            "[#14]"),
    ("has_Ti_Fe",         "[#22,#26]"),
    ("has_Mn_Cr",         "[#25,#24]"),
    ("is_salt",           None),          # handled specially: '.' in SMILES or fragment count > 1
    ("very_short",        None),          # handled specially: heavy atom count <= 3
    ("many_OH",           "[OX2H]"),      # flag fires when count >= 2
    ("has_amine",         "[NX3;H1,H2;!$(NC=O)]"),
    ("carboxylic_acid",   "[CX3](=O)[OX2H1]"),
    ("ester",             "[CX3](=O)[OX2H0][#6]"),
    ("phenol",            "[OX2H][cX3]"),
]

_FLAG_PATTERNS = None  # lazily compiled


def _compile_flag_patterns():
    global _FLAG_PATTERNS
    if _FLAG_PATTERNS is not None:
        return
    _FLAG_PATTERNS = []
    for name, smarts in _FLAG_SMARTS:
        if smarts is not None:
            pat = Chem.MolFromSmarts(smarts)
        else:
            pat = None
        _FLAG_PATTERNS.append((name, pat))


@lru_cache(maxsize=10000)
def chemistry_flags(smiles: str) -> np.ndarray:
    """
    18-bit chemistry flag vector, salt-aware.
    Returns float32 array of shape (18,).
    Each flag is 1.0 if ANY fragment matches.
    """
    _compile_flag_patterns()
    mols = _parse_salt_aware(smiles)
    flags = np.zeros(18, dtype=np.float32)
    if not mols:
        return flags

    for i, (name, pat) in enumerate(_FLAG_PATTERNS):
        if name == "no_carbons":
            # True if no fragment has a carbon atom
            has_carbon = any(
                any(a.GetAtomicNum() == 6 for a in m.GetAtoms())
                for m in mols
            )
            flags[i] = 0.0 if has_carbon else 1.0
        elif name == "is_salt":
            # True if original SMILES has a dot OR more than one fragment parsed
            flags[i] = 1.0 if ("." in smiles or len(mols) > 1) else 0.0
        elif name == "very_short":
            # True if every fragment has <= 3 heavy atoms
            all_short = all(m.GetNumHeavyAtoms() <= 3 for m in mols)
            flags[i] = 1.0 if all_short else 0.0
        elif name == "many_OH":
            # True if any fragment has >= 2 hydroxyl matches
            total = sum(len(m.GetSubstructMatches(pat)) for m in mols if pat)
            flags[i] = 1.0 if total >= 2 else 0.0
        else:
            if pat is not None:
                match = any(m.HasSubstructMatch(pat) for m in mols)
                flags[i] = 1.0 if match else 0.0

    return flags


# 12 RDKit descriptors (replaces the 21 currently in descriptors.py for PGB models)
PGB_DESCRIPTOR_NAMES = [
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
    "NumHeteroatoms",
    "qed",
]


@lru_cache(maxsize=10000)
def pgb_descriptors(smiles: str) -> np.ndarray:
    """
    12 physicochemical descriptors for PGB.
    Returns float32 (12,). NaN on parse failure — caller must nan_to_num.
    """
    from rdkit.Chem import QED
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(12, float("nan"), dtype=np.float32)

    vals = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)),
        QED.qed(mol),
    ]
    return np.array(vals, dtype=np.float32)


@lru_cache(maxsize=10000)
def salt_context(smiles: str) -> np.ndarray:
    """
    8-dimensional salt/inorganic context vector.
    Features:
      0: num_fragments
      1: num_metal_atoms
      2: num_inorganic_fragments (fragments with no carbons)
      3: formal_charge_total
      4: max_abs_formal_charge
      5: organic_heavy_atoms
      6: inorganic_heavy_atoms
      7: has_counterion (1 if num_fragments > 1, else 0)
    """
    mols = _parse_salt_aware(smiles)
    if not mols:
        return np.zeros(8, dtype=np.float32)

    num_fragments = len(mols)
    num_metal_atoms = 0
    num_inorganic_fragments = 0
    formal_charge_total = 0
    max_abs_formal_charge = 0
    organic_heavy = 0
    inorganic_heavy = 0

    metals = {3, 11, 12, 13, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30}

    for m in mols:
        has_carbon = False
        frag_heavy = m.GetNumHeavyAtoms()
        frag_charge = 0
        for a in m.GetAtoms():
            if a.GetAtomicNum() == 6:
                has_carbon = True
            if a.GetAtomicNum() in metals:
                num_metal_atoms += 1
            fc = a.GetFormalCharge()
            frag_charge += fc
            if abs(fc) > max_abs_formal_charge:
                max_abs_formal_charge = abs(fc)
                
        formal_charge_total += frag_charge
        
        if has_carbon:
            organic_heavy += frag_heavy
        else:
            inorganic_heavy += frag_heavy
            num_inorganic_fragments += 1

    return np.array([
        num_fragments,
        num_metal_atoms,
        num_inorganic_fragments,
        formal_charge_total,
        max_abs_formal_charge,
        organic_heavy,
        inorganic_heavy,
        1.0 if num_fragments > 1 else 0.0
    ], dtype=np.float32)
