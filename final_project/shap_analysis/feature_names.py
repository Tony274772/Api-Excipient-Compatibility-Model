"""
Feature names and grouping definitions for the PGB Morgan model.
Maps the 2,468-dimensional flat feature representation into interpretable names and hierarchical groups.
"""

from typing import Dict, List, Tuple

# 12 PGB Physicochemical Descriptors
PGB_DESCRIPTOR_BASE_NAMES = [
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
    "QED",
]

# 8 Salt Context Features
SALT_CONTEXT_BASE_NAMES = [
    "num_fragments",
    "num_metal_atoms",
    "num_inorganic_fragments",
    "formal_charge_total",
    "max_abs_formal_charge",
    "organic_heavy_atoms",
    "inorganic_heavy_atoms",
    "has_counterion",
]

# 18 Chemistry Flags
CHEMISTRY_FLAG_BASE_NAMES = [
    "no_carbons",
    "oxidizer",
    "strong_acid",
    "strong_base",
    "has_metal",
    "has_Mg",
    "has_Ca",
    "has_Al",
    "has_Si",
    "has_Ti_Fe",
    "has_Mn_Cr",
    "is_salt",
    "very_short",
    "many_OH",
    "has_amine",
    "carboxylic_acid",
    "ester",
    "phenol",
]

# 5 Excipient Prior Features
PRIOR_BASE_NAMES = [
    "p_exact",
    "p_delta",
    "p_family",
    "log_n",
    "unseen",
]

# 5 Mechanism Reaction Flags, in the order src/lookup.py computes them
MECHANISM_BASE_NAMES = [
    "Maillard_reaction",
    "Ester_amide_hydrolysis",
    "Metal_catalyzed_oxidation",
    "Acid_base_salt_displacement",
    "Schiff_base_formation",
]


def check_names_match_features():
    """Raise if the hard-coded names above are not the order src/ computes the features in."""
    import re
    from src.features import PGB_DESCRIPTOR_NAMES, _FLAG_SMARTS
    from src.lookup import get_mechanism_names
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    pairs = [
        ("descriptors", [n.lower() for n in PGB_DESCRIPTOR_NAMES], [n.lower() for n in PGB_DESCRIPTOR_BASE_NAMES]),
        ("chemistry flags", [n for n, _ in _FLAG_SMARTS], CHEMISTRY_FLAG_BASE_NAMES),
        ("mechanisms", [norm(n).replace("metalion", "metal") for n in get_mechanism_names()], [norm(n) for n in MECHANISM_BASE_NAMES]),
    ]
    for what, src_names, ours in pairs:
        if list(src_names) != list(ours):
            raise ValueError(f"SHAP feature names for {what} do not match src/: {src_names} vs {ours}")


def build_feature_metadata() -> Tuple[List[str], Dict[str, Tuple[int, int]], Dict[str, List[str]]]:
    """
    Constructs the list of all 2,468 feature names and their index slice ranges.
    Returns:
        (feature_names, feature_groups, macro_groups)
    """
    names: List[str] = []
    groups: Dict[str, Tuple[int, int]] = {}

    # 1. API Primary Fingerprint (Morgan 1024-bit)
    start_idx = len(names)
    for b in range(1024):
        names.append(f"API_Morgan_Bit_{b}")
    groups["api_primary"] = (start_idx, len(names))

    # 2. Excipient Primary Fingerprint (Morgan 1024-bit)
    start_idx = len(names)
    for b in range(1024):
        names.append(f"Exc_Morgan_Bit_{b}")
    groups["exc_primary"] = (start_idx, len(names))

    # 3. API Secondary Fingerprint (MACCS 167-bit)
    start_idx = len(names)
    for b in range(167):
        names.append(f"API_MACCS_Bit_{b}")
    groups["api_secondary"] = (start_idx, len(names))

    # 4. Excipient Secondary Fingerprint (MACCS 167-bit)
    start_idx = len(names)
    for b in range(167):
        names.append(f"Exc_MACCS_Bit_{b}")
    groups["exc_secondary"] = (start_idx, len(names))

    # 5. API PGB Descriptors (12-d)
    start_idx = len(names)
    for desc in PGB_DESCRIPTOR_BASE_NAMES:
        names.append(f"API_{desc}")
    groups["api_pgb_desc"] = (start_idx, len(names))

    # 6. Excipient PGB Descriptors (12-d)
    start_idx = len(names)
    for desc in PGB_DESCRIPTOR_BASE_NAMES:
        names.append(f"Exc_{desc}")
    groups["exc_pgb_desc"] = (start_idx, len(names))

    # 7. API Salt Context (8-d)
    start_idx = len(names)
    for salt in SALT_CONTEXT_BASE_NAMES:
        names.append(f"API_salt_{salt}")
    groups["api_salt"] = (start_idx, len(names))

    # 8. Excipient Salt Context (8-d)
    start_idx = len(names)
    for salt in SALT_CONTEXT_BASE_NAMES:
        names.append(f"Exc_salt_{salt}")
    groups["exc_salt"] = (start_idx, len(names))

    # 9. API Chemistry Flags (18-d)
    start_idx = len(names)
    for flag in CHEMISTRY_FLAG_BASE_NAMES:
        names.append(f"API_flag_{flag}")
    groups["api_flags"] = (start_idx, len(names))

    # 10. Excipient Chemistry Flags (18-d)
    start_idx = len(names)
    for flag in CHEMISTRY_FLAG_BASE_NAMES:
        names.append(f"Exc_flag_{flag}")
    groups["exc_flags"] = (start_idx, len(names))

    # 11. Prior Vector (5-d)
    start_idx = len(names)
    for p in PRIOR_BASE_NAMES:
        names.append(f"Prior_{p}")
    groups["prior_vec"] = (start_idx, len(names))

    # 12. Mechanism Reaction Flags (5-d)
    start_idx = len(names)
    for m in MECHANISM_BASE_NAMES:
        names.append(f"Mech_{m}")
    groups["mech_flags"] = (start_idx, len(names))

    # Macro-level groupings for higher-level attribution
    macro_groups = {
        "Morgan Fingerprints (Primary 1024-d)": ["api_primary", "exc_primary"],
        "MACCS Keys (Secondary 167-d)": ["api_secondary", "exc_secondary"],
        "Physicochemical Descriptors (12-d)": ["api_pgb_desc", "exc_pgb_desc"],
        "Chemistry Flags (18-d)": ["api_flags", "exc_flags"],
        "Salt Context (8-d)": ["api_salt", "exc_salt"],
        "Excipient Prior Vector (5-d)": ["prior_vec"],
        "Reaction Mechanisms (5-d)": ["mech_flags"],
        # Tower/Molecule level breakdown
        "API Molecular Features": [
            "api_primary",
            "api_secondary",
            "api_pgb_desc",
            "api_salt",
            "api_flags",
        ],
        "Excipient Molecular Features": [
            "exc_primary",
            "exc_secondary",
            "exc_pgb_desc",
            "exc_salt",
            "exc_flags",
        ],
        "Pairwise & Prior Features": [
            "prior_vec",
            "mech_flags",
        ],
    }

    return names, groups, macro_groups


# Precompute standard constants
FEATURE_NAMES, FEATURE_GROUPS, MACRO_GROUPS = build_feature_metadata()
TOTAL_FEATURES = len(FEATURE_NAMES)  # 2468


def get_feature_name(index: int) -> str:
    """Returns the name for a given feature index."""
    if 0 <= index < len(FEATURE_NAMES):
        return FEATURE_NAMES[index]
    raise IndexError(f"Index {index} out of range [0, {len(FEATURE_NAMES)-1}]")


def get_feature_group(index: int) -> str:
    """Finds which feature group contains the given index."""
    for group_name, (start, end) in FEATURE_GROUPS.items():
        if start <= index < end:
            return group_name
    return "unknown"
