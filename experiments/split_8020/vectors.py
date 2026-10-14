"""
Coverage checks and held-out filling for FixedVectorEncoder (precomputed vectors read from CSV).

FixedVectorEncoder returns an all-zero vector for any molecule it cannot find, silently.
Held-out SMILES carry stereochemistry and are not in its SMILES->CID map, so without this
step 27 of the 39 held-out molecules would be encoded as zeros.

Every held-out molecule is given the vector training used for the same CID. Molecules whose
CID is not in the vector file get a computed vector, using the same functions that built the
files (data/compute_fixed_vectors.py) or, for PubChem, the fingerprints fetched from PubChem
(data/pubchem_fps_heldout.csv).
"""

import importlib.util
import os

import numpy as np
import pandas as pd

FIXED_VECTOR_FILES = {
    "pubchemfp": "data/pubchem_fps.csv",
    "maccs": "data/maccs_keys.csv",
    "mol2vec": "data/mol2vec_embeddings.csv",
    "morgan": "data/morgan_fps.csv",
}
PUBCHEM_HELDOUT_FILE = "data/pubchem_fps_heldout.csv"


def _has_vector(enc, smiles: str) -> bool:
    return smiles in enc._vectors or (smiles in enc._smiles_to_cid and enc._smiles_to_cid[smiles] in enc._vectors)


def assert_coverage(enc, smiles: list[str], what: str) -> None:
    """Raise if any molecule would be encoded as an all-zero 'unknown' vector."""
    missing = sorted({s for s in smiles if s and not _has_vector(enc, s)})
    if missing:
        raise RuntimeError(f"{len(missing)} {what} molecules have no {enc.source} vector "
                           f"(they would silently become zeros), e.g. {missing[:3]}")


def _compute_fixed_vectors_module():
    spec = importlib.util.spec_from_file_location("compute_fixed_vectors", "data/compute_fixed_vectors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compute(source: str, smiles: list[str], cids: list[str], dim: int) -> np.ndarray:
    if source == "pubchemfp":
        if not os.path.exists(PUBCHEM_HELDOUT_FILE):
            raise FileNotFoundError(f"{PUBCHEM_HELDOUT_FILE} not found; PubChem fingerprints for held-out CIDs {cids} are needed")
        ex = pd.read_csv(PUBCHEM_HELDOUT_FILE)
        ex["cid"] = ex["cid"].astype(str)
        cols = [c for c in ex.columns if c not in ("cid", "smiles")]
        table = dict(zip(ex["cid"], ex[cols].to_numpy(np.float32)))
        absent = [c for c in cids if c not in table]
        if absent:
            raise RuntimeError(f"No PubChem fingerprint for held-out CIDs {absent} in {PUBCHEM_HELDOUT_FILE}")
        return np.stack([table[c] for c in cids])
    mod = _compute_fixed_vectors_module()
    if source == "morgan":
        return mod.compute_morgan(smiles, n_bits=dim)
    if source == "maccs":
        return mod.compute_maccs(smiles)
    if source == "mol2vec":
        return mod.compute_mol2vec(smiles)
    raise ValueError(f"Unknown fixed-vector source: {source}")


def fill_heldout_vectors(enc, heldout_df: pd.DataFrame) -> dict:
    """Register a vector for every held-out molecule. Returns counts for run_info.json."""
    pairs = {}
    for smi_col, cid_col in [("API_Smiles", "API_CID"), ("Excipient_Smiles", "Excipient_CID")]:
        for smi, cid in zip(heldout_df[smi_col], heldout_df[cid_col]):
            if pd.notna(smi) and smi:
                pairs[str(smi)] = str(int(cid))

    already = by_cid = 0
    to_compute = []
    for smi, cid in pairs.items():
        if _has_vector(enc, smi):
            already += 1
        elif cid in enc._vectors:
            enc._vectors[smi] = enc._vectors[cid]
            by_cid += 1
        else:
            to_compute.append((smi, cid))

    if to_compute:
        vecs = _compute(enc.source, [s for s, _ in to_compute], [c for _, c in to_compute], enc.output_dim)
        if vecs.shape[1] != enc.output_dim:
            raise RuntimeError(f"computed {enc.source} vectors have dim {vecs.shape[1]}, encoder expects {enc.output_dim}")
        for (smi, _), v in zip(to_compute, vecs):
            enc._vectors[smi] = v.astype(np.float32)

    assert_coverage(enc, list(pairs), "held-out")
    return {"heldout_molecules": len(pairs), "found_by_smiles": already, "found_by_cid": by_cid,
            "computed": len(to_compute)}
