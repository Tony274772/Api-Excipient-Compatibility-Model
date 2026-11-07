"""Data for the comparison: the ARES split, and one representation per family.

The split is the one the shipped ARES model was trained on (train 2770 / val 774) and the
external test set is the 22 held-out pairs. Both are read from the ARES project folder; pass
--data-root to point somewhere else.
"""

import os

import numpy as np
import pandas as pd

from src.dataset import PGBDataset
from src.features import maccs_bits, morgan_bits

# Default: the ARES project folder, next to this repository.
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_DATA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "..", "ARES", "final_project", "data"))

# Precomputed vector tables for the lookup families (live in this repository).
VECTOR_FILES = {
    "pubchemfp": ["data/pubchem_fps.csv", "data/pubchem_fps_heldout.csv"],
    "mol2vec": ["data/mol2vec_embeddings.csv", "data/mol2vec_embeddings_heldout.csv"],
}


class SplitPaths:
    """train / val / held-out CSV paths, resolved once and printed in the run info."""

    def __init__(self, data_root: str = None, split: str = "api_cluster"):
        root = os.path.abspath(data_root or DEFAULT_DATA_ROOT)
        self.root = root
        self.train = os.path.join(root, "splits", split, "train.csv")
        self.val = os.path.join(root, "splits", split, "val.csv")
        self.heldout = os.path.join(root, "heldout", "held_out_test_set.csv")
        missing = [p for p in (self.train, self.val, self.heldout) if not os.path.exists(p)]
        if missing:
            raise SystemExit(
                "data files not found:\n  " + "\n  ".join(missing)
                + f"\nPass --data-root <folder> (expected layout <root>/splits/{split}/train.csv, "
                  "<root>/heldout/held_out_test_set.csv)."
            )

    def frame(self, which: str) -> pd.DataFrame:
        """train / val / heldout as a DataFrame with the label in Outcome1."""
        path = {"train": self.train, "val": self.val, "heldout": self.heldout}[which]
        return pd.read_csv(path).rename(columns={"ground_truth": "Outcome1"})


def check_heldout_is_external(paths: SplitPaths) -> None:
    """Fail loudly if a held-out pair also occurs in train or val."""
    ho = paths.frame("heldout")
    ho_pairs = set(zip(ho["API_CID"], ho["Excipient_CID"]))
    for which in ("train", "val"):
        df = paths.frame(which)
        overlap = ho_pairs & set(zip(df["API_CID"], df["Excipient_CID"]))
        if overlap:
            raise SystemExit(f"{len(overlap)} held-out pair(s) also occur in {which}: {sorted(overlap)}")


def build_vector_lookup(family: str, paths: SplitPaths, out_dir: str) -> str:
    """Write one lookup CSV for pubchemfp / mol2vec covering every molecule in the data.

    The shipped vector tables are split over a main file and a held-out file, and 13 of their
    SMILES strings are written differently from ours (tautomer / Kekule / stereo notation), so a
    SMILES lookup silently misses them and the molecule arrives as an all-zero vector. Joining on
    CID and re-keying on our own SMILES gives full coverage for both lookup families.
    """
    frames = [pd.read_csv(os.path.join(PROJECT_ROOT, f)) for f in VECTOR_FILES[family]]
    vec = pd.concat(frames, ignore_index=True)
    vec["cid"] = vec["cid"].astype(str)
    vec = vec.drop_duplicates("cid", keep="first")
    feature_cols = [c for c in vec.columns if c not in ("cid", "smiles")]

    wanted = {}
    for which in ("train", "val", "heldout"):
        df = paths.frame(which)
        for cid, smi in list(zip(df["API_CID"], df["API_Smiles"])) + list(zip(df["Excipient_CID"], df["Excipient_Smiles"])):
            wanted[str(smi)] = str(cid)

    by_cid = vec.set_index("cid")
    rows, missing = [], []
    for smi, cid in wanted.items():
        if cid in by_cid.index:
            rows.append([cid, smi] + by_cid.loc[cid, feature_cols].tolist())
        else:
            missing.append((cid, smi))
    if missing:
        raise SystemExit(f"{family}: no precomputed vector for {len(missing)} molecule(s), e.g. {missing[:3]}")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{family}_lookup.csv")
    pd.DataFrame(rows, columns=["cid", "smiles"] + feature_cols).to_csv(path, index=False)
    return path


class OneRepresentationPGBDataset(PGBDataset):
    """PGBDataset that gives each family exactly one molecular representation.

    Overrides the primary / secondary choice of PGBDataset:
      * no secondary fingerprint for any family,
      * fingerprint families get their own fingerprint,
      * sequence / graph families get no fingerprint at all (their representation is the
        encoder embedding), so the primary slot carries a single zero that the tower ignores.
    """

    def _primary(self, smiles: str):
        if self.enc_name != "fixed_vector":
            return np.zeros(1, dtype=np.float32), 1.0          # placeholder, ignored by the tower
        if self.fvs == "morgan":
            return morgan_bits(smiles, n_bits=1024), 1.0
        if self.fvs == "maccs":
            return maccs_bits(smiles), 1.0
        if self.fvs in ("pubchemfp", "mol2vec"):
            if self.fv_enc is None:
                raise RuntimeError(f"{self.fvs} needs a vector lookup encoder")
            v = self.fv_enc.encode([smiles]).cpu().numpy()[0]
            if not v.any():
                raise RuntimeError(f"no precomputed {self.fvs} vector for SMILES {smiles!r}")
            return v.astype(np.float32), 1.0
        raise RuntimeError(f"unsupported fixed-vector source {self.fvs!r}")

    def _secondary(self, smiles: str):
        return None
