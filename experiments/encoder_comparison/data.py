"""Data for the grid: the ARES split, and the primary / secondary description of one cell.

The split is the one the shipped ARES model was trained on (train 2770 / val 774) and the external
test set is the 22 held-out pairs. Both are read from final_project/data in this repository; pass
--data-root to point somewhere else.

Held-out is 22 pairs. There is no 24-pair variant and no heldout_unseen metric anywhere in this
experiment.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from experiments.encoder_comparison.reps import REP_SPECS, build_representation
from src.dataset import PGBDataset
from src.features import maccs_bits, morgan_bits

# Default: this repository's final_project/data. Its api_cluster train/val and the 22 held-out pairs
# are byte-identical (same md5) to the ARES copy these runs originally read from
# (<repo>/../../ARES/final_project/data), which does not exist on other machines such as the DGX.
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "final_project", "data")

# Precomputed vector tables for the lookup fingerprints (shipped in this repository).
VECTOR_FILES = {
    "pubchemfp": ["data/pubchem_fps.csv", "data/pubchem_fps_heldout.csv"],
    "mol2vec": ["data/mol2vec_embeddings.csv", "data/mol2vec_embeddings_heldout.csv"],
}

# Where precompute_embeddings.py writes the 5 encoders' mean-pooled vectors.
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "encoder_grid", "_embeddings")


def embedding_table_path(name: str) -> str:
    return os.path.join(EMBEDDING_DIR, f"{name}_mean.csv")


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

    def molecules(self, which: str) -> list:
        """Every API and excipient SMILES in one split, blanks dropped."""
        df = self.frame(which)
        mols = list(df["API_Smiles"].dropna()) + list(df["Excipient_Smiles"].dropna())
        return [str(m) for m in mols if str(m)]

    def all_molecules(self) -> list:
        """Every molecule in train + val + held-out. Used to build the shuffle derangement."""
        return sorted(set(self.molecules("train") + self.molecules("val") + self.molecules("heldout")))


def check_heldout_is_external(paths: SplitPaths) -> None:
    """Fail loudly if a held-out pair also occurs in train or val."""
    ho = paths.frame("heldout")
    ho_pairs = set(zip(ho["API_CID"], ho["Excipient_CID"]))
    for which in ("train", "val"):
        df = paths.frame(which)
        overlap = ho_pairs & set(zip(df["API_CID"], df["Excipient_CID"]))
        if overlap:
            raise SystemExit(f"{len(overlap)} held-out pair(s) also occur in {which}: {sorted(overlap)}")


def build_vector_lookup(family: str, paths: SplitPaths, out_dir: str, rebuild: bool = False) -> str:
    """Write one lookup CSV for pubchemfp / mol2vec covering every molecule in the data.

    The shipped vector tables are split over a main file and a held-out file, and 13 of their
    SMILES strings are written differently from ours (tautomer / Kekule / stereo notation), so a
    SMILES lookup silently misses them and the molecule arrives as an all-zero vector. Joining on
    CID and re-keying on our own SMILES gives full coverage for both lookup families.

    Reused if it is already there and covers every molecule, and written through a per-process
    temporary file and os.replace when it is not. Both matter once the sweep is sharded across
    GPUs: every one of the 505 runs asks for this table, so without the reuse it is rebuilt 505
    times, and without the atomic write two shards can write the same path at the same moment.
    A reader would then get a truncated CSV - a crash at best, and silently wrong vectors at
    worst, which is a whole run wasted and no way to tell which.
    """
    out_path = os.path.join(out_dir, f"{family}_lookup.csv")
    want_mols = set(paths.all_molecules())

    if not rebuild and os.path.exists(out_path):
        try:
            have = pd.read_csv(out_path, usecols=["smiles"])
            if want_mols <= set(have["smiles"].astype(str)):
                return out_path
        except Exception:
            pass        # unreadable or half-written: fall through and rebuild it properly

    frames = [pd.read_csv(os.path.join(PROJECT_ROOT, f)) for f in VECTOR_FILES[family]]
    vec = pd.concat(frames, ignore_index=True)
    vec["cid"] = vec["cid"].astype(str)
    vec = vec.drop_duplicates("cid", keep="first")
    feature_cols = [c for c in vec.columns if c not in ("cid", "smiles")]

    wanted = {}
    for which in ("train", "val", "heldout"):
        df = paths.frame(which)
        for cid, smi in (list(zip(df["API_CID"], df["API_Smiles"]))
                         + list(zip(df["Excipient_CID"], df["Excipient_Smiles"]))):
            if pd.notna(smi) and str(smi):
                wanted[str(smi)] = str(cid)

    by_cid = vec.set_index("cid")
    rows, missing = [], []
    for smi, cid in wanted.items():
        if cid in by_cid.index:
            rows.append([cid, smi] + by_cid.loc[cid, feature_cols].tolist())
        else:
            missing.append((cid, smi))
    if missing:
        raise SystemExit(f"{family}: no precomputed vector for {len(missing)} molecule(s), "
                         f"e.g. {missing[:3]}")

    os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{out_path}.{os.getpid()}.tmp"
    pd.DataFrame(rows, columns=["cid", "smiles"] + feature_cols).to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)      # atomic: a reader sees the old file or the new one
    return out_path


def table_paths_for(rep_names, paths: SplitPaths, lookup_dir: str) -> dict:
    """Resolve the lookup table every named representation needs.

    Fingerprint tables are rebuilt from the shipped CSVs (cheap, and it re-checks coverage).
    Embedding tables must already exist - they come from precompute_embeddings.py.
    """
    out = {}
    for name in sorted(set(rep_names)):
        base = name.replace("shuf_", "")
        if base not in REP_SPECS or REP_SPECS[base]["source"] != "table":
            continue
        if base in VECTOR_FILES:
            out[base] = build_vector_lookup(base, paths, lookup_dir)
        else:
            path = embedding_table_path(base)
            if not os.path.exists(path):
                raise SystemExit(
                    f"'{base}' needs its mean-pooled embedding table and {path} does not exist.\n"
                    "Run this once before any grid training:\n"
                    "  python -m experiments.encoder_comparison.precompute_embeddings")
            out[base] = path
    return out


def build_cell_representations(cell, paths: SplitPaths, lookup_dir: str):
    """(primary_rep, secondary_rep) for one cell, scalers fitted on TRAIN molecules only.

    Returns None in either slot where the cell has no such description. The shuffle derangement is
    built over train + val + held-out together so a molecule's donor never changes between splits.
    """
    names = cell.representations()
    tables = table_paths_for(names, paths, lookup_dir)
    train_mols = paths.molecules("train")
    all_mols = paths.all_molecules()

    def make(name):
        if name is None:
            return None
        rep = build_representation(name, tables)
        rep.build_shuffle(all_mols)
        rep.fit_scaler(train_mols)
        return rep

    primary_name = None
    if cell.primary is not None and (cell.no_cross_attn or cell.kind == "fixed_vector"):
        primary_name = cell.primary
    return make(primary_name), make(cell.secondary)


class NullEncoder(nn.Module):
    """Stand-in encoder for every cell that does not run a pretrained network.

    PGBCompatibilityModel asks an encoder for `output_dim` and `is_sequence_capable` only. With
    is_sequence_capable False the model takes the concat path, builds no projection / attention /
    pooling layers, and never calls encode(). Used by the fingerprint cells, the floor, and the
    no-cross-attention control (whose embedding is read from a table like a fingerprint).
    """

    is_sequence_capable = False

    def __init__(self, output_dim: int = 1):
        super().__init__()
        self.output_dim = output_dim

    def encode(self, smiles_batch):
        raise RuntimeError("NullEncoder.encode() must never be called - the model should be on "
                           "the concat path. This means use_cross_attn was true unexpectedly.")


class GridPGBDataset(PGBDataset):
    """PGBDataset whose primary and secondary descriptions come from this cell's Representations.

    Overrides both choices of PGBDataset so nothing is inherited from the old confounded
    assignment (where Morgan got MACCS, MACCS got Morgan, ChemBERTa got Morgan, and so on).

      * primary_rep None  -> the primary slot carries a single zero the tower ignores. That is the
                             case for a cross-attended encoder (its description is the structural
                             vector) and for the floor (no description at all).
      * secondary_rep None -> no secondary key in the batch, so the tower has no secondary block.
    """

    def __init__(self, csv_path, prior_table, pgb_norm_stats, primary_rep=None,
                 secondary_rep=None, sample_weight: float = 1.0, row_priors=None):
        super().__init__(csv_path, "fixed_vector", None, prior_table, pgb_norm_stats,
                         fixed_vector_encoder=None, sample_weight=sample_weight,
                         row_priors=row_priors)
        self.primary_rep = primary_rep
        self.secondary_rep = secondary_rep

    def _primary(self, smiles: str):
        if self.primary_rep is None:
            return np.zeros(1, dtype=np.float32), 1.0     # placeholder, ignored by the tower
        return self.primary_rep.vector(smiles), 1.0

    def _secondary(self, smiles: str):
        if self.secondary_rep is None:
            return None
        return self.secondary_rep.vector(smiles)


def make_grid_loader(csv, config, cell, prior_table, pgb_norm, primary_rep, secondary_rep,
                     row_priors=None, shuffle=False):
    from torch.utils.data import DataLoader
    from src.dataset import pgb_collate_fn
    ds = GridPGBDataset(csv, prior_table, pgb_norm, primary_rep=primary_rep,
                        secondary_rep=secondary_rep, row_priors=row_priors)
    return DataLoader(ds, batch_size=config.pgb_batch_size, shuffle=shuffle,
                      collate_fn=pgb_collate_fn, num_workers=0)


class OneRepresentationPGBDataset(PGBDataset):
    """The dataset the 45 completed `none`-column runs used. KEPT UNCHANGED.

    run_comparison.py reproduces those 45 runs and must keep reading the data exactly as it did,
    so this class is frozen. New grid cells use GridPGBDataset above.

      * no secondary fingerprint for any family,
      * fingerprint families get their own fingerprint,
      * sequence / graph families get no fingerprint at all (their representation is the encoder
        embedding), so the primary slot carries a single zero that the tower ignores.
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


def sanity_check_batch(batch: dict, cell, primary_rep, secondary_rep) -> None:
    """Catch a dataset/tower disagreement on the first batch instead of 40 epochs later."""
    if secondary_rep is None:
        if "api_secondary" in batch:
            raise RuntimeError(f"{cell.name}: cell has no secondary but the batch carries one")
    else:
        if "api_secondary" not in batch:
            raise RuntimeError(f"{cell.name}: cell has a secondary but the batch carries none")
        got = int(batch["api_secondary"].shape[1])
        if got != secondary_rep.dim:
            raise RuntimeError(f"{cell.name}: secondary width {got}, expected {secondary_rep.dim}")
        if not torch.isfinite(batch["api_secondary"]).all():
            raise RuntimeError(f"{cell.name}: non-finite values in the secondary vector")
    want = primary_rep.dim if primary_rep is not None else 1
    got = int(batch["api_primary"].shape[1])
    if got != want:
        raise RuntimeError(f"{cell.name}: primary width {got}, expected {want}")
    if not torch.isfinite(batch["api_primary"]).all():
        raise RuntimeError(f"{cell.name}: non-finite values in the primary vector")
