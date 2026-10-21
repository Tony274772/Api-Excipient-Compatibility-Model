"""
Pre-flight and post-training checks. Each one raises instead of letting a run go on with silently
substituted values (zero vectors, zero descriptors, empty graphs, stale checkpoints).

The repo code has these silent fallbacks, which the checks rule out for the data actually used:
  - CompatibilityDataset: a CID missing from the descriptor table gets an all-zero descriptor vector
  - FixedVectorEncoder / GBCompatibilityDataset: an unknown molecule gets an all-zero vector
  - PGBDataset: an all-zero PubChem / Mol2vec vector is marked "unavailable"
  - StanfordGATEncoder: a molecule that fails featurisation is skipped and gets all-zero embeddings
  - train loops: the best checkpoint is re-loaded from disk, so a stale best_model.pt could be scored
"""

import random
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.metrics import average_precision_score

from src.descriptors import DESCRIPTOR_NAMES

REQUIRED = ["API_CID", "Excipient_CID", "Outcome1", "API_Smiles", "Excipient_Smiles"]


class PreflightError(RuntimeError):
    pass


@contextmanager
def preserved_rng():
    """Checks must not change training: restore every RNG stream afterwards."""
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def unique_smiles(*frames):
    return sorted({str(s) for df in frames for c in ["API_Smiles", "Excipient_Smiles"] for s in df[c]})


def check_split(train_df, val_df, heldout_df):
    for name, df in [("train", train_df), ("val", val_df), ("held-out", heldout_df)]:
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise PreflightError(f"{name}: missing columns {missing}")
        if df[REQUIRED].isna().any().any():
            raise PreflightError(f"{name}: NaN in {df[REQUIRED].columns[df[REQUIRED].isna().any()].tolist()}")
        for c in ["API_Smiles", "Excipient_Smiles"]:
            if (df[c].astype(str).str.strip() == "").any():
                raise PreflightError(f"{name}: empty {c}")
        if not set(df["Outcome1"].unique()) <= {0, 1}:
            raise PreflightError(f"{name}: labels must be 0/1, got {sorted(df['Outcome1'].unique())}")
        if df["Outcome1"].nunique() < 2:
            raise PreflightError(f"{name}: only one class")
        if df.duplicated(["API_CID", "Excipient_CID"]).any():
            raise PreflightError(f"{name}: duplicate (API_CID, Excipient_CID) rows")
    pairs = lambda d: set(zip(d.API_CID.astype(int), d.Excipient_CID.astype(int)))
    if pairs(train_df) & pairs(val_df):
        raise PreflightError(f"{len(pairs(train_df) & pairs(val_df))} CID pairs are in both train and val")
    bad = [s for s in unique_smiles(train_df, val_df, heldout_df) if Chem.MolFromSmiles(s) is None]
    if bad:
        raise PreflightError(f"{len(bad)} SMILES RDKit cannot parse, e.g. {bad[:3]}")


def check_descriptors(df, api_desc_path, exc_desc_path, what):
    """Every CID has a descriptor row without NaN (CompatibilityDataset would use zeros)."""
    for col, path in [("API_CID", api_desc_path), ("Excipient_CID", exc_desc_path)]:
        table = pd.read_csv(path)
        if table[col].duplicated().any():
            raise PreflightError(f"{path}: duplicate {col}")
        missing = sorted(set(df[col]) - set(table[col]))
        if missing:
            raise PreflightError(f"{what}: {len(missing)} {col}s have no row in {path} (would become zeros): {missing[:5]}")
        used = table[table[col].isin(set(df[col]))]
        if used[DESCRIPTOR_NAMES].isna().any().any():
            raise PreflightError(f"{what}: NaN descriptors in {path} for {used[used[DESCRIPTOR_NAMES].isna().any(axis=1)][col].tolist()[:5]}")


def check_norm_stats(stats):
    for k in ["api_mean", "api_std", "exc_mean", "exc_std"]:
        v = np.asarray(stats[k], dtype=float)
        if not np.isfinite(v).all() or (k.endswith("std") and not (v > 0).all()):
            raise PreflightError(f"descriptor norm stats {k} has non-finite or zero-std columns")


@torch.no_grad()
def check_encoder(encoder, smiles, what, batch_size=32):
    """Every molecule gets a finite, non-zero embedding (and at least one valid token for token encoders)."""
    was_training = encoder.training
    encoder.eval()
    bad = []
    try:
        with preserved_rng():
            for i in range(0, len(smiles), batch_size):
                batch = smiles[i:i + batch_size]
                out = encoder.encode(batch)
                if encoder.is_sequence_capable:
                    tok, pooled, mask = out
                    tok_ok = torch.isfinite(tok).flatten(1).all(dim=1).cpu()
                    has_token = (~mask).any(dim=1).cpu()
                else:
                    pooled = out
                    tok_ok = has_token = torch.ones(len(batch), dtype=torch.bool)
                pooled = pooled.float().cpu()
                ok = torch.isfinite(pooled).all(dim=1) & (pooled.abs().sum(dim=1) > 0) & tok_ok & has_token
                bad += [s for s, good in zip(batch, ok.tolist()) if not good]
    finally:
        encoder.train(was_training)
    if bad:
        raise PreflightError(f"{what}: {len(bad)} molecules get a zero / non-finite / empty encoding, e.g. {bad[:3]}")
    return len(smiles)


def check_dataset(ds, what, flags_must_be_one=()):
    """Every item tensor is finite; the named availability flags are 1 for every row."""
    with preserved_rng():
        for i in range(len(ds)):
            item = ds[i]
            for k, v in item.items():
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                    raise PreflightError(f"{what}: non-finite {k} in row {i}")
            for k in flags_must_be_one:
                if float(item[k]) != 1.0:
                    raise PreflightError(f"{what}: {k} = {float(item[k])} in row {i} (vector lookup failed)")
    return len(ds)


def verify_restored(ckpt_path, val_probs, val_labels, tol=1e-4):
    """The scored model is the saved best checkpoint of this run: its val PR-AUC matches the stored one."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stored = float(ckpt["val_pr_auc"])
    now = float(average_precision_score(val_labels.astype(int), val_probs))
    if abs(stored - now) > tol:
        raise PreflightError(f"restored model val PR-AUC {now:.6f} != checkpoint {stored:.6f} ({ckpt_path})")
    return int(ckpt["epoch"]), stored
