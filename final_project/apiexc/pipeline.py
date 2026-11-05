"""The shared recipe of the Morgan-PGB model: data splits, training config, data loaders, checkpoint
loading and scoring.

run.py and every script in scripts/ build the model and its inputs through these functions, so
training, evaluation, ablations, SHAP and the benchmark all use exactly the same steps.
"""

import hashlib
import json
import os
import pickle
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from apiexc import paths
from apiexc.config import Config
from apiexc.dataset import PGBDataset, PGBNormStats, pgb_collate_fn
from apiexc.evaluation.metrics import collect_predictions
from apiexc.features.excipient_prior import ExcipientPriorTable
from apiexc.models.encoders import ENCODER_REGISTRY
from apiexc.models.pgb_model import pgb_model_from_config

SPLIT_DESCRIPTIONS = {
    "api_cluster": "API group split (80/20, Butina API clusters at Tanimoto >= 0.85, GroupShuffleSplit random_state=42)",
    "random": "random split (80/20, stratified on Outcome1, random_state=42)",
    "excipient_new": "excipient group split (80/20, parent-compound excipient groups, label-stratified)",
    "api_excipient_new": "API + excipient group split (val API and excipient both unseen in train)",
    "api_cluster06": "API group split at Tanimoto >= 0.6 (80/20, Butina API clusters incl. close analogues)",
}
SPLIT_FILE = "split.txt"   # written into each new checkpoint folder: the name of the split it was trained on


# --------------------------------------------------------------------------- data splits
@dataclass(frozen=True)
class DataSplit:
    """A train / val split folder. Every split is tested on the same 22 held-out pairs."""

    name: str
    folder: Path

    @property
    def train_csv(self) -> Path:
        return self.folder / "train.csv"

    @property
    def val_csv(self) -> Path:
        return self.folder / "val.csv"

    @property
    def heldout_csv(self) -> Path:
        return paths.HELDOUT_CSV

    @property
    def description(self) -> str:
        return SPLIT_DESCRIPTIONS.get(self.name, self.name)

    def csv(self, which: str) -> Path:
        return {"train": self.train_csv, "val": self.val_csv, "heldout": self.heldout_csv}[which]

    def frame(self, which: str) -> pd.DataFrame:
        """train / val / heldout as a DataFrame, with the label in Outcome1 (as PGBDataset expects)."""
        return pd.read_csv(self.csv(which)).rename(columns={"ground_truth": "Outcome1"})


def get_split(name_or_path: Optional[str] = None) -> DataSplit:
    """A split by name (a folder in data/splits/, e.g. "random") or by path to a folder with train.csv and val.csv."""
    if not name_or_path:
        name_or_path = paths.DEFAULT_SPLIT
    folder = paths.split_dir(name_or_path)
    if not (folder / "train.csv").exists():
        folder = paths.resolve(name_or_path)
    for f in ("train.csv", "val.csv"):
        if not (folder / f).exists():
            raise SystemExit(f"{folder / f} not found. Known splits: {', '.join(paths.available_splits())}")
    folder = folder.resolve()
    name = folder.name if folder.parent == paths.SPLITS.resolve() else paths.rel(folder)
    return DataSplit(name, folder)


def checkpoint_split(ckpt_dir) -> DataSplit:
    """The split a checkpoint was trained on (its split.txt; the shipped model has none: the default split)."""
    f = Path(ckpt_dir) / SPLIT_FILE
    return get_split(f.read_text().strip() if f.exists() else None)


def heldout_overlap(split: DataSplit) -> dict:
    """Held-out pairs whose (API_CID, Excipient_CID) also occurs in train or val. Both must be empty: the
    held-out pairs are an external test set (the 2 pairs that were also in train were removed)."""
    heldout = split.frame("heldout")
    pairs = set(zip(heldout["API_CID"], heldout["Excipient_CID"]))
    out = {}
    for which in ("train", "val"):
        df = split.frame(which)
        out[which] = sorted(pairs & set(zip(df["API_CID"], df["Excipient_CID"])))
    return out


def check_heldout_unseen(split: DataSplit) -> None:
    overlap = heldout_overlap(split)
    for which, hits in overlap.items():
        if hits:
            raise SystemExit(f"{len(hits)} held-out pair(s) (API_CID, Excipient_CID) also occur in {split.name}/{which}.csv: "
                             f"{hits}. The held-out set must not overlap the training data.")


# --------------------------------------------------------------------------- config, encoder, loaders
def make_config(split: DataSplit, seed: int = 42, device: str = "auto", epochs: Optional[int] = None) -> Config:
    """Training config of the Morgan-PGB model: Config() defaults + the Morgan fixed-vector family +
    the PGB recipe overrides (asymmetric focal loss, no balanced sampler, prior from the train positive rate)."""
    config = Config()
    config.use_pgb_head = True
    config.split_type = "cluster"
    config.seed = seed
    config.encoder = "fixed_vector"
    config.fixed_vector_source = "morgan"
    config.fusion = "concat"
    config.encoder_output_dim = 1024   # metadata only on the concat path
    config.pgb_max_epochs = epochs or 40
    config.device = device
    config.resolve_paths()
    config.train_csv, config.val_csv, config.test_csv = paths.rel(split.train_csv), paths.rel(split.val_csv), paths.rel(split.heldout_csv)
    config.positive_prior = float(pd.read_csv(split.train_csv)["Outcome1"].mean())
    config.use_balanced_sampler = False
    config.loss = "asym_focal"
    return config


def build_encoder(config: Config, device):
    """Morgan-PGB computes its fingerprints from SMILES in PGBDataset; the encoder holds no weights and
    only routes the model to the concat (no structural branch) path."""
    encoder = ENCODER_REGISTRY[config.encoder](source=config.fixed_vector_source, vector_path=None, device=str(device))
    encoder.to(device)
    return encoder


def make_loader(csv, config, prior_table, pgb_norm, encoder, row_priors=None, shuffle=False, dataset_cls=PGBDataset):
    ds = dataset_cls(str(csv), config.encoder, config.fixed_vector_source, prior_table, pgb_norm,
                     fixed_vector_encoder=encoder, row_priors=row_priors)
    return DataLoader(ds, batch_size=config.pgb_batch_size, shuffle=shuffle, collate_fn=pgb_collate_fn, num_workers=0)


def md5(path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# --------------------------------------------------------------------------- checkpoints
# Checkpoints saved before the code moved from src/ into apiexc/ pickle their config as src.config.Config.
_RENAMED_MODULES = {"src.config": "apiexc.config"}


class _RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(_RENAMED_MODULES.get(module, module), name)


_pickle_module = types.ModuleType("apiexc_checkpoint_pickle")
_pickle_module.__dict__.update({k: getattr(pickle, k) for k in ("load", "loads", "dump", "dumps", "Pickler")})
_pickle_module.Unpickler = _RenamingUnpickler


def torch_load(path, device) -> dict:
    """torch.load of a best_model.pt (weights, config and training info), old or new."""
    return torch.load(path, map_location=device, weights_only=False, pickle_module=_pickle_module)


def read_threshold(ckpt_dir, default: Optional[float] = None) -> Optional[float]:
    f = Path(ckpt_dir) / "threshold.json"
    return json.loads(f.read_text())["threshold"] if f.exists() else default


@dataclass
class LoadedModel:
    """A trained checkpoint ready for scoring."""

    folder: Path
    model: torch.nn.Module
    config: Config
    encoder: torch.nn.Module
    prior_table: ExcipientPriorTable
    pgb_norm: PGBNormStats
    info: dict                                  # the rest of best_model.pt (epoch, val_pr_auc, ...)
    threshold: Optional[float]                  # threshold.json, None if missing
    device: torch.device
    dataset_cls: type = field(default=PGBDataset)

    def loader(self, csv, row_priors=None, shuffle=False):
        return make_loader(csv, self.config, self.prior_table, self.pgb_norm, self.encoder, row_priors, shuffle,
                           self.dataset_cls)

    def score(self, df: pd.DataFrame):
        """(probabilities of incompatible, labels) for a DataFrame with API_Smiles, Excipient_Smiles, Outcome1."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.csv")
            df.to_csv(path, index=False)
            return collect_predictions(self.model, self.loader(path), self.device)


def load_checkpoint(ckpt_dir, device, build_model: Callable = pgb_model_from_config,
                    load_prior_table: Optional[Callable] = None, dataset_cls=PGBDataset) -> LoadedModel:
    """Load best_model.pt, prior_table.json, pgb_norm_stats.npz and threshold.json from a checkpoint folder.
    The ablation harness passes its own build_model / load_prior_table / dataset_cls."""
    ckpt_dir = Path(ckpt_dir)
    for f in ("best_model.pt", "prior_table.json", "pgb_norm_stats.npz"):
        if not (ckpt_dir / f).exists():
            raise SystemExit(f"{ckpt_dir / f} not found")
    device = torch.device(device)
    ckpt = torch_load(ckpt_dir / "best_model.pt", device)
    config = ckpt["config"]
    encoder = build_encoder(config, device)
    model = build_model(config, encoder)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    prior_path = str(ckpt_dir / "prior_table.json")
    prior_table = load_prior_table(config, prior_path) if load_prior_table else ExcipientPriorTable.load(prior_path)
    pgb_norm = PGBNormStats.load(str(ckpt_dir / "pgb_norm_stats.npz"))
    info = {k: v for k, v in ckpt.items() if k not in ("model_state_dict", "config")}
    return LoadedModel(ckpt_dir, model, config, encoder, prior_table, pgb_norm, info, read_threshold(ckpt_dir), device,
                       dataset_cls)
