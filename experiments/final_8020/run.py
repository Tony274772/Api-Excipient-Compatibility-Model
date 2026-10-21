"""
Train every model on the final_project 80/20 split and evaluate it on the 24 held-out pairs.

Split:  final_project/data/train.csv (2770) + val.csv (774), the split Morgan-PGB 20/24 was trained on
        (Butina API clusters, GroupShuffleSplit test_size=0.2, random_state=42). --data_dir picks another folder
        with the same files (train.csv, val.csv, held_out_test_set.csv).
Val:    early stopping, LR schedule and decision threshold. The held-out pairs are never used for any choice.
Test:   the 24 held-out pairs, also scored on two subsets:
          unseen  pairs whose CID pair is not in train.csv
          clean   pairs whose CID pair is in neither train nor val, minus the near-duplicates in NEAR_DUPLICATES

Models (models.py), each with the recipe of the script that trained it originally:
  baseline          54  CompatibilityModel runs in metrics/                (main.py); the 4 concat global-gated
                        ones use ConcatGatedPoolingModel, so none is a copy of its concat cls twin
  pgb                9  Prior-Gated Bilinear families                      (experiments/pgb)
  pgb_global_gated   2  PGB ChemBERTa / D-MPNN with global gated pooling   (experiments/pgb/global_gated)
  gb                 9  Gated Bilinear families                            (experiments/gated_bilinear), with
                        leave-own-cluster-out priors for train rows (gb_prior.py)

Every run first passes the checks in checks.py (no molecule, descriptor or vector silently replaced by zeros),
starts from empty output folders, and after training verifies that the scored model is this run's saved best
checkpoint. heldout_metrics.json is written last: it marks a finished run.

Outputs per model and seed (a finished run is skipped unless --overwrite):
  checkpoints/<out>/<model>/seed_<s>/best_model.pt  (+ descriptor_norm_stats.json | prior_table.json, pgb_norm_stats.npz)
  metrics/<out>/<model>/seed_<s>/heldout_predictions.csv, training_history.json, val_metrics.json, run_info.json,
                                 heldout_metrics.json
  metrics/<out>/summary.csv (+ summary_by_model.csv with several seeds), failures.log

Usage (from the project root):
  python -m experiments.final_8020.run --list
  python -m experiments.final_8020.run --models pgb_fixed_vector_morgan_concat_asl
  python -m experiments.final_8020.run --models all
  python -m experiments.final_8020.run --models all --seeds 42 43 44
  python -m experiments.final_8020.run --summary
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from experiments.final_8020.checks import (check_dataset, check_descriptors, check_encoder, check_norm_stats,
                                           check_split, preserved_rng, unique_smiles, verify_restored)
from experiments.final_8020.models import (ALL_MODELS, BASELINE_MODELS, CONCAT_GATED_POOLING, GB_MODELS, GROUPS,
                                           PGB_GLOBAL_GATED_MODELS, PGB_MODELS, assert_no_duplicate_models,
                                           baseline_architecture, kind_of)
from experiments.split_8020.vectors import FIXED_VECTOR_FILES, fill_heldout_vectors
from src.config import Config
from src.dataset import CompatibilityDataset, collate_fn
from src.evaluate import collect_predictions, compute_metrics, save_metrics, tune_threshold
from src.utils import get_device, seed_everything

RDLogger.DisableLog("rdApp.*")

HELDOUT_API_DESC = "held_out_testset/api_descriptors.csv"
HELDOUT_EXC_DESC = "held_out_testset/excipient_descriptors.csv"
API_DESC = "data/api_descriptors.csv"
EXC_DESC = "data/excipient_descriptors.csv"

# Held-out pairs whose chemistry is in val under another excipient CID (excluded from the "clean" subset).
NEAR_DUPLICATES = {
    (54675776, 21226083): "Tetracycline + calcium phosphate (val has Tetracycline + CID 24441 / 24456)",
    (5284583, 165411828): "Levetiracetam + talc (val has Levetiracetam + talc CID 26924)",
}
RUN_FILES = ["heldout_metrics.json", "val_metrics.json", "run_info.json", "training_history.json", "heldout_predictions.csv"]


@dataclass
class Paths:
    data_dir: str
    out: str

    @property
    def train_csv(self): return os.path.join(self.data_dir, "train.csv")
    @property
    def val_csv(self): return os.path.join(self.data_dir, "val.csv")
    @property
    def heldout_csv(self): return os.path.join(self.data_dir, "held_out_test_set.csv")
    @property
    def ckpt_root(self): return os.path.join("checkpoints", self.out)
    @property
    def metrics_root(self): return os.path.join("metrics", self.out)


P = Paths("final_project/data", "final_8020")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments", "main.py"], text=True).strip())
        return commit + ("+uncommitted-changes" if dirty else "")
    except Exception:
        return "unknown"


def run_dirs(name, seed):
    return os.path.join(P.ckpt_root, name, f"seed_{seed}"), os.path.join(P.metrics_root, name, f"seed_{seed}")


def is_finished(name, seed):
    ckpt_dir, metrics_dir = run_dirs(name, seed)
    return (all(os.path.exists(os.path.join(metrics_dir, f)) for f in RUN_FILES)
            and os.path.exists(os.path.join(ckpt_dir, "best_model.pt")))


def fresh_run_dirs(name, seed):
    """Empty checkpoint and metrics folders for this model + seed, so nothing from an earlier run is left."""
    dirs = run_dirs(name, seed)
    for d, root in zip(dirs, [P.ckpt_root, P.metrics_root]):
        d_abs, root_abs = os.path.abspath(d), os.path.abspath(root)
        if os.path.commonpath([d_abs, root_abs]) != root_abs or d_abs == root_abs:
            raise RuntimeError(f"refusing to clear {d} (outside {root})")
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    return dirs


def load_frames():
    ho = pd.read_csv(P.heldout_csv)
    ho = ho.rename(columns={"ground_truth": "Outcome1"}) if "ground_truth" in ho.columns else ho
    return pd.read_csv(P.train_csv), pd.read_csv(P.val_csv), ho


def heldout_subsets(ho):
    """Boolean masks over the held-out rows: unseen (not in train) and clean (not in train/val, no near-duplicate)."""
    pair = lambda df: set(zip(df["API_CID"].astype(int), df["Excipient_CID"].astype(int)))
    train_df, val_df, _ = load_frames()
    train_pairs, val_pairs = pair(train_df), pair(val_df)
    keys = list(zip(ho["API_CID"].astype(int), ho["Excipient_CID"].astype(int)))
    unseen = np.array([k not in train_pairs for k in keys])
    clean = np.array([k not in train_pairs and k not in val_pairs and k not in NEAR_DUPLICATES for k in keys])
    return {"unseen": unseen, "clean": clean}


def train_with_clusters(train_df):
    """Train rows with an API cluster_id: the split's own column, else Butina on the train APIs (pgb_single_run)."""
    if "cluster_id" in train_df.columns:
        if train_df["cluster_id"].isna().any():
            raise RuntimeError("train.csv has missing cluster_id values")
        return train_df, "cluster_id column of train.csv"
    from src.cross_validate import butina_cluster
    df = train_df.copy()
    df["cluster_id"] = df["API_Smiles"].map(butina_cluster(train_df["API_Smiles"].unique().tolist()))
    return df, "Butina on train APIs"


def metric_suite(probs, labels, threshold):
    m = compute_metrics(probs, labels, threshold)
    labels = labels.astype(int)
    m["roc_auc"] = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan")
    cm = np.zeros((2, 2), dtype=int)
    for y, yhat in zip(labels, (probs >= threshold).astype(int)):
        cm[y, yhat] += 1
    m["confusion_matrix"] = cm.tolist()
    m["tn"], m["fp"], m["fn"], m["tp"] = (int(x) for x in cm.ravel())
    m["n"] = int(len(labels))
    m["correct"] = int(m["tp"] + m["tn"])
    return m


def heldout_metric_suite(ho, probs, labels, threshold):
    if not np.array_equal(labels.astype(int), ho["Outcome1"].astype(int).values):
        raise RuntimeError("held-out predictions are not in held-out row order")
    m = metric_suite(probs, labels, threshold)
    m["subsets"] = {k: metric_suite(probs[mask], labels[mask], threshold) for k, mask in heldout_subsets(ho).items()}
    return m


def save_outputs(metrics_dir, val_metrics, heldout_metrics, history, heldout_df, ho_probs, threshold, info):
    """Writes every file; heldout_metrics.json last, so it only exists for a complete run."""
    cols = [c for c in ["Api_name", "Excipient_name", "API_CID", "Excipient_CID", "API_Smiles", "Excipient_Smiles"] if c in heldout_df]
    pred = heldout_df[cols].copy()
    pred["label"] = heldout_df["Outcome1"].astype(int).values
    pred["probability"] = ho_probs
    pred["prediction"] = (ho_probs >= threshold).astype(int)
    pred["correct"] = (pred["prediction"] == pred["label"]).astype(int)
    for k, mask in heldout_subsets(heldout_df).items():
        pred[f"in_{k}"] = mask.astype(int)
    pred.to_csv(os.path.join(metrics_dir, "heldout_predictions.csv"), index=False)
    save_metrics({"history": history}, os.path.join(metrics_dir, "training_history.json"))
    save_metrics(val_metrics, os.path.join(metrics_dir, "val_metrics.json"))
    with open(os.path.join(metrics_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    save_metrics(heldout_metrics, os.path.join(metrics_dir, "heldout_metrics.json"))


def base_info(name, seed, device, history, best_epoch, best_val_pr_auc, threshold, threshold_rule, checks, t0):
    return {
        "model": name, "kind": kind_of(name), "seed": seed, "device": str(device),
        "split": {"train_csv": P.train_csv, "val_csv": P.val_csv, "heldout_csv": P.heldout_csv,
                  "train_md5": _md5(P.train_csv), "val_md5": _md5(P.val_csv), "heldout_md5": _md5(P.heldout_csv)},
        "git_commit": _git_commit(),
        "epochs_trained": len(history.get("val_pr_auc", [])),
        "best_epoch": best_epoch,
        "best_val_pr_auc_checkpoint": best_val_pr_auc,
        "threshold_from_val": float(threshold),
        "threshold_rule": threshold_rule,
        "checks_passed": checks,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "minutes": round((time.time() - t0) / 60, 2),
    }


def _dataset_from_frame(ckpt_dir, df, make):
    """Build a dataset from a DataFrame via a temporary CSV inside this run's checkpoint folder."""
    path = os.path.join(ckpt_dir, "_tmp_rows.csv")
    df.to_csv(path, index=False)
    try:
        return make(path)
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# baseline CompatibilityModel (same recipe as main.py)
# ---------------------------------------------------------------------------
def _baseline_loader(config, ds, train: bool):
    sampler, shuffle = None, train
    if train and config.use_balanced_sampler:
        labels = ds.df["Outcome1"].values.astype(int)
        weights = 1.0 / np.bincount(labels)[labels]
        sampler, shuffle = WeightedRandomSampler(weights.tolist(), num_samples=len(ds), replacement=True), False
    return DataLoader(ds, batch_size=config.batch_size, shuffle=shuffle, sampler=sampler,
                      collate_fn=collate_fn, drop_last=False, num_workers=0)


def train_baseline(name, seed, device_str):
    from main import _train_with_separate_lr, build_encoder
    from experiments.final_8020.concat_pooling import ConcatGatedPoolingModel
    from src.descriptors import compute_norm_stats
    from src.model import CompatibilityModel
    from src.train import train

    t0 = time.time()
    ckpt_dir, metrics_dir = fresh_run_dirs(name, seed)
    config = Config()
    for k, v in BASELINE_MODELS[name].items():
        setattr(config, k, v)
    config.seed = seed
    if device_str:
        config.device = device_str
    # Paths are set explicitly; never call config.resolve_csv_paths() / build_dataloaders(),
    # they reset any */train.csv path back to data/train.csv.
    config.split_type = "cluster"
    config.train_csv, config.val_csv, config.test_csv = P.train_csv, P.val_csv, None
    config.api_descriptors_path, config.excipient_descriptors_path = API_DESC, EXC_DESC
    config.checkpoint_dir, config.metrics_dir = ckpt_dir, metrics_dir
    if config.encoder == "fixed_vector":
        config.fixed_vector_path = FIXED_VECTOR_FILES[config.fixed_vector_source]

    seed_everything(config.seed)
    device = get_device(config.device)
    train_df, val_df, ho = load_frames()
    checks = {}

    # descriptor normalisation stats from the train rows only (same method as data/data_split.py)
    stats = compute_norm_stats(train_df, pd.read_csv(API_DESC), pd.read_csv(EXC_DESC))
    check_norm_stats(stats)
    config.descriptor_norm_stats_path = os.path.join(ckpt_dir, "descriptor_norm_stats.json")
    with open(config.descriptor_norm_stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    check_descriptors(pd.concat([train_df, val_df]), API_DESC, EXC_DESC, "train/val")
    check_descriptors(ho, HELDOUT_API_DESC, HELDOUT_EXC_DESC, "held-out")
    checks["descriptors"] = "every train/val/held-out CID has a descriptor row without NaN"

    config.positive_prior = float(train_df["Outcome1"].mean())
    if config.encoder == "dmpnn_chemprop":
        config.chemeleon_checkpoint_path = config.dmpnn_pretrained_checkpoint
        config.chemeleon_checkpoint_md5 = "6a80b54fdb7de37ef0374d302f01e8ce"

    encoder = build_encoder(config, device)
    vec_report = None
    if config.encoder == "fixed_vector":
        with preserved_rng():
            vec_report = fill_heldout_vectors(encoder, ho)  # held-out molecules get the vector of their CID
    checks["encoder"] = f"{check_encoder(encoder, unique_smiles(train_df, val_df, ho), name)} molecules with a non-zero encoding"
    if not encoder.is_sequence_capable:
        config.fusion = "concat"

    arch = baseline_architecture(name)
    model_cls = ConcatGatedPoolingModel if name in CONCAT_GATED_POOLING else CompatibilityModel
    model = model_cls(config, encoder)
    model.init_bias(config.positive_prior)
    model.to(device)

    make = lambda api_d, exc_d: (lambda p: CompatibilityDataset(p, api_d, exc_d, config.descriptor_norm_stats_path))
    train_ds = make(API_DESC, EXC_DESC)(P.train_csv)
    val_ds = make(API_DESC, EXC_DESC)(P.val_csv)
    ho_ds = _dataset_from_frame(ckpt_dir, ho, make(HELDOUT_API_DESC, HELDOUT_EXC_DESC))
    checks["datasets"] = {k: check_dataset(ds, k) for k, ds in [("train", train_ds), ("val", val_ds), ("heldout", ho_ds)]}
    train_loader = _baseline_loader(config, train_ds, train=True)
    val_loader = _baseline_loader(config, val_ds, train=False)
    ho_loader = DataLoader(ho_ds, batch_size=len(ho_ds), shuffle=False, collate_fn=collate_fn)
    print(f"[{name} | seed {seed}] {arch['model_class']} encoder={config.encoder} fusion={arch['fusion']} "
          f"pooling={arch['pooling']} loss={config.loss} | train {len(train_ds)} val {len(val_ds)}")

    if config.encoder == "dmpnn_chemprop":
        model, history = _train_with_separate_lr(model, train_loader, val_loader, config, device)
    else:
        model, history = train(model, train_loader, val_loader, config, device)

    val_probs, val_labels = collect_predictions(model, val_loader, device)
    best_epoch, best_val = verify_restored(os.path.join(ckpt_dir, "best_model.pt"), val_probs, val_labels)
    threshold, _ = tune_threshold(val_probs, val_labels, config.threshold_step)  # same rule as main.py (best F1)
    val_metrics = metric_suite(val_probs, val_labels, threshold)
    ho_probs, ho_labels = collect_predictions(model, ho_loader, device)
    heldout_metrics = heldout_metric_suite(ho, ho_probs, ho_labels, threshold)

    info = base_info(name, seed, device, history, best_epoch, best_val, threshold, "best val F1 (main.py)", checks, t0)
    info["architecture"] = arch
    info["config"] = {k: getattr(config, k) for k in ["encoder", "fixed_vector_source", "fixed_vector_path", "fusion", "pooling",
                                                      "loss", "use_descriptors", "use_balanced_sampler", "lr", "batch_size",
                                                      "max_epochs", "early_stop_patience", "threshold_step"]}
    info["heldout_fixed_vectors"] = vec_report
    save_outputs(metrics_dir, val_metrics, heldout_metrics, history, ho, ho_probs, threshold, info)

    del model, encoder
    return val_metrics, heldout_metrics


# ---------------------------------------------------------------------------
# PGB and PGB global-gated (same recipe as src/single_run.pgb_single_run / final_project/run.py)
# ---------------------------------------------------------------------------
def train_pgb(name, seed, device_str):
    from experiments.pgb.config import PGBConfig
    from experiments.pgb.global_gated.config import GATED_FAMILIES
    from src.cross_validate import build_encoder
    from src.dataset import PGBDataset, PGBNormStats, pgb_collate_fn
    from src.model import pgb_model_from_config
    from src.pgb_prior import ExcipientPriorTable, build_leave_cluster_out_prior_vectors
    from src.train import pgb_train

    t0 = time.time()
    family = PGB_MODELS.get(name) or PGB_GLOBAL_GATED_MODELS[name]
    ckpt_dir, metrics_dir = fresh_run_dirs(name, seed)
    config = PGBConfig()
    config.data_dir = "data"
    config.split_type = "cluster"
    config.apply_family_spec(family)
    if name in PGB_GLOBAL_GATED_MODELS:
        for k, v in GATED_FAMILIES[family].items():
            setattr(config, k, v)
    config.seed = seed
    if device_str:
        config.device = device_str

    seed_everything(config.seed)
    device = get_device(config.device)
    train_df, val_df, ho = load_frames()
    checks = {}

    run_config = copy.deepcopy(config)
    run_config.train_csv, run_config.val_csv, run_config.test_csv = P.train_csv, P.val_csv, None
    run_config.checkpoint_dir, run_config.metrics_dir = ckpt_dir, metrics_dir
    run_config.positive_prior = float(train_df["Outcome1"].mean())
    run_config.use_balanced_sampler = False
    run_config.loss = "asym_focal"

    encoder = build_encoder(run_config, device)
    if not encoder.is_sequence_capable:
        run_config.fusion = "concat"
    fv_enc = encoder if run_config.encoder == "fixed_vector" else None
    fvs = getattr(run_config, "fixed_vector_source", "maccs")
    vec_report = None
    all_smiles = unique_smiles(train_df, val_df, ho)
    if fv_enc is not None and fvs in ("pubchemfp", "mol2vec"):   # maccs / morgan bits are computed from SMILES
        with preserved_rng():
            vec_report = fill_heldout_vectors(fv_enc, ho)
        checks["encoder"] = f"{check_encoder(fv_enc, all_smiles, name)} molecules with a non-zero {fvs} vector"
    elif fv_enc is None:
        checks["encoder"] = f"{check_encoder(encoder, all_smiles, name)} molecules with a non-zero encoding"
    else:
        checks["encoder"] = f"not used ({fvs} bits computed from SMILES)"

    all_train_smiles = list(set(train_df["API_Smiles"].dropna().tolist() + train_df["Excipient_Smiles"].dropna().tolist()))
    pgb_norm = PGBNormStats.from_smiles_list(all_train_smiles)
    pgb_norm.save(os.path.join(ckpt_dir, "pgb_norm_stats.npz"))

    prior_table = ExcipientPriorTable()
    prior_table.fit(train_df, leave_out_cluster_id=None)
    prior_table.save(os.path.join(ckpt_dir, "prior_table.json"))
    # leave-own-cluster-out priors for train rows (every prior component; the table is refit without the cluster)
    train_clustered, cluster_source = train_with_clusters(train_df)
    train_priors = build_leave_cluster_out_prior_vectors(train_clustered, cluster_col="cluster_id")

    make = lambda row_priors=None: (lambda p: PGBDataset(p, run_config.encoder, fvs, prior_table, pgb_norm,
                                                         fixed_vector_encoder=fv_enc, row_priors=row_priors))
    train_ds = make(train_priors)(P.train_csv)
    val_ds = make()(P.val_csv)
    ho_ds = _dataset_from_frame(ckpt_dir, ho, make())
    flags = ("api_prim_avail", "exc_prim_avail")
    checks["datasets"] = {k: check_dataset(ds, k, flags) for k, ds in [("train", train_ds), ("val", val_ds), ("heldout", ho_ds)]}
    loader = lambda ds, shuffle=False: DataLoader(ds, batch_size=run_config.pgb_batch_size, shuffle=shuffle,
                                                  collate_fn=pgb_collate_fn, num_workers=0)
    train_loader, val_loader, ho_loader = loader(train_ds, shuffle=True), loader(val_ds), loader(ho_ds)

    model = pgb_model_from_config(run_config, encoder)
    model.to(device)
    print(f"[{name} | seed {seed}] PGB family={family} encoder={run_config.encoder} fusion={run_config.fusion} "
          f"pooling={run_config.pooling if run_config.fusion == 'cross_attn' else 'n/a'} | train {len(train_df)} val {len(val_df)}")

    model, history = pgb_train(model, train_loader, val_loader, run_config, device)

    val_probs, val_labels = collect_predictions(model, val_loader, device)
    best_epoch, best_val = verify_restored(os.path.join(ckpt_dir, "best_model.pt"), val_probs, val_labels)
    tnr_floor = getattr(run_config, "pgb_tnr_floor", 0.0)
    threshold, _ = tune_threshold(val_probs, val_labels, run_config.threshold_step, tnr_floor=tnr_floor)
    val_metrics = metric_suite(val_probs, val_labels, threshold)
    ho_probs, ho_labels = collect_predictions(model, ho_loader, device)
    heldout_metrics = heldout_metric_suite(ho, ho_probs, ho_labels, threshold)

    info = base_info(name, seed, device, history, best_epoch, best_val, threshold,
                     f"best val MCC with TNR >= {tnr_floor}", checks, t0)
    info["architecture"] = {"model_class": "PGBCompatibilityModel", "family": family, "encoder": run_config.encoder,
                            "fusion": run_config.fusion,
                            "pooling": run_config.pooling if run_config.fusion == "cross_attn" else "n/a",
                            "train_prior": f"leave-own-cluster-out ({cluster_source})"}
    info["config"] = {"family": family, **{k: getattr(run_config, k, None) for k in
                      ["encoder", "fixed_vector_source", "fusion", "pooling", "loss", "pgb_lr", "pgb_encoder_lr",
                       "pgb_batch_size", "pgb_max_epochs", "pgb_early_stop_patience", "pgb_tnr_floor", "pgb_prior_vec_dim"]}}
    info["heldout_fixed_vectors"] = vec_report
    save_outputs(metrics_dir, val_metrics, heldout_metrics, history, ho, ho_probs, threshold, info)

    del model, encoder
    return val_metrics, heldout_metrics


# ---------------------------------------------------------------------------
# Gated Bilinear (same recipe as experiments/gated_bilinear/train_gb.py, plus leave-own-cluster-out priors)
# ---------------------------------------------------------------------------
GB_FAMILY_TO_ENCODER = {"maccs": "fixed_vector", "morgan": "fixed_vector", "pubchem": "fixed_vector", "mol2vec": "fixed_vector",
                        "molformer": "molformer", "chemberta": "chemberta", "gin": "pretrained_gin", "gat": "pretrained_gat",
                        "dmpnn": "dmpnn_chemprop"}
GB_VECTOR_FILES = {"pubchem": ("data/pubchem_fps.csv", "data/pubchem_fps_heldout.csv"),
                   "mol2vec": ("data/mol2vec_embeddings.csv", "data/mol2vec_embeddings_heldout.csv")}


def _gb_vectors(family, frames):
    """PubChem / Mol2vec lookup for GB families, with the held-out CIDs added from the *_heldout.csv files
    (GBCompatibilityDataset silently uses a zero vector for a molecule it cannot find)."""
    from experiments.gated_bilinear.dataset import load_precomputed_vectors
    if family not in GB_VECTOR_FILES:
        return None, None
    base, extra = GB_VECTOR_FILES[family]
    vectors = load_precomputed_vectors(base)
    for k, v in load_precomputed_vectors(extra).items():
        vectors.setdefault(k, v)
    missing = set()
    for what, df in frames.items():
        for smi_col, cid_col in [("API_Smiles", "API_CID"), ("Excipient_Smiles", "Excipient_CID")]:
            for smi, cid in zip(df[smi_col], df[cid_col]):
                v = vectors.get(str(smi), vectors.get(str(int(cid))))
                if v is None or not np.isfinite(v).all() or not v.any():
                    missing.add((what, int(cid)))
    if missing:
        raise RuntimeError(f"{len(missing)} molecules have no usable {family} vector (would silently become zeros): {sorted(missing)[:5]}")
    return (vectors, None) if family == "pubchem" else (None, vectors)


def train_gb(name, seed, device_str, original_prior=False):
    from experiments.final_8020.gb_prior import GBLeaveClusterOutDataset, build_gb_train_priors
    from experiments.gated_bilinear.config import GBConfig
    from experiments.gated_bilinear.dataset import GBCompatibilityDataset, gb_collate_fn
    from experiments.gated_bilinear.evaluate_gb import collect_gb_predictions, get_val_pr_auc_gb, tune_thresholds_gb
    from experiments.gated_bilinear.features import compute_gb_descriptor_norm_stats
    from experiments.gated_bilinear.loss import AsymmetricFocalLoss
    from experiments.gated_bilinear.model import FIXED_VECTOR_FAMILIES, GatedBilinearModel
    from experiments.gated_bilinear.prior import ExcipientPriorTable
    from experiments.gated_bilinear.train_gb import GNN_ENCODERS, build_optimizer_and_scheduler, compute_pos_weight, train_one_epoch_gb
    from src.encoders import ENCODER_REGISTRY

    t0 = time.time()
    family = GB_MODELS[name]
    ckpt_dir, metrics_dir = fresh_run_dirs(name, seed)
    config = GBConfig()
    config.gb_family = family
    config.encoder = GB_FAMILY_TO_ENCODER[family]
    config.seed = seed
    if device_str:
        config.device = device_str
    config.resolve_checkpoint_paths()
    config.train_csv, config.val_csv, config.test_csv = P.train_csv, P.val_csv, None
    config.gb_checkpoint_dir, config.gb_metrics_dir = ckpt_dir, metrics_dir
    config.gb_heldout_csv = P.heldout_csv

    seed_everything(config.seed)
    device = get_device(config.device)
    train_df, val_df, ho = load_frames()
    checks = {}

    train_exc = train_df["Excipient_Smiles"].fillna("").astype(str).tolist()
    train_api = train_df["API_Smiles"].fillna("").astype(str).tolist()
    train_labels = train_df["Outcome1"].astype(int).tolist()
    config.base_rate = float(np.mean(train_labels))

    # prior table of all train rows (val / held-out rows); train rows get leave-own-cluster-out priors
    prior_table = ExcipientPriorTable(train_exc_smiles=train_exc, train_labels=train_labels,
                                      train_api_clusters=None, global_rate=config.base_rate)
    if original_prior:
        row_priors, prior_mode = None, "original train_gb.py (train rows' priors include their own label)"
    else:
        train_clustered, cluster_source = train_with_clusters(train_df)
        row_priors = build_gb_train_priors(prior_table, train_clustered, cluster_col="cluster_id")
        prior_mode = f"leave-own-cluster-out for every prior component ({cluster_source})"
    desc_stats = compute_gb_descriptor_norm_stats(train_api + train_exc)
    pubchem_vectors, mol2vec_vectors = _gb_vectors(family, {"train": train_df, "val": val_df, "heldout": ho})
    morgan_bits = 1024 if family == "morgan" else 512
    ds_kwargs = dict(prior_table=prior_table, descriptor_norm_stats=desc_stats, family=family, morgan_bits=morgan_bits,
                     pubchem_vectors=pubchem_vectors, mol2vec_vectors=mol2vec_vectors)

    if row_priors is None:
        train_ds = GBCompatibilityDataset(csv_path=P.train_csv, is_train=True, **ds_kwargs)
    else:
        train_ds = GBLeaveClusterOutDataset(csv_path=P.train_csv, is_train=True, row_priors=row_priors,
                                            expected_df=train_df, **ds_kwargs)
    val_ds = GBCompatibilityDataset(csv_path=P.val_csv, is_train=False, **ds_kwargs)
    ho_ds = _dataset_from_frame(ckpt_dir, ho, lambda p: GBCompatibilityDataset(csv_path=p, is_train=False, **ds_kwargs))
    loader = lambda ds, shuffle=False, bs=None: DataLoader(ds, batch_size=bs or config.gb_batch_size, shuffle=shuffle,
                                                           collate_fn=gb_collate_fn, drop_last=False, num_workers=0)
    train_loader, val_loader, ho_loader = loader(train_ds, shuffle=True), loader(val_ds), loader(ho_ds, bs=len(ho_ds))

    encoder_cls = ENCODER_REGISTRY[config.encoder]
    if config.encoder == "molformer":
        encoder = encoder_cls(config.molformer_model_path, device=str(device))
    elif config.encoder == "pretrained_gin":
        encoder = encoder_cls(config.gin_pretrained_name, device=str(device))
    elif config.encoder == "chemberta":
        encoder = encoder_cls(config.chemberta_model_path, device=str(device))
    elif config.encoder == "fixed_vector":
        encoder = encoder_cls(source="mol2vec", device=str(device))  # never called: GB fixed families read their own features
    elif config.encoder in GNN_ENCODERS:
        encoder = encoder_cls(config, device=str(device))
    else:
        raise ValueError(f"Unknown encoder: {config.encoder}")
    encoder.to(device)
    if family in FIXED_VECTOR_FAMILIES:
        checks["encoder"] = "not used (fixed-vector family: features computed / looked up per molecule, checked above)"
    else:
        checks["encoder"] = f"{check_encoder(encoder, unique_smiles(train_df, val_df, ho), name)} molecules with a non-zero encoding"
    checks["datasets"] = {k: check_dataset(ds, k) for k, ds in [("train", train_ds), ("val", val_ds), ("heldout", ho_ds)]}

    model = GatedBilinearModel(config, encoder, family)
    model.to(device)
    print(f"[{name} | seed {seed}] Gated Bilinear family={family} encoder={config.encoder} | prior: {prior_mode} "
          f"| train {len(train_df)} val {len(val_df)}")

    loss_fn = AsymmetricFocalLoss(gamma_pos=config.gb_afl_gamma_pos, gamma_neg=config.gb_afl_gamma_neg,
                                  pos_weight=compute_pos_weight(train_df))
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    max_epochs = config.gb_max_epochs_fixed if family in FIXED_VECTOR_FAMILIES else config.gb_max_epochs_deep

    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")
    best_pr_auc, patience = -1.0, 0
    history = {"train_loss": [], "val_pr_auc": [], "lr": []}
    for epoch in range(1, max_epochs + 1):
        te = time.time()
        avg_loss = train_one_epoch_gb(model, train_loader, optimizer, loss_fn, device, config)
        val_pr_auc = get_val_pr_auc_gb(model, val_loader, device)
        scheduler.step(val_pr_auc)
        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["lr"].append(lr)
        print(f"Epoch {epoch:3d}/{max_epochs} | Loss: {avg_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | LR: {lr:.2e} | {time.time() - te:.1f}s")
        if val_pr_auc > best_pr_auc + config.gb_early_stop_min_delta:
            best_pr_auc, patience = val_pr_auc, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_pr_auc": val_pr_auc,
                        "config": config}, best_ckpt)
        else:
            patience += 1
            if patience >= config.gb_early_stop_patience:
                print(f"Early stopping at epoch {epoch} (patience={config.gb_early_stop_patience})")
                break
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=False)["model_state_dict"])

    val_probs, val_labels = collect_gb_predictions(model, val_loader, device)
    best_epoch, best_val = verify_restored(best_ckpt, val_probs, val_labels)
    print(f"Restored best model from epoch {best_epoch} (val PR-AUC {best_val:.4f})")
    threshold, screen = tune_thresholds_gb(val_probs, val_labels, tnr_floor=config.gb_tnr_floor,
                                           min_recall=config.gb_screen_min_recall)
    val_metrics = metric_suite(val_probs, val_labels, threshold)
    val_metrics["screen_threshold"] = float(screen)
    ho_probs, ho_labels = collect_gb_predictions(model, ho_loader, device)
    heldout_metrics = heldout_metric_suite(ho, ho_probs, ho_labels, threshold)

    info = base_info(name, seed, device, history, best_epoch, best_val, threshold,
                     f"best val MCC with TNR >= {config.gb_tnr_floor}", checks, t0)
    info["architecture"] = {"model_class": "GatedBilinearModel", "family": family, "encoder": config.encoder,
                            "train_prior": prior_mode}
    info["config"] = {"family": family, "encoder": config.encoder,
                      **{k: getattr(config, k) for k in ["gb_lr_fixed", "gb_lr_new", "gb_lr_kept", "gb_batch_size",
                                                         "gb_max_epochs_fixed", "gb_max_epochs_deep", "gb_early_stop_patience",
                                                         "gb_tnr_floor", "gb_afl_gamma_pos", "gb_afl_gamma_neg"]}}
    save_outputs(metrics_dir, val_metrics, heldout_metrics, history, ho, ho_probs, threshold, info)

    del model, encoder
    return val_metrics, heldout_metrics


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def summarize():
    rows = []
    root = P.metrics_root
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if name not in ALL_MODELS:
            continue
        for sd in sorted(os.listdir(os.path.join(root, name))):
            seed = sd.replace("seed_", "")
            if not (sd.startswith("seed_") and seed.lstrip("-").isdigit() and is_finished(name, int(seed))):
                continue
            mdir = os.path.join(root, name, sd)
            ho = json.load(open(os.path.join(mdir, "heldout_metrics.json")))
            va = json.load(open(os.path.join(mdir, "val_metrics.json")))
            sub = ho["subsets"]
            rows.append({"model": name, "kind": kind_of(name), "seed": seed,
                         "heldout_correct": ho["correct"], "heldout_TP": ho["tp"], "heldout_TN": ho["tn"],
                         "heldout_FP": ho["fp"], "heldout_FN": ho["fn"],
                         "unseen_correct": f"{sub['unseen']['correct']}/{sub['unseen']['n']}",
                         "clean_correct": f"{sub['clean']['correct']}/{sub['clean']['n']}",
                         "heldout_pr_auc": ho["pr_auc"], "heldout_roc_auc": ho["roc_auc"], "heldout_mcc": ho["mcc"],
                         "val_pr_auc": va["pr_auc"], "val_roc_auc": va["roc_auc"], "val_mcc": va["mcc"], "val_f1": va["f1"],
                         "threshold": va["threshold"]})
    if not rows:
        print(f"No finished runs under {root}/")
        return
    df = pd.DataFrame(rows).sort_values(["heldout_correct", "val_pr_auc"], ascending=False)
    df.to_csv(os.path.join(root, "summary.csv"), index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 500)
    print(df.round(4).to_string(index=False))
    if df["seed"].nunique() > 1:
        agg = df.groupby("model").agg(seeds=("seed", "count"), heldout_correct_mean=("heldout_correct", "mean"),
                                      heldout_correct_std=("heldout_correct", "std"), val_pr_auc_mean=("val_pr_auc", "mean"),
                                      val_pr_auc_std=("val_pr_auc", "std"), heldout_pr_auc_mean=("heldout_pr_auc", "mean"))
        agg = agg.sort_values(["heldout_correct_mean", "val_pr_auc_mean"], ascending=False)
        agg.to_csv(os.path.join(root, "summary_by_model.csv"))
        print("\n" + agg.round(4).to_string())
    print(f"\nSaved {os.path.join(root, 'summary.csv')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def resolve_models(selection):
    out = []
    for s in selection:
        if s == "all":
            out += ALL_MODELS
        elif s in GROUPS:
            out += GROUPS[s]
        elif s in ALL_MODELS:
            out.append(s)
        else:
            raise SystemExit(f"Unknown model or group '{s}'. See --list.")
    return list(dict.fromkeys(out))


def main():
    global P
    p = argparse.ArgumentParser(description="Train models on the final_project 80/20 split and evaluate on the 24 held-out pairs.")
    p.add_argument("--models", nargs="+", default=None, help="model names, or groups: all | " + " | ".join(GROUPS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--data_dir", default="final_project/data", help="folder with train.csv, val.csv, held_out_test_set.csv")
    p.add_argument("--out", default="final_8020", help="output name: checkpoints/<out>/ and metrics/<out>/")
    p.add_argument("--gb_original_prior", action="store_true",
                   help="Gated Bilinear: the original train_gb.py priors, which include each train row's own label")
    p.add_argument("--overwrite", action="store_true", help="retrain finished runs (their folders are cleared first)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()
    P = Paths(args.data_dir, args.out)
    assert_no_duplicate_models()

    if args.list:
        for group, names in GROUPS.items():
            print(f"\n{group} ({len(names)})")
            for m in names:
                if group == "baseline":
                    a = baseline_architecture(m)
                    print(f"  {m:50s} {a['model_class']:24s} fusion={a['fusion']:10s} pooling={a['pooling']}")
                else:
                    print("  " + m)
        print(f"\ntotal {len(ALL_MODELS)} models, no two baseline models build the same network")
        return
    if args.summary:
        summarize()
        return
    if not args.models:
        raise SystemExit("Give --models (names or groups), --list or --summary.")

    for path in [P.train_csv, P.val_csv, P.heldout_csv]:
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found (--data_dir must hold train.csv, val.csv and held_out_test_set.csv)")
    check_split(*load_frames())
    models = resolve_models(args.models)
    print(f"{len(models)} model(s) x {len(args.seeds)} seed(s) on {P.data_dir} -> {P.metrics_root}/")
    failures = []
    for seed in args.seeds:
        for name in models:
            if is_finished(name, seed) and not args.overwrite:
                print(f"skip {name} seed {seed} (finished; --overwrite to retrain)")
                continue
            print(f"\n{'=' * 70}\n{name} | seed {seed}\n{'=' * 70}")
            try:
                kind = kind_of(name)
                if kind == "baseline":
                    val_m, ho_m = train_baseline(name, seed, args.device)
                elif kind in ("pgb", "pgb_global_gated"):
                    val_m, ho_m = train_pgb(name, seed, args.device)
                else:
                    val_m, ho_m = train_gb(name, seed, args.device, original_prior=args.gb_original_prior)
                s = ho_m["subsets"]
                print(f"[{name} | seed {seed}] val PR-AUC {val_m['pr_auc']:.4f} | held-out {ho_m['correct']}/{ho_m['n']} "
                      f"(TP {ho_m['tp']}, TN {ho_m['tn']}) PR-AUC {ho_m['pr_auc']:.4f} | "
                      f"unseen {s['unseen']['correct']}/{s['unseen']['n']} | clean {s['clean']['correct']}/{s['clean']['n']}")
            except Exception as e:
                failures.append((name, seed, repr(e)))
                os.makedirs(P.metrics_root, exist_ok=True)
                with open(os.path.join(P.metrics_root, "failures.log"), "a") as f:
                    f.write(f"{datetime.now().isoformat(timespec='seconds')} {name} seed {seed}\n{traceback.format_exc()}\n")
                print(f"FAILED {name} seed {seed}: {e!r} (traceback in {P.metrics_root}/failures.log)")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for f in failures:
            print("  ", *f)
    summarize()


if __name__ == "__main__":
    main()
