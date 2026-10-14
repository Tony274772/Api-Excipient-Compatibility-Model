"""
Train models on the 80/20 split (data/split_8020) and evaluate them on the 24 held-out pairs.

Split: data/data_split.py --mode 8020 (same API-cluster grouping as the 60/20/20 split;
       val = the 60/20/20 test fold; rows matching a held-out pair removed).
Val:   early stopping, LR schedule and decision threshold (never the held-out set).
Test:  the 24 held-out pairs (held_out_testset/held_out_test_set.csv).

Models (see models.py):
  - 54 baseline CompatibilityModel runs from metrics/ (same recipe as main.py)
  - 9 PGB families (same recipe as experiments/pgb)

Outputs per model and seed (an existing run is skipped unless --overwrite):
  checkpoints/split_8020/<model>/seed_<s>/best_model.pt   (+ prior_table.json, pgb_norm_stats.npz for PGB)
  metrics/split_8020/<model>/seed_<s>/
      val_metrics.json, heldout_metrics.json, training_history.json,
      heldout_predictions.csv, run_info.json

Usage (from the project root):
  python data/data_split.py --mode 8020                                  # once, builds data/split_8020/
  python -m experiments.split_8020.run --list
  python -m experiments.split_8020.run --models fixed_vector_morgan_concat_asl
  python -m experiments.split_8020.run --models all --skip_duplicates
  python -m experiments.split_8020.run --models pgb --seeds 42 43 44
  python -m experiments.split_8020.run --summary
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from experiments.split_8020.models import BASELINE_MODELS, DUPLICATE_OF, PGB_MODELS, ALL_MODELS
from experiments.split_8020.vectors import FIXED_VECTOR_FILES, assert_coverage, fill_heldout_vectors
from src.config import Config
from src.dataset import CompatibilityDataset, collate_fn
from src.evaluate import collect_predictions, compute_metrics, tune_threshold, save_metrics
from src.utils import seed_everything, get_device

SPLIT_DIR = "data/split_8020"
TRAIN_CSV = f"{SPLIT_DIR}/train.csv"
VAL_CSV = f"{SPLIT_DIR}/val.csv"
NORM_STATS = f"{SPLIT_DIR}/descriptor_norm_stats.json"
HELDOUT_CSV = "held_out_testset/held_out_test_set.csv"
HELDOUT_API_DESC = "held_out_testset/api_descriptors.csv"
HELDOUT_EXC_DESC = "held_out_testset/excipient_descriptors.csv"
CKPT_ROOT = "checkpoints/split_8020"
METRICS_ROOT = "metrics/split_8020"


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
    return os.path.join(CKPT_ROOT, name, f"seed_{seed}"), os.path.join(METRICS_ROOT, name, f"seed_{seed}")


def check_split_files():
    for p in [TRAIN_CSV, VAL_CSV, NORM_STATS]:
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found. Build the split first:  python data/data_split.py --mode 8020")


def heldout_frame():
    """Held-out pairs with an Outcome1 column, as the dataset classes expect."""
    ho = pd.read_csv(HELDOUT_CSV)
    return ho.rename(columns={"ground_truth": "Outcome1"}) if "ground_truth" in ho.columns else ho


def metric_suite(probs, labels, threshold):
    m = compute_metrics(probs, labels, threshold)
    labels = labels.astype(int)
    m["roc_auc"] = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan")
    cm = m["confusion_matrix"]
    m["tn"], m["fp"], m["fn"], m["tp"] = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    m["n"] = int(len(labels))
    m["correct"] = int(m["tp"] + m["tn"])
    return m


def save_outputs(metrics_dir, val_metrics, heldout_metrics, history, heldout_df, ho_probs, threshold, info):
    os.makedirs(metrics_dir, exist_ok=True)
    save_metrics(val_metrics, os.path.join(metrics_dir, "val_metrics.json"))
    save_metrics(heldout_metrics, os.path.join(metrics_dir, "heldout_metrics.json"))
    save_metrics({"history": history}, os.path.join(metrics_dir, "training_history.json"))
    cols = [c for c in ["Api_name", "Excipient_name", "API_CID", "Excipient_CID", "API_Smiles", "Excipient_Smiles"] if c in heldout_df]
    pred = heldout_df[cols].copy()
    pred["label"] = heldout_df["Outcome1"].astype(int).values
    pred["probability"] = ho_probs
    pred["prediction"] = (ho_probs >= threshold).astype(int)
    pred["correct"] = (pred["prediction"] == pred["label"]).astype(int)
    pred.to_csv(os.path.join(metrics_dir, "heldout_predictions.csv"), index=False)
    with open(os.path.join(metrics_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)


def base_info(name, kind, seed, device, history, threshold, t0):
    return {
        "model": name, "kind": kind, "seed": seed, "device": str(device),
        "split": {"train_csv": TRAIN_CSV, "val_csv": VAL_CSV, "train_md5": _md5(TRAIN_CSV), "val_md5": _md5(VAL_CSV),
                  "heldout_csv": HELDOUT_CSV, "heldout_md5": _md5(HELDOUT_CSV)},
        "git_commit": _git_commit(),
        "epochs_trained": len(history.get("val_pr_auc", [])),
        "best_epoch": int(np.argmax(history["val_pr_auc"])) + 1 if history.get("val_pr_auc") else None,
        "threshold_from_val": float(threshold),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "minutes": round((time.time() - t0) / 60, 2),
    }


# ---------------------------------------------------------------------------
# baseline CompatibilityModel (same recipe as main.py)
# ---------------------------------------------------------------------------
def _baseline_loader(config, csv_path, train: bool):
    ds = CompatibilityDataset(csv_path, config.api_descriptors_path, config.excipient_descriptors_path,
                              config.descriptor_norm_stats_path)
    sampler, shuffle = None, train
    if train and config.use_balanced_sampler:
        labels = ds.df["Outcome1"].values.astype(int)
        weights = 1.0 / np.bincount(labels)[labels]
        sampler, shuffle = WeightedRandomSampler(weights.tolist(), num_samples=len(ds), replacement=True), False
    return DataLoader(ds, batch_size=config.batch_size, shuffle=shuffle, sampler=sampler,
                      collate_fn=collate_fn, drop_last=False, num_workers=0)


def train_baseline(name, seed, device_str):
    from main import build_encoder, _train_with_separate_lr
    from src.model import CompatibilityModel
    from src.train import train

    t0 = time.time()
    ckpt_dir, metrics_dir = run_dirs(name, seed)
    config = Config()
    for k, v in BASELINE_MODELS[name].items():
        setattr(config, k, v)
    config.seed = seed
    if device_str:
        config.device = device_str
    # Paths are set explicitly; never call config.resolve_csv_paths() / build_dataloaders(),
    # they reset any */train.csv path back to data/train.csv.
    config.split_type = "cluster"
    config.train_csv, config.val_csv, config.test_csv = TRAIN_CSV, VAL_CSV, None
    config.descriptor_norm_stats_path = NORM_STATS
    config.checkpoint_dir, config.metrics_dir = ckpt_dir, metrics_dir
    if config.encoder == "fixed_vector":
        config.fixed_vector_path = FIXED_VECTOR_FILES[config.fixed_vector_source]

    seed_everything(config.seed)
    device = get_device(config.device)
    train_df, val_df = pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)
    config.positive_prior = float(train_df["Outcome1"].mean())
    if config.encoder == "dmpnn_chemprop":
        config.chemeleon_checkpoint_path = config.dmpnn_pretrained_checkpoint
        config.chemeleon_checkpoint_md5 = "6a80b54fdb7de37ef0374d302f01e8ce"

    encoder = build_encoder(config, device)
    if config.encoder == "fixed_vector":
        assert_coverage(encoder, pd.concat([train_df.API_Smiles, train_df.Excipient_Smiles,
                                            val_df.API_Smiles, val_df.Excipient_Smiles]).dropna().astype(str).tolist(), "train/val")
    if not encoder.is_sequence_capable:
        config.fusion = "concat"

    model = CompatibilityModel(config, encoder)
    model.init_bias(config.positive_prior)
    model.to(device)

    train_loader = _baseline_loader(config, TRAIN_CSV, train=True)
    val_loader = _baseline_loader(config, VAL_CSV, train=False)
    print(f"[{name} | seed {seed}] encoder={config.encoder} fusion={config.fusion} pooling={config.pooling} "
          f"loss={config.loss} | train {len(train_loader.dataset)} val {len(val_loader.dataset)}")

    if config.encoder == "dmpnn_chemprop":
        model, history = _train_with_separate_lr(model, train_loader, val_loader, config, device)
    else:
        model, history = train(model, train_loader, val_loader, config, device)

    val_probs, val_labels = collect_predictions(model, val_loader, device)
    threshold, _ = tune_threshold(val_probs, val_labels, config.threshold_step)  # same rule as main.py
    val_metrics = metric_suite(val_probs, val_labels, threshold)

    ho = heldout_frame()
    vec_report = fill_heldout_vectors(encoder, ho) if config.encoder == "fixed_vector" else None
    tmp = os.path.join(ckpt_dir, "_heldout_tmp.csv")
    ho.to_csv(tmp, index=False)
    try:
        ho_ds = CompatibilityDataset(tmp, HELDOUT_API_DESC, HELDOUT_EXC_DESC, NORM_STATS)
        ho_loader = DataLoader(ho_ds, batch_size=len(ho_ds), shuffle=False, collate_fn=collate_fn)
        ho_probs, ho_labels = collect_predictions(model, ho_loader, device)
    finally:
        os.remove(tmp)
    heldout_metrics = metric_suite(ho_probs, ho_labels, threshold)

    info = base_info(name, "baseline", seed, device, history, threshold, t0)
    info["config"] = {k: getattr(config, k) for k in ["encoder", "fixed_vector_source", "fixed_vector_path", "fusion", "pooling",
                                                      "loss", "use_descriptors", "use_balanced_sampler", "lr", "batch_size",
                                                      "max_epochs", "early_stop_patience", "threshold_step"]}
    info["descriptor_norm_stats"] = NORM_STATS
    info["heldout_fixed_vectors"] = vec_report
    save_outputs(metrics_dir, val_metrics, heldout_metrics, history, ho, ho_probs, threshold, info)

    del model, encoder
    return val_metrics, heldout_metrics


# ---------------------------------------------------------------------------
# PGB (same recipe as experiments/pgb / src/single_run.py)
# ---------------------------------------------------------------------------
def train_pgb(name, seed, device_str):
    from experiments.pgb.config import PGBConfig
    from src.cross_validate import build_encoder, butina_cluster
    from src.dataset import PGBDataset, pgb_collate_fn, PGBNormStats
    from src.model import pgb_model_from_config
    from src.pgb_prior import ExcipientPriorTable, build_leave_cluster_out_prior_vectors
    from src.train import pgb_train

    t0 = time.time()
    family = PGB_MODELS[name]
    ckpt_dir, metrics_dir = run_dirs(name, seed)
    config = PGBConfig()
    config.data_dir = "data"
    config.split_type = "cluster"
    config.apply_family_spec(family)
    config.seed = seed
    if device_str:
        config.device = device_str

    seed_everything(config.seed)
    device = get_device(config.device)
    train_df, val_df = pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)

    run_config = copy.deepcopy(config)
    run_config.train_csv, run_config.val_csv, run_config.test_csv = TRAIN_CSV, VAL_CSV, None
    run_config.checkpoint_dir, run_config.metrics_dir = ckpt_dir, metrics_dir
    run_config.positive_prior = float(train_df["Outcome1"].mean())
    run_config.use_balanced_sampler = False
    run_config.loss = "asym_focal"
    os.makedirs(ckpt_dir, exist_ok=True)

    encoder = build_encoder(run_config, device)
    if not encoder.is_sequence_capable:
        run_config.fusion = "concat"
    fv_enc = encoder if run_config.encoder == "fixed_vector" else None
    fvs = getattr(run_config, "fixed_vector_source", "maccs")
    if fv_enc is not None and fvs in ("pubchemfp", "mol2vec"):
        assert_coverage(fv_enc, pd.concat([train_df.API_Smiles, train_df.Excipient_Smiles,
                                           val_df.API_Smiles, val_df.Excipient_Smiles]).dropna().astype(str).tolist(), "train/val")

    all_train_smiles = list(set(train_df["API_Smiles"].dropna().tolist() + train_df["Excipient_Smiles"].dropna().tolist()))
    pgb_norm = PGBNormStats.from_smiles_list(all_train_smiles)
    pgb_norm.save(os.path.join(ckpt_dir, "pgb_norm_stats.npz"))

    prior_table = ExcipientPriorTable()
    prior_table.fit(train_df, leave_out_cluster_id=None)
    prior_table.save(os.path.join(ckpt_dir, "prior_table.json"))
    train_clustered = train_df.copy()
    train_clustered["cluster_id"] = train_clustered["API_Smiles"].map(butina_cluster(train_df["API_Smiles"].unique().tolist()))
    train_priors = build_leave_cluster_out_prior_vectors(train_clustered, cluster_col="cluster_id")

    def loader(csv, row_priors=None, shuffle=False):
        ds = PGBDataset(csv, run_config.encoder, fvs, prior_table, pgb_norm, fixed_vector_encoder=fv_enc, row_priors=row_priors)
        return DataLoader(ds, batch_size=run_config.pgb_batch_size, shuffle=shuffle, collate_fn=pgb_collate_fn, num_workers=0)

    train_loader = loader(TRAIN_CSV, row_priors=train_priors, shuffle=True)
    val_loader = loader(VAL_CSV)
    model = pgb_model_from_config(run_config, encoder)
    model.to(device)
    print(f"[{name} | seed {seed}] PGB family={family} | train {len(train_df)} val {len(val_df)}")

    model, history = pgb_train(model, train_loader, val_loader, run_config, device)

    val_probs, val_labels = collect_predictions(model, val_loader, device)
    threshold, _ = tune_threshold(val_probs, val_labels, run_config.threshold_step, tnr_floor=getattr(run_config, "pgb_tnr_floor", 0.0))
    val_metrics = metric_suite(val_probs, val_labels, threshold)

    ho = heldout_frame()
    vec_report = fill_heldout_vectors(fv_enc, ho) if fv_enc is not None and fvs in ("pubchemfp", "mol2vec") else None
    tmp = os.path.join(ckpt_dir, "_heldout_tmp.csv")
    ho.to_csv(tmp, index=False)
    try:
        ho_probs, ho_labels = collect_predictions(model, loader(tmp), device)
    finally:
        os.remove(tmp)
    heldout_metrics = metric_suite(ho_probs, ho_labels, threshold)

    info = base_info(name, "pgb", seed, device, history, threshold, t0)
    info["config"] = {"family": family, **{k: getattr(run_config, k, None) for k in
                      ["encoder", "fixed_vector_source", "fusion", "pooling", "loss", "pgb_lr", "pgb_encoder_lr",
                       "pgb_batch_size", "pgb_max_epochs", "pgb_early_stop_patience", "pgb_tnr_floor", "pgb_prior_vec_dim"]}}
    info["heldout_fixed_vectors"] = vec_report
    save_outputs(metrics_dir, val_metrics, heldout_metrics, history, ho, ho_probs, threshold, info)

    del model, encoder
    return val_metrics, heldout_metrics


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def summarize():
    rows = []
    for name in sorted(os.listdir(METRICS_ROOT)) if os.path.isdir(METRICS_ROOT) else []:
        mdir = os.path.join(METRICS_ROOT, name)
        if not os.path.isdir(mdir):
            continue
        for sd in sorted(os.listdir(mdir)):
            p = os.path.join(mdir, sd, "heldout_metrics.json")
            if not os.path.exists(p):
                continue
            ho = json.load(open(p))
            va = json.load(open(os.path.join(mdir, sd, "val_metrics.json")))
            rows.append({"model": name, "seed": sd.replace("seed_", ""),
                         "heldout_correct": ho["correct"], "heldout_TP": ho["tp"], "heldout_TN": ho["tn"],
                         "heldout_pr_auc": ho["pr_auc"], "heldout_roc_auc": ho["roc_auc"], "heldout_mcc": ho["mcc"],
                         "val_pr_auc": va["pr_auc"], "val_roc_auc": va["roc_auc"], "val_mcc": va["mcc"], "val_f1": va["f1"],
                         "threshold": va["threshold"]})
    if not rows:
        print(f"No finished runs under {METRICS_ROOT}/")
        return
    df = pd.DataFrame(rows).sort_values(["heldout_correct", "val_pr_auc"], ascending=False)
    df.to_csv(os.path.join(METRICS_ROOT, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(df.round(4).to_string(index=False))
    if df["seed"].nunique() > 1:
        agg = df.groupby("model").agg(seeds=("seed", "count"), heldout_correct_mean=("heldout_correct", "mean"),
                                      heldout_correct_std=("heldout_correct", "std"), val_pr_auc_mean=("val_pr_auc", "mean"),
                                      val_pr_auc_std=("val_pr_auc", "std"), heldout_pr_auc_mean=("heldout_pr_auc", "mean"))
        agg = agg.sort_values(["heldout_correct_mean", "val_pr_auc_mean"], ascending=False)
        agg.to_csv(os.path.join(METRICS_ROOT, "summary_by_model.csv"))
        print("\n" + agg.round(4).to_string())
    print(f"\nSaved {METRICS_ROOT}/summary.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def resolve_models(selection, skip_duplicates):
    out = []
    for s in selection:
        if s == "all":
            out += ALL_MODELS
        elif s == "baseline":
            out += list(BASELINE_MODELS)
        elif s == "pgb":
            out += list(PGB_MODELS)
        elif s in BASELINE_MODELS or s in PGB_MODELS:
            out.append(s)
        else:
            raise SystemExit(f"Unknown model '{s}'. See --list.")
    out = list(dict.fromkeys(out))
    if skip_duplicates:
        out = [m for m in out if m not in DUPLICATE_OF]
    return out


def main():
    p = argparse.ArgumentParser(description="Train models on the 80/20 split and evaluate on the 24 held-out pairs.")
    p.add_argument("--models", nargs="+", default=None, help="model names, or: all | baseline | pgb")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--device", default=None)
    p.add_argument("--skip_duplicates", action="store_true", help="skip the 4 concat runs identical to their *_cls twin")
    p.add_argument("--overwrite", action="store_true", help="retrain runs that already have heldout_metrics.json")
    p.add_argument("--list", action="store_true")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()

    if args.list:
        for m in ALL_MODELS:
            print(m + (f"   (duplicate of {DUPLICATE_OF[m]})" if m in DUPLICATE_OF else ""))
        return
    if args.summary:
        summarize()
        return
    if not args.models:
        raise SystemExit("Give --models (names, all, baseline or pgb), --list or --summary.")

    check_split_files()
    models = resolve_models(args.models, args.skip_duplicates)
    print(f"{len(models)} model(s) x {len(args.seeds)} seed(s) on {SPLIT_DIR}")
    failures = []
    for seed in args.seeds:
        for name in models:
            _, metrics_dir = run_dirs(name, seed)
            if os.path.exists(os.path.join(metrics_dir, "heldout_metrics.json")) and not args.overwrite:
                print(f"skip {name} seed {seed} (done; --overwrite to retrain)")
                continue
            print(f"\n{'=' * 70}\n{name} | seed {seed}\n{'=' * 70}")
            try:
                fn = train_pgb if name in PGB_MODELS else train_baseline
                val_m, ho_m = fn(name, seed, args.device)
                print(f"[{name} | seed {seed}] val PR-AUC {val_m['pr_auc']:.4f} | held-out {ho_m['correct']}/{ho_m['n']} "
                      f"(TP {ho_m['tp']}, TN {ho_m['tn']}) PR-AUC {ho_m['pr_auc']:.4f}")
            except Exception as e:
                failures.append((name, seed, repr(e)))
                os.makedirs(METRICS_ROOT, exist_ok=True)
                with open(os.path.join(METRICS_ROOT, "failures.log"), "a") as f:
                    f.write(f"{datetime.now().isoformat(timespec='seconds')} {name} seed {seed}\n{traceback.format_exc()}\n")
                print(f"FAILED {name} seed {seed}: {e!r} (traceback in {METRICS_ROOT}/failures.log)")
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
