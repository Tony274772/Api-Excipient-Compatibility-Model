"""Train and evaluate one encoder family under the fixed comparison protocol.

Protocol (identical for all 9 families, see paper/encoder_comparison_plan.md):
    data      ARES api_cluster split, train 2770 / val 774; the 22 held-out pairs are external
    head      PGB head, unchanged
    extras    excipient prior (train only, leave-own-cluster-out for train rows), mechanism
              flags, chemistry flags, PGB descriptors and salt context (both z-scored with
              train-only statistics)
    loss      asymmetric focal, gamma+ 1.2 / gamma- 2.5, pos_weight from train, no sampler
    optimiser AdamW, PGB lr 1.2e-3, encoder-side lr 3e-4, weight decay 2e-3, batch 64, clip 1.0
    budget    max 40 epochs, early stopping on val PR-AUC (patience 7), LR patience 4
    threshold tuned on val only: best MCC with true-negative rate >= 0.97
    seed      42

The selection metric is validation PR-AUC. Held-out metrics are written to a separate
`_heldout_sealed/` folder and are NOT part of the summary table until `collect.py --reveal-heldout`
is run, so the test set cannot inform the choice of family.

Usage (see README.md):
    python -m experiments.encoder_comparison.run_comparison --family morgan
    python -m experiments.encoder_comparison.run_comparison --family all --device cuda
"""

import argparse
import inspect
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd
from rdkit import RDLogger
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.encoder_comparison.data import (OneRepresentationPGBDataset, SplitPaths,  # noqa: E402
                                                 build_vector_lookup, check_heldout_is_external)
from experiments.encoder_comparison.families import ORDER, spec  # noqa: E402
from experiments.encoder_comparison.towers import ComparisonTower  # noqa: E402
from src.config import Config  # noqa: E402
from src.dataset import PGBNormStats, pgb_collate_fn  # noqa: E402
from src.encoders import ENCODER_REGISTRY  # noqa: E402
from src.evaluate import collect_predictions, compute_metrics, tune_threshold  # noqa: E402
from src.model import PGBCompatibilityModel  # noqa: E402
from src.pgb_prior import ExcipientPriorTable, build_leave_cluster_out_prior_vectors  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "results", "encoder_comparison")
SEALED_ROOT = os.path.join(RESULTS_ROOT, "_heldout_sealed")
CKPT_ROOT = os.path.join(PROJECT_ROOT, "checkpoints", "encoder_comparison")


def run_name(family: str, seed: int) -> str:
    """Folder name for one family at one seed."""
    return f"{family}_seed{seed}"


# --------------------------------------------------------------------------- config
def make_config(family: str, paths: SplitPaths, seed: int, device: str, epochs: int) -> Config:
    """The one training recipe, with only the family's encoder fields changed."""
    s = spec(family)
    config = Config()
    config.use_pgb_head = True
    config.split_type = "cluster"
    config.seed = seed
    config.encoder = s["encoder"]
    if s["source"]:
        config.fixed_vector_source = s["source"]
    config.fusion = s["fusion"]
    if s["pooling"]:
        config.pooling = s["pooling"]
    config.pgb_max_epochs = epochs
    config.device = device
    config.resolve_paths()
    config.train_csv, config.val_csv, config.test_csv = paths.train, paths.val, paths.heldout
    config.positive_prior = float(pd.read_csv(paths.train)["Outcome1"].mean())
    config.use_balanced_sampler = False
    config.loss = "asym_focal"
    config.pgb_normalize_salt = True   # fix: z-score the salt context with train statistics
    return config


def build_encoder(config: Config, family: str, lookup_path: str, device):
    """Construct the family's encoder. The 5 pretrained encoders are frozen feature
    extractors (requires_grad=False, eval mode) — only the PGB layers are trained."""
    s = spec(family)
    cls = ENCODER_REGISTRY[config.encoder]
    if config.encoder == "fixed_vector":
        enc = cls(source=s["source"], vector_path=lookup_path, device=str(device))
    else:
        params = list(inspect.signature(cls.__init__).parameters)[1:]
        kwargs = {"device": str(device)}
        if "config" in params:
            kwargs["config"] = config
        if "model_path" in params:
            kwargs["model_path"] = {"molformer": config.molformer_model_path,
                                    "chemberta": config.chemberta_model_path}[config.encoder]
        if "pretrained_name" in params:
            kwargs["pretrained_name"] = config.gin_pretrained_name
        enc = cls(**kwargs)
    enc.to(device)
    return enc


def build_model(config: Config, family: str, encoder) -> PGBCompatibilityModel:
    """PGB model whose towers take exactly one representation."""
    s = spec(family)
    struct_dim = config.proj_dim if s["kind"] == "seq" else 0
    towers = [ComparisonTower(struct_dim, s["primary_dim"], config.pgb_tower_dim, config.proj_dropout)
              for _ in range(2)]
    model = PGBCompatibilityModel(config, encoder, towers[0], towers[1])
    model.head.init_bias(global_rate=config.positive_prior, pgb_prior_offset=config.pgb_prior_offset)
    return model


def make_loader(csv, config, prior_table, pgb_norm, encoder, row_priors=None, shuffle=False):
    ds = OneRepresentationPGBDataset(csv, config.encoder, getattr(config, "fixed_vector_source", None),
                                     prior_table, pgb_norm, fixed_vector_encoder=encoder,
                                     row_priors=row_priors)
    return DataLoader(ds, batch_size=config.pgb_batch_size, shuffle=shuffle,
                      collate_fn=pgb_collate_fn, num_workers=0)


# --------------------------------------------------------------------------- metrics
def metric_suite(probs, labels, threshold) -> dict:
    m = compute_metrics(probs, labels, threshold)
    labels = labels.astype(int)
    m["roc_auc"] = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan")
    cm = m["confusion_matrix"]
    m["tn"], m["fp"], m["fn"], m["tp"] = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    m["n"] = int(len(labels))
    m["correct"] = int(m["tp"] + m["tn"])
    return m


def prediction_table(df, probs, labels, threshold) -> pd.DataFrame:
    cols = [c for c in ["Api_name", "Excipient_name", "API_CID", "Excipient_CID"] if c in df]
    out = df[cols].copy()
    out["label"], out["probability"] = labels.astype(int), probs
    out["prediction"] = (probs >= threshold).astype(int)
    out["correct"] = (out["prediction"] == out["label"]).astype(int)
    return out


def score(model, df, config, prior_table, pgb_norm, encoder, device):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pairs.csv")
        df.to_csv(path, index=False)
        return collect_predictions(model, make_loader(path, config, prior_table, pgb_norm, encoder), device)


# --------------------------------------------------------------------------- one family
def run_family(family: str, paths: SplitPaths, seed: int, device_str: str, epochs: int,
               overwrite: bool) -> dict:
    s = spec(family)
    run = run_name(family, seed)
    out_dir = os.path.join(RESULTS_ROOT, run)
    sealed_dir = os.path.join(SEALED_ROOT, run)
    ckpt_dir = os.path.join(CKPT_ROOT, run)
    if os.path.exists(os.path.join(out_dir, "val_metrics.json")) and not overwrite:
        raise SystemExit(f"{out_dir} already has results; pass --overwrite to rerun")
    for d in (out_dir, sealed_dir, ckpt_dir):
        os.makedirs(d, exist_ok=True)

    t0 = time.time()
    seed_everything(seed)
    config = make_config(family, paths, seed, device_str, epochs)
    device = get_device(config.device)
    config.checkpoint_dir, config.metrics_dir = ckpt_dir, out_dir

    lookup_path = build_vector_lookup(family, paths, os.path.join(RESULTS_ROOT, "_lookup")) \
        if s["needs_lookup"] else None

    train_df, val_df = pd.read_csv(paths.train), pd.read_csv(paths.val)
    encoder = build_encoder(config, family, lookup_path, device)

    # train-only statistics
    train_smiles = list(set(train_df["API_Smiles"].dropna()) | set(train_df["Excipient_Smiles"].dropna()))
    pgb_norm = PGBNormStats.from_smiles_list(train_smiles, with_salt=config.pgb_normalize_salt)
    pgb_norm.save(os.path.join(ckpt_dir, "pgb_norm_stats.npz"))

    prior_table = ExcipientPriorTable(global_rate=config.positive_prior)   # train rate, not the legacy 0.094
    prior_table.fit(train_df, leave_out_cluster_id=None)
    prior_table.save(os.path.join(ckpt_dir, "prior_table.json"))
    train_priors = build_leave_cluster_out_prior_vectors(
        train_df, cluster_col="cluster_id", global_rate=prior_table.global_rate)

    train_loader = make_loader(paths.train, config, prior_table, pgb_norm, encoder,
                               row_priors=train_priors, shuffle=True)
    val_loader = make_loader(paths.val, config, prior_table, pgb_norm, encoder)

    model = build_model(config, family, encoder)
    model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 100}\n[{family}] {s['label']}")
    print(f"[{family}] device {device} | train {len(train_df)} | val {len(val_df)} | "
          f"trainable params {n_train:,} of {n_total:,} | max epochs {config.pgb_max_epochs}")
    print("=" * 100)

    from src.train import pgb_train
    model, history = pgb_train(model, train_loader, val_loader, config, device)

    # threshold from val only
    val_probs, val_labels = collect_predictions(model, val_loader, device)
    threshold, _ = tune_threshold(val_probs, val_labels, config.threshold_step,
                                  tnr_floor=config.pgb_tnr_floor)
    threshold = float(threshold)

    val_metrics = metric_suite(val_probs, val_labels, threshold)
    ho_df = paths.frame("heldout")
    ho_probs, ho_labels = score(model, ho_df, config, prior_table, pgb_norm, encoder, device)
    ho_metrics = metric_suite(ho_probs, ho_labels, threshold)

    minutes = round((time.time() - t0) / 60, 2)
    run_info = {
        "family": family, "seed_run": run, "label": s["label"], "representation_dim": s["primary_dim"] or config.proj_dim,
        "encoder": config.encoder, "fusion": config.fusion, "pooling": getattr(config, "pooling", None),
        "seed": seed, "device": str(device), "parameters_total": n_total, "parameters_trainable": n_train,
        "epochs_trained": len(history["val_pr_auc"]),
        "best_epoch": int(pd.Series(history["val_pr_auc"]).idxmax()) + 1,
        "max_epochs": config.pgb_max_epochs,
        "threshold_from_val": threshold, "tnr_floor": config.pgb_tnr_floor,
        "data": {"train": paths.train, "val": paths.val, "heldout": paths.heldout,
                 "rows": {"train": len(train_df), "val": len(val_df), "heldout": len(ho_df)}},
        "minutes": minutes, "finished": datetime.now().isoformat(timespec="seconds"),
    }

    # selection-relevant output
    with open(os.path.join(out_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    prediction_table(val_df, val_probs, val_labels, threshold).to_csv(
        os.path.join(out_dir, "predictions_val.csv"), index=False)

    # sealed: not read until the family has been chosen
    with open(os.path.join(sealed_dir, "heldout_metrics.json"), "w") as f:
        json.dump(ho_metrics, f, indent=2)
    prediction_table(ho_df, ho_probs, ho_labels, threshold).to_csv(
        os.path.join(sealed_dir, "predictions_heldout.csv"), index=False)

    print(f"\n[{run}] val PR-AUC {val_metrics['pr_auc']:.4f} | val MCC {val_metrics['mcc']:.4f} | "
          f"val correct {val_metrics['correct']}/{val_metrics['n']} | threshold {threshold:.3f} | {minutes} min")
    print(f"[{run}] selection results -> {out_dir}")
    print(f"[{run}] held-out results sealed in {sealed_dir} (not used for the choice)")
    return val_metrics


def main():
    p = argparse.ArgumentParser(description="Encoder-family comparison under one fixed protocol.")
    p.add_argument("--family", default="all", help=f"one of: {', '.join(ORDER)}, or 'all'")
    p.add_argument("--data-root", default=None, help="folder holding splits/ and heldout/ (default: final_project/data in this repo)")
    p.add_argument("--split", default="api_cluster", help="split folder name under <data-root>/splits/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available). Use ONE device for all runs.")
    p.add_argument("--epochs", type=int, default=40, help="max epochs, identical for every family")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    families = ORDER if args.family == "all" else [args.family]
    for f in families:
        spec(f)

    paths = SplitPaths(args.data_root, args.split)
    check_heldout_is_external(paths)
    device_str = args.device or "auto"
    print(f"data root  : {paths.root}")
    print(f"train/val  : {paths.train} | {paths.val}")
    print(f"held-out   : {paths.heldout}")
    print(f"families   : {', '.join(families)}")
    print(f"seed {args.seed} | max epochs {args.epochs} | device {get_device(device_str)}")
    print("prior      : train positive rate as the global rate (src/pgb_prior.py)")

    done, failed = [], []
    for family in families:
        try:
            run_family(family, paths, args.seed, device_str, args.epochs, args.overwrite)
            done.append(family)
        except SystemExit:
            raise
        except Exception as e:
            failed.append((family, f"{type(e).__name__}: {e}"))
            print(f"\n[{family}] FAILED — {type(e).__name__}: {e}\n")

    print(f"\nfinished: {', '.join(done) if done else 'none'}")
    for family, err in failed:
        print(f"failed  : {family} — {err}")
    print("\nNext: python -m experiments.encoder_comparison.collect")
    print("      (all seeds of all families: python -m experiments.encoder_comparison.run_all_seeds)")


if __name__ == "__main__":
    main()
