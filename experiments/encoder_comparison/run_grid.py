"""Train and evaluate ONE grid cell (one primary + optional secondary) at one seed.

The protocol is the one the 45 completed `none`-column runs used, unchanged. Only the molecular
description differs between cells. See paper/encoder_grid_plan.md Part 5 for the full list of what
is held fixed:

    data      ARES api_cluster split, train 2770 / val 774; the 22 held-out pairs are external
    head      PGB head, unchanged
    extras    excipient prior (train only, leave-own-cluster-out for train rows), mechanism flags,
              chemistry flags, PGB descriptors and salt context (both z-scored on train statistics)
    loss      asymmetric focal, gamma+ 1.2 / gamma- 2.5, pos_weight from train, no sampler
    optimiser AdamW, PGB lr 1.2e-3, encoder-side lr 3e-4, weight decay 2e-3, batch 64, clip 1.0
    budget    max 40 epochs, early stopping on val PR-AUC (patience 7), LR patience 4
    threshold tuned on val only: best MCC with true-negative rate >= 0.97
    seeds     42, 43, 44, 45, 46 - the same five in every cell, which is what makes the paired
              tests valid

Selection metric is validation PR-AUC. Held-out metrics go to `_heldout_sealed/` and are not part
of any summary until the choice is locked and collect_grid.py --reveal-heldout is run.

A run is finished only when the `COMPLETE` sentinel exists. It is written LAST, after every other
output, so a run killed part-way is retried instead of being silently treated as done.

    python -m experiments.encoder_comparison.run_grid --cell morgan__x_maccs --seed 42
    python -m experiments.encoder_comparison.run_grid --cell floor --seed 42 --epochs 1
    python -m experiments.encoder_comparison.run_grid --list
"""

import argparse
import inspect
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd
import torch
from rdkit import RDLogger
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.encoder_comparison.data import (NullEncoder, SplitPaths,  # noqa: E402
                                                 build_cell_representations,
                                                 check_heldout_is_external, make_grid_loader,
                                                 sanity_check_batch)
from experiments.encoder_comparison.families import (FAMILIES, all_cells,  # noqa: E402
                                                     cell_by_name, grid_summary)
from experiments.encoder_comparison.towers import ComparisonTower  # noqa: E402
from src.config import Config  # noqa: E402
from src.dataset import PGBNormStats  # noqa: E402
from src.encoders import ENCODER_REGISTRY  # noqa: E402
from src.evaluate import collect_predictions, compute_metrics, tune_threshold  # noqa: E402
from src.model import PGBCompatibilityModel  # noqa: E402
from src.pgb_prior import ExcipientPriorTable, build_leave_cluster_out_prior_vectors  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "results", "encoder_grid")
SEALED_ROOT = os.path.join(RESULTS_ROOT, "_heldout_sealed")
LOOKUP_DIR = os.path.join(RESULTS_ROOT, "_lookup")
CKPT_ROOT = os.path.join(PROJECT_ROOT, "checkpoints", "encoder_grid")

# Written as the very last action of a finished run. run_all_grid.py keys resume on this file and
# nothing else, which is the fix for problem 1 in Part 6 of the plan: the old marker was
# val_metrics.json, written FIRST, so a run killed before the held-out metrics were saved counted
# as done forever.
SENTINEL = "COMPLETE"


def run_name(cell_name: str, seed: int) -> str:
    return f"{cell_name}_seed{seed}"


def is_complete(cell_name: str, seed: int) -> bool:
    return os.path.exists(os.path.join(RESULTS_ROOT, run_name(cell_name, seed), SENTINEL))


# --------------------------------------------------------------------------- config
def make_config(cell, paths: SplitPaths, seed: int, device: str, epochs: int) -> Config:
    """The one training recipe, with only this cell's description fields changed."""
    config = Config()
    config.use_pgb_head = True
    config.split_type = "cluster"
    config.seed = seed

    if cell.kind == "seq":
        spec = FAMILIES[cell.primary]
        config.encoder = spec["encoder"]
        config.fusion = "cross_attn"
        config.pooling = cell.pooling
    else:
        # Fingerprint cells, the floor, and the no-cross-attention control all take the concat
        # path. NullEncoder reports is_sequence_capable False, so the model builds no projection,
        # attention or pooling layers and never calls the encoder.
        config.encoder = "fixed_vector"
        config.fusion = "concat"

    config.pgb_max_epochs = epochs
    config.device = device
    config.resolve_paths()
    config.train_csv, config.val_csv, config.test_csv = paths.train, paths.val, paths.heldout
    config.positive_prior = float(pd.read_csv(paths.train)["Outcome1"].mean())
    config.use_balanced_sampler = False
    config.loss = "asym_focal"
    config.pgb_normalize_salt = True        # z-score the salt context with train statistics
    config.pgb_save_trainable_only = True   # skip the frozen encoder in the checkpoint
    return config


def build_encoder(config: Config, cell, device):
    """The cell's encoder: a frozen pretrained network, or a stub for the concat cells."""
    if cell.kind != "seq":
        return NullEncoder().to(device)

    cls = ENCODER_REGISTRY[config.encoder]
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
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def build_model(config: Config, cell, encoder, primary_rep, secondary_rep):
    """PGB model whose two towers carry this cell's description(s)."""
    struct_dim = config.proj_dim if cell.kind == "seq" else 0
    primary_dim = primary_rep.dim if primary_rep is not None else 0
    secondary_dim = secondary_rep.dim if secondary_rep is not None else 0

    towers = [ComparisonTower(struct_dim=struct_dim, primary_dim=primary_dim,
                              secondary_dim=secondary_dim, pgb_dim=config.pgb_tower_dim,
                              dropout=config.proj_dropout,
                              allow_no_representation=(cell.block == "floor"))
              for _ in range(2)]
    model = PGBCompatibilityModel(config, encoder, towers[0], towers[1])

    # The one failure this must not have: a deep cell silently losing its encoder. Setting
    # fusion="concat" on a sequence encoder makes _encode_struct return (None, None) and the
    # encoder output is dropped with no error, which is why the no-cross-attention control runs
    # through the fixed-vector path instead.
    if cell.kind == "seq" and not model.use_cross_attn:
        raise RuntimeError(f"{cell.name}: expected the cross-attention path but the model is on "
                           "concat - the encoder's output would be discarded")
    if cell.kind != "seq" and model.use_cross_attn:
        raise RuntimeError(f"{cell.name}: expected the concat path but the model is on "
                           "cross-attention")

    model.head.init_bias(global_rate=config.positive_prior,
                         pgb_prior_offset=config.pgb_prior_offset)
    return model


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


def score(model, df, config, cell, prior_table, pgb_norm, primary_rep, secondary_rep, device):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pairs.csv")
        df.to_csv(path, index=False)
        loader = make_grid_loader(path, config, cell, prior_table, pgb_norm,
                                  primary_rep, secondary_rep)
        return collect_predictions(model, loader, device)


# --------------------------------------------------------------------------- one cell
def run_cell(cell, paths: SplitPaths, seed: int, device_str: str, epochs: int,
             overwrite: bool) -> dict:
    run = run_name(cell.name, seed)
    out_dir = os.path.join(RESULTS_ROOT, run)
    sealed_dir = os.path.join(SEALED_ROOT, run)
    ckpt_dir = os.path.join(CKPT_ROOT, run)
    if is_complete(cell.name, seed) and not overwrite:
        raise SystemExit(f"{out_dir} is already COMPLETE; pass --overwrite to rerun")
    # A leftover sentinel from a previous attempt must not survive a rerun that then crashes.
    sentinel_path = os.path.join(out_dir, SENTINEL)
    if os.path.exists(sentinel_path):
        os.remove(sentinel_path)
    for d in (out_dir, sealed_dir, ckpt_dir, LOOKUP_DIR):
        os.makedirs(d, exist_ok=True)

    t0 = time.time()
    seed_everything(seed)
    config = make_config(cell, paths, seed, device_str, epochs)
    device = get_device(config.device)
    config.checkpoint_dir, config.metrics_dir = ckpt_dir, out_dir

    train_df, val_df = pd.read_csv(paths.train), pd.read_csv(paths.val)

    # Representations: tables loaded, shuffle derangement built over every split, scalers fitted
    # on TRAIN molecules only.
    primary_rep, secondary_rep = build_cell_representations(cell, paths, LOOKUP_DIR)
    for rep, slot in ((primary_rep, "primary"), (secondary_rep, "secondary")):
        if rep is not None:
            rep.save_scaler(os.path.join(ckpt_dir, f"{slot}_scaler.npz"))

    encoder = build_encoder(config, cell, device)

    # train-only statistics for the descriptors and salt context
    train_smiles = list(set(train_df["API_Smiles"].dropna())
                        | set(train_df["Excipient_Smiles"].dropna()))
    pgb_norm = PGBNormStats.from_smiles_list(train_smiles, with_salt=config.pgb_normalize_salt)
    pgb_norm.save(os.path.join(ckpt_dir, "pgb_norm_stats.npz"))

    prior_table = ExcipientPriorTable(global_rate=config.positive_prior)  # train rate, not 0.094
    prior_table.fit(train_df, leave_out_cluster_id=None)
    prior_table.save(os.path.join(ckpt_dir, "prior_table.json"))
    train_priors = build_leave_cluster_out_prior_vectors(
        train_df, cluster_col="cluster_id", global_rate=prior_table.global_rate)

    train_loader = make_grid_loader(paths.train, config, cell, prior_table, pgb_norm,
                                    primary_rep, secondary_rep, row_priors=train_priors,
                                    shuffle=True)
    val_loader = make_grid_loader(paths.val, config, cell, prior_table, pgb_norm,
                                  primary_rep, secondary_rep)

    # Catch a dataset/tower disagreement on batch one rather than after 40 epochs.
    sanity_check_batch(next(iter(val_loader)), cell, primary_rep, secondary_rep)

    model = build_model(config, cell, encoder, primary_rep, secondary_rep)
    model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'=' * 100}\n[{cell.name}] {cell.label()}")
    print(f"[{cell.name}] block {cell.block} | seed {seed} | device {device} | "
          f"train {len(train_df)} | val {len(val_df)}")
    print(f"[{cell.name}] primary {primary_rep.name + ' ' + str(primary_rep.dim) if primary_rep else ('cross-attended encoder' if cell.kind == 'seq' else 'NONE')}"
          f" | secondary {secondary_rep.name + ' ' + str(secondary_rep.dim) if secondary_rep else 'none'}"
          f" | description blocks {cell.extra_block_params():,} params")
    print(f"[{cell.name}] trainable params {n_train:,} of {n_total:,} | "
          f"max epochs {config.pgb_max_epochs}")
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
    ho_probs, ho_labels = score(model, ho_df, config, cell, prior_table, pgb_norm,
                                primary_rep, secondary_rep, device)
    ho_metrics = metric_suite(ho_probs, ho_labels, threshold)

    minutes = round((time.time() - t0) / 60, 2)
    run_info = {
        "cell": cell.name, "seed_run": run, "block": cell.block, "label": cell.label(),
        "note": cell.note,
        "primary": cell.primary,
        "primary_rep": primary_rep.name if primary_rep else None,
        "primary_dim": primary_rep.dim if primary_rep else (config.proj_dim if cell.kind == "seq" else 0),
        "primary_scaled": bool(primary_rep.scaled) if primary_rep else None,
        "secondary": cell.secondary,
        "secondary_dim": secondary_rep.dim if secondary_rep else 0,
        "secondary_scaled": bool(secondary_rep.scaled) if secondary_rep else None,
        "secondary_shuffle_sha1": secondary_rep.shuffle_fingerprint() if secondary_rep else "",
        "description_block_params": cell.extra_block_params(),
        "no_cross_attn": cell.no_cross_attn,
        "cross_attention": bool(model.use_cross_attn),
        "encoder": config.encoder, "fusion": config.fusion,
        "pooling": cell.pooling,
        "seed": seed, "device": str(device),
        "parameters_total": n_total, "parameters_trainable": n_train,
        "epochs_trained": len(history["val_pr_auc"]),
        "best_epoch": int(pd.Series(history["val_pr_auc"]).idxmax()) + 1,
        "max_epochs": config.pgb_max_epochs,
        "threshold_from_val": threshold, "tnr_floor": config.pgb_tnr_floor,
        "data": {"train": paths.train, "val": paths.val, "heldout": paths.heldout,
                 "rows": {"train": len(train_df), "val": len(val_df), "heldout": len(ho_df)}},
        "minutes": minutes, "finished": datetime.now().isoformat(timespec="seconds"),
    }

    # ---- selection-relevant output
    with open(os.path.join(out_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    prediction_table(val_df, val_probs, val_labels, threshold).to_csv(
        os.path.join(out_dir, "predictions_val.csv"), index=False)

    # ---- sealed: not read until the cell has been chosen
    with open(os.path.join(sealed_dir, "heldout_metrics.json"), "w") as f:
        json.dump(ho_metrics, f, indent=2)
    prediction_table(ho_df, ho_probs, ho_labels, threshold).to_csv(
        os.path.join(sealed_dir, "predictions_heldout.csv"), index=False)

    # ---- the sentinel, LAST, once every file above is on disk
    for required in ("val_metrics.json", "run_info.json", "training_history.json",
                     "predictions_val.csv"):
        if not os.path.exists(os.path.join(out_dir, required)):
            raise RuntimeError(f"{run}: {required} was not written; refusing to mark COMPLETE")
    for required in ("heldout_metrics.json", "predictions_heldout.csv"):
        if not os.path.exists(os.path.join(sealed_dir, required)):
            raise RuntimeError(f"{run}: sealed {required} was not written; refusing to mark COMPLETE")
    with open(sentinel_path, "w") as f:
        f.write(json.dumps({"run": run, "val_pr_auc": val_metrics["pr_auc"],
                            "finished": run_info["finished"]}) + "\n")

    print(f"\n[{run}] val PR-AUC {val_metrics['pr_auc']:.4f} | val MCC {val_metrics['mcc']:.4f} | "
          f"val correct {val_metrics['correct']}/{val_metrics['n']} | "
          f"threshold {threshold:.3f} | {minutes} min")
    print(f"[{run}] selection results -> {out_dir}")
    print(f"[{run}] held-out sealed in {sealed_dir} (not used for the choice)")
    return val_metrics


def main():
    p = argparse.ArgumentParser(description="Train one cell of the encoder grid.")
    p.add_argument("--cell", help="cell name, e.g. morgan__x_maccs, floor, dmpnn__noca")
    p.add_argument("--list", action="store_true", help="list every cell and stop")
    p.add_argument("--data-root", default=None,
                   help="folder holding splits/ and heldout/ (default: final_project/data)")
    p.add_argument("--split", default="api_cluster")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument("--epochs", type=int, default=40, help="max epochs, identical for every cell")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.list:
        print(grid_summary())
        print()
        for c in all_cells():
            print(f"{c.block:14s} {c.name:34s} {c.label()[:90]}")
        return
    if not args.cell:
        raise SystemExit("pass --cell <name>, or --list to see them all")

    cell = cell_by_name(args.cell)
    paths = SplitPaths(args.data_root, args.split)
    check_heldout_is_external(paths)
    device_str = args.device or "auto"
    print(f"data root  : {paths.root}")
    print(f"cell       : {cell.name}  (block {cell.block})")
    print(f"seed {args.seed} | max epochs {args.epochs} | device {get_device(device_str)}")
    print("prior      : train positive rate as the global rate (src/pgb_prior.py)")

    run_cell(cell, paths, args.seed, device_str, args.epochs, args.overwrite)
    print("\nNext: python -m experiments.encoder_comparison.collect_grid")


if __name__ == "__main__":
    main()
