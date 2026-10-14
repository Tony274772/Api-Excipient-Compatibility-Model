"""
Excipient-prior ablation for the PGB model (default family: morgan).

Variants (everything except the prior is identical to experiments/pgb/single_run.py):
    baseline_4level_5d : original PGB  — 4-level prior cascade, 5-d prior vector
    ablation_3level_5d : family level removed, 5-d prior vector      (isolates change A)
    ablation_3level_1d : family level removed, 1-d prior [p_exact]  (A + B)

Layer widths follow config.pgb_prior_vec_dim:
    5-d : gate in 46, pair representation 410, prior-trust in 5
    1-d : gate in 42, pair representation 406, prior-trust in 1
The final logit is unchanged: residual + w * logit(p_exact) + b.

Outputs per run:
    experiments/results/ablation_studies/<variant>/<family>/seed_<s>/
        val_metrics.json, test_metrics.json, heldout_metrics.json, training_history.json,
        predictions_{val,test,heldout}.csv, prior_level_counts.json, run_info.json
    checkpoints/ablation_studies/<variant>/<family>/seed_<s>/

Usage (from the project root):
    python -m experiments.ablation_studies.run_ablation --variant all --seeds 42 43 44 45 46
    python -m experiments.ablation_studies.run_ablation --variant ablation_3level_1d --seeds 42
"""

import argparse
import copy
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.pgb.config import PGBConfig, FAMILIES
from experiments.ablation_studies.dataset import PriorSlicedPGBDataset
from experiments.ablation_studies.prior_3level import (
    ThreeLevelPriorTable,
    build_leave_cluster_out_prior_vectors,
    prior_level,
)
from src.dataset import pgb_collate_fn, PGBNormStats
from src.pgb_prior import ExcipientPriorTable
from src.model import pgb_model_from_config
from src.train import pgb_train
from src.evaluate import collect_predictions, tune_threshold, save_metrics
from src.cross_validate import build_encoder, butina_cluster
from src.single_run import compute_extended_metrics
from src.utils import seed_everything, get_device

VARIANTS = {
    "baseline_4level_5d": {"table_cls": ExcipientPriorTable,  "prior_dim": 5},
    "ablation_3level_5d": {"table_cls": ThreeLevelPriorTable, "prior_dim": 5},
    "ablation_3level_1d": {"table_cls": ThreeLevelPriorTable, "prior_dim": 1},
}
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
RESULTS_ROOT = "experiments/results/ablation_studies"
CHECKPOINT_ROOT = "checkpoints/ablation_studies"
HELDOUT_CSV = "held_out_testset/held_out_test_set.csv"


def run_dir(root: str, variant: str, family: str, seed: int) -> str:
    return os.path.join(root, variant, family, f"seed_{seed}")


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _predictions_frame(df, probs, labels, prior_table, prior_dim):
    out = pd.DataFrame({
        "API_Smiles": df["API_Smiles"].values,
        "Excipient_Smiles": df["Excipient_Smiles"].values,
        "label": labels.astype(int),
        "prob": probs,
    })
    exc = df["Excipient_Smiles"].map(lambda s: "" if pd.isna(s) else str(s))
    out["prior_p_exact"] = [prior_table.lookup(s)["p_exact"] if s else 0.094 for s in exc]
    out["prior_level"] = [prior_level(prior_table, s) for s in exc]
    out["prior_dim"] = prior_dim
    return out


def run_one(variant: str, family: str, seed: int, epochs=None, device_str=None,
            results_root: str = RESULTS_ROOT, checkpoint_root: str = CHECKPOINT_ROOT) -> dict:
    spec = VARIANTS[variant]
    table_cls, prior_dim = spec["table_cls"], spec["prior_dim"]

    # Same config construction as experiments/pgb/single_run.py
    config = PGBConfig()
    config.data_dir = "data"
    config.split_type = "cluster"
    config.apply_family_spec(family)
    config.seed = seed
    if epochs:
        config.pgb_max_epochs = epochs
    if device_str:
        config.device = device_str

    metrics_dir = run_dir(results_root, variant, family, seed)
    checkpoint_dir = run_dir(checkpoint_root, variant, family, seed)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- From here on the order of operations mirrors src/single_run.py:pgb_single_run ----
    seed_everything(config.seed)
    device = get_device(config.device)

    train_csv = config.train_csv or os.path.join(config.data_dir, "train.csv")
    val_csv   = config.val_csv   or os.path.join(config.data_dir, "val.csv")
    test_csv  = config.test_csv  or os.path.join(config.data_dir, "test.csv")

    train_fold = pd.read_csv(train_csv)
    val_fold   = pd.read_csv(val_csv)
    test_fold  = pd.read_csv(test_csv)

    run_config = copy.deepcopy(config)
    run_config.train_csv = train_csv
    run_config.val_csv   = val_csv
    run_config.test_csv  = test_csv
    run_config.checkpoint_dir = checkpoint_dir
    run_config.metrics_dir = metrics_dir
    run_config.positive_prior = float(train_fold["Outcome1"].mean())
    run_config.use_balanced_sampler = False
    run_config.loss = "asym_focal"
    run_config.pgb_prior_vec_dim = prior_dim   # the only architecture change

    encoder = build_encoder(run_config, device)
    if not encoder.is_sequence_capable:
        run_config.fusion = "concat"

    all_train_smiles = list(set(
        train_fold["API_Smiles"].dropna().tolist() +
        train_fold["Excipient_Smiles"].dropna().tolist()
    ))
    pgb_norm = PGBNormStats.from_smiles_list(all_train_smiles)
    pgb_norm.save(os.path.join(checkpoint_dir, "pgb_norm_stats.npz"))

    prior_table = table_cls()
    prior_table.fit(train_fold, leave_out_cluster_id=None)
    prior_table.save(os.path.join(checkpoint_dir, "prior_table.json"))

    unique_apis = train_fold["API_Smiles"].unique().tolist()
    smiles_to_cluster = butina_cluster(unique_apis)
    train_fold_copy = train_fold.copy()
    train_fold_copy["cluster_id"] = train_fold_copy["API_Smiles"].map(smiles_to_cluster)
    train_priors, train_levels = build_leave_cluster_out_prior_vectors(
        train_fold_copy, table_cls=table_cls, cluster_col="cluster_id"
    )

    fv_enc = encoder if run_config.encoder == "fixed_vector" else None
    fvs = getattr(run_config, "fixed_vector_source", "maccs")

    def make_ds(csv_path, row_priors=None):
        return PriorSlicedPGBDataset(
            csv_path, run_config.encoder, fvs, prior_table, pgb_norm,
            fixed_vector_encoder=fv_enc, row_priors=row_priors, prior_dim=prior_dim,
        )

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=run_config.pgb_batch_size, shuffle=shuffle,
                          collate_fn=pgb_collate_fn, num_workers=0)

    train_loader = make_loader(make_ds(train_csv, row_priors=train_priors), shuffle=True)
    val_loader   = make_loader(make_ds(val_csv), shuffle=False)
    test_loader  = make_loader(make_ds(test_csv), shuffle=False)

    model = pgb_model_from_config(run_config, encoder)
    model.to(device)

    # Confirm the head was built with the intended widths
    head = model.head
    gate_in = head.gate[0].in_features
    concat_dim = head.mlp[0].in_features
    trust_in = head.prior_trust_mlp[0].in_features
    d, r = run_config.pgb_tower_dim, run_config.pgb_bilinear_rank
    assert gate_in == prior_dim + 2 * run_config.pgb_num_flags + run_config.pgb_num_mechanisms
    assert concat_dim == 4 * d + r + prior_dim + run_config.pgb_num_mechanisms
    assert trust_in == prior_dim
    print(f"\n[{variant} | {family} | seed {seed}] prior_dim={prior_dim} "
          f"gate_in={gate_in} pair_rep={concat_dim} trust_in={trust_in}")

    model, history = pgb_train(model, train_loader, val_loader, run_config, device)

    val_probs, val_labels = collect_predictions(model, val_loader, device)
    tnr_floor = getattr(run_config, "pgb_tnr_floor", 0.0)
    best_thresh, _ = tune_threshold(val_probs, val_labels, run_config.threshold_step, tnr_floor=tnr_floor)
    val_metrics = compute_extended_metrics(val_probs, val_labels, best_thresh)

    test_probs, test_labels = collect_predictions(model, test_loader, device)
    test_metrics = compute_extended_metrics(test_probs, test_labels, best_thresh)

    preds = {
        "val": _predictions_frame(val_fold, val_probs, val_labels, prior_table, prior_dim),
        "test": _predictions_frame(test_fold, test_probs, test_labels, prior_table, prior_dim),
    }

    heldout_metrics = None
    heldout_csv = getattr(config, "heldout_csv", HELDOUT_CSV)
    if os.path.exists(heldout_csv):
        heldout_df = pd.read_csv(heldout_csv)
        if "ground_truth" in heldout_df.columns:
            heldout_df = heldout_df.rename(columns={"ground_truth": "Outcome1"})
        temp_heldout_csv = os.path.join(checkpoint_dir, "temp_heldout.csv")
        heldout_df.to_csv(temp_heldout_csv, index=False)
        try:
            heldout_loader = make_loader(make_ds(temp_heldout_csv), shuffle=False)
            ho_probs, ho_labels = collect_predictions(model, heldout_loader, device)
            heldout_metrics = compute_extended_metrics(ho_probs, ho_labels, best_thresh)
            preds["heldout"] = _predictions_frame(heldout_df, ho_probs, ho_labels, prior_table, prior_dim)
        finally:
            if os.path.exists(temp_heldout_csv):
                os.remove(temp_heldout_csv)

    save_metrics(val_metrics, os.path.join(metrics_dir, "val_metrics.json"))
    save_metrics(test_metrics, os.path.join(metrics_dir, "test_metrics.json"))
    if heldout_metrics is not None:
        save_metrics(heldout_metrics, os.path.join(metrics_dir, "heldout_metrics.json"))
    save_metrics({"history": history}, os.path.join(metrics_dir, "training_history.json"))
    for split, frame in preds.items():
        frame.to_csv(os.path.join(metrics_dir, f"predictions_{split}.csv"), index=False)

    level_counts = {"train_leave_cluster_out": dict(Counter(train_levels))}
    level_counts.update({split: frame["prior_level"].value_counts().to_dict() for split, frame in preds.items()})
    _save_json(level_counts, os.path.join(metrics_dir, "prior_level_counts.json"))

    _save_json({
        "variant": variant,
        "family": family,
        "seed": seed,
        "prior_dim": prior_dim,
        "prior_table": table_cls.__name__,
        "prior_levels": "exact -> family -> nn -> global" if getattr(table_cls, "USE_FAMILY", True)
                        else "exact -> nn -> global",
        "gate_in": gate_in,
        "pair_rep_dim": concat_dim,
        "prior_trust_in": trust_in,
        "n_trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "max_epochs": run_config.pgb_max_epochs,
        "epochs_trained": len(history["val_pr_auc"]),
        "best_val_pr_auc_during_training": float(max(history["val_pr_auc"])),
        "threshold": float(best_thresh),
        "device": str(device),
    }, os.path.join(metrics_dir, "run_info.json"))

    print(f"[{variant} | {family} | seed {seed}] "
          f"val PR-AUC={val_metrics['pr_auc']:.4f}  test PR-AUC={test_metrics['pr_auc']:.4f}  "
          f"test MCC={test_metrics['mcc']:.4f}"
          + (f"  heldout PR-AUC={heldout_metrics['pr_auc']:.4f}" if heldout_metrics else ""))

    del model, encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="PGB excipient-prior ablation (4-level/5-d vs 3-level/5-d vs 3-level/1-d).")
    parser.add_argument("--variant", default="all", choices=list(VARIANTS) + ["all"])
    parser.add_argument("--family", default="morgan", choices=list(FAMILIES))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=None, help="override max epochs (smoke tests only)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="re-run seeds that already have test_metrics.json")
    args = parser.parse_args()

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    for seed in args.seeds:
        for variant in variants:
            done = os.path.exists(os.path.join(run_dir(args.results_root, variant, args.family, seed), "test_metrics.json"))
            if done and not args.overwrite:
                print(f"skip {variant} seed {seed} (already done; use --overwrite to re-run)")
                continue
            run_one(variant, args.family, seed, epochs=args.epochs, device_str=args.device,
                    results_root=args.results_root, checkpoint_root=args.checkpoint_root)


if __name__ == "__main__":
    main()
