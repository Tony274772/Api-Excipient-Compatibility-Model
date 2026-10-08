import os
import copy
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

from src.config import Config
from src.utils import seed_everything, get_device
from src.dataset import PGBDataset, pgb_collate_fn, PGBNormStats
from src.pgb_prior import ExcipientPriorTable, build_leave_cluster_out_prior_vectors
from src.model import pgb_model_from_config
from src.train import pgb_train
from src.evaluate import collect_predictions, tune_threshold, save_metrics
from src.cross_validate import build_encoder, butina_cluster


def compute_extended_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Computes standard metric suite matching experiments/results/metrics_bilinear format."""
    preds = (probs >= threshold).astype(int)
    labels_int = labels.astype(int)

    pr_auc = float(average_precision_score(labels_int, probs))
    try:
        roc_auc = float(roc_auc_score(labels_int, probs))
    except Exception:
        roc_auc = 0.5

    f1 = float(f1_score(labels_int, preds, zero_division=0))
    mcc = float(matthews_corrcoef(labels_int, preds))
    precision = float(precision_score(labels_int, preds, zero_division=0))
    recall = float(recall_score(labels_int, preds, zero_division=0))
    accuracy = float(accuracy_score(labels_int, preds))
    cm = confusion_matrix(labels_int, preds).tolist()
    tp = int(cm[1][1]) if len(cm) > 1 and len(cm[1]) > 1 else 0
    fp = int(cm[0][1]) if len(cm) > 0 and len(cm[0]) > 1 else 0

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "confusion_matrix": cm,
    }


def pgb_single_run(config: Config):
    """
    Trains PGBCompatibilityModel on the standard 60/20/20 split from data/ folder.
    Saves metrics to experiments/results/metrics_pgb/{family}/:
      - val_metrics.json
      - test_metrics.json
      - heldout_metrics.json
      - training_history.json
    """
    seed_everything(config.seed)
    device = get_device(config.device)

    # Resolve family name
    family_name = getattr(config, "family", getattr(config, "fixed_vector_source", config.encoder))

    # 1. Load Data from data/ folder (60/20/20 split)
    train_csv = config.train_csv or os.path.join(config.data_dir, "train.csv")
    val_csv   = config.val_csv   or os.path.join(config.data_dir, "val.csv")
    test_csv  = config.test_csv  or os.path.join(config.data_dir, "test.csv")
    heldout_csv = getattr(config, "heldout_csv", "held_out_testset/held_out_test_set.csv")

    train_fold = pd.read_csv(train_csv)
    val_fold   = pd.read_csv(val_csv)
    test_fold  = pd.read_csv(test_csv)

    print(f"\nLoaded datasets from {config.data_dir} (60/20/20 split):")
    print(f"  Train: {len(train_fold)} rows")
    print(f"  Val:   {len(val_fold)} rows")
    print(f"  Test:  {len(test_fold)} rows")

    # Metrics directory: experiments/results/metrics_pgb/{family}
    metrics_root = getattr(config, "pgb_metrics_root", "experiments/results/metrics_pgb")
    metrics_dir = os.path.join(metrics_root, family_name)
    os.makedirs(metrics_dir, exist_ok=True)

    # Checkpoint directory: checkpoints/pgb/{family}
    ckpt_root = getattr(config, "pgb_checkpoint_root", "checkpoints/pgb")
    checkpoint_dir = os.path.join(ckpt_root, family_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    run_config = copy.deepcopy(config)
    run_config.train_csv = train_csv
    run_config.val_csv   = val_csv
    run_config.test_csv  = test_csv
    run_config.checkpoint_dir = checkpoint_dir
    run_config.metrics_dir = metrics_dir
    run_config.positive_prior = float(train_fold["Outcome1"].mean())
    run_config.use_balanced_sampler = False
    run_config.loss = "asym_focal"

    # 2. Build encoder
    encoder = build_encoder(run_config, device)
    if not encoder.is_sequence_capable:
        run_config.fusion = "concat"

    # 3. Build PGB norm stats from unique training SMILES only
    all_train_smiles = list(set(
        train_fold["API_Smiles"].dropna().tolist() +
        train_fold["Excipient_Smiles"].dropna().tolist()
    ))
    pgb_norm = PGBNormStats.from_smiles_list(all_train_smiles)
    pgb_norm.save(os.path.join(checkpoint_dir, "pgb_norm_stats.npz"))

    # 4. Build excipient prior from training rows (leave-cluster-out for training)
    prior_table = ExcipientPriorTable()
    prior_table.fit(train_fold, leave_out_cluster_id=None)
    prior_table.save(os.path.join(checkpoint_dir, "prior_table.json"))

    unique_apis = train_fold["API_Smiles"].unique().tolist()
    smiles_to_cluster = butina_cluster(unique_apis)
    train_fold_copy = train_fold.copy()
    train_fold_copy["cluster_id"] = train_fold_copy["API_Smiles"].map(smiles_to_cluster)
    train_priors = build_leave_cluster_out_prior_vectors(train_fold_copy, cluster_col="cluster_id")

    fv_enc = encoder if run_config.encoder == "fixed_vector" else None

    # 5. Datasets & Loaders
    train_ds = PGBDataset(
        train_csv, run_config.encoder,
        getattr(run_config, "fixed_vector_source", "maccs"),
        prior_table, pgb_norm, fixed_vector_encoder=fv_enc, row_priors=train_priors
    )
    val_ds = PGBDataset(
        val_csv, run_config.encoder,
        getattr(run_config, "fixed_vector_source", "maccs"),
        prior_table, pgb_norm, fixed_vector_encoder=fv_enc
    )
    test_ds = PGBDataset(
        test_csv, run_config.encoder,
        getattr(run_config, "fixed_vector_source", "maccs"),
        prior_table, pgb_norm, fixed_vector_encoder=fv_enc
    )

    train_loader = DataLoader(
        train_ds, batch_size=run_config.pgb_batch_size,
        shuffle=True, collate_fn=pgb_collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=run_config.pgb_batch_size,
        shuffle=False, collate_fn=pgb_collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=run_config.pgb_batch_size,
        shuffle=False, collate_fn=pgb_collate_fn, num_workers=0
    )

    # 6. Build model & Train
    model = pgb_model_from_config(run_config, encoder)
    model.to(device)

    model, history = pgb_train(model, train_loader, val_loader, run_config, device)

    # 7. Collect predictions & tune threshold on validation set
    val_probs, val_labels = collect_predictions(model, val_loader, device)
    tnr_floor = getattr(run_config, "pgb_tnr_floor", 0.0)
    best_thresh, _ = tune_threshold(val_probs, val_labels, run_config.threshold_step, tnr_floor=tnr_floor)

    val_metrics = compute_extended_metrics(val_probs, val_labels, best_thresh)

    # Evaluate on test set (60/20/20 test split)
    test_probs, test_labels = collect_predictions(model, test_loader, device)
    test_metrics = compute_extended_metrics(test_probs, test_labels, best_thresh)

    # Evaluate on 24-pairs held-out benchmark set if available
    heldout_metrics = None
    if os.path.exists(heldout_csv):
        heldout_df = pd.read_csv(heldout_csv)
        if "ground_truth" in heldout_df.columns:
            heldout_df = heldout_df.rename(columns={"ground_truth": "Outcome1"})
        temp_heldout_csv = os.path.join(checkpoint_dir, "temp_heldout.csv")
        heldout_df.to_csv(temp_heldout_csv, index=False)
        try:
            heldout_ds = PGBDataset(
                temp_heldout_csv, run_config.encoder,
                getattr(run_config, "fixed_vector_source", "maccs"),
                prior_table, pgb_norm, fixed_vector_encoder=fv_enc
            )
            heldout_loader = DataLoader(
                heldout_ds, batch_size=run_config.pgb_batch_size,
                shuffle=False, collate_fn=pgb_collate_fn, num_workers=0
            )
            heldout_probs, heldout_labels = collect_predictions(model, heldout_loader, device)
            heldout_metrics = compute_extended_metrics(heldout_probs, heldout_labels, best_thresh)
        finally:
            if os.path.exists(temp_heldout_csv):
                os.remove(temp_heldout_csv)

    # 8. Save metrics to experiments/results/metrics_pgb/{family}/
    save_metrics(val_metrics, os.path.join(metrics_dir, "val_metrics.json"))
    save_metrics(test_metrics, os.path.join(metrics_dir, "test_metrics.json"))
    if heldout_metrics is not None:
        save_metrics(heldout_metrics, os.path.join(metrics_dir, "heldout_metrics.json"))
    save_metrics({"history": history}, os.path.join(metrics_dir, "training_history.json"))

    print(f"\n{'='*60}")
    print(f"PGB RESULTS — {family_name.upper()}")
    print(f"{'='*60}")
    print(f"Optimal Val Threshold: {best_thresh:.4f}")
    print(f"Validation Metrics: PR-AUC={val_metrics['pr_auc']:.4f}, ROC-AUC={val_metrics['roc_auc']:.4f}, F1={val_metrics['f1']:.4f}, MCC={val_metrics['mcc']:.4f}")
    print(f"Test Set Metrics:   PR-AUC={test_metrics['pr_auc']:.4f}, ROC-AUC={test_metrics['roc_auc']:.4f}, F1={test_metrics['f1']:.4f}, MCC={test_metrics['mcc']:.4f}")
    if heldout_metrics is not None:
        print(f"Held-Out 24-Pairs:  PR-AUC={heldout_metrics['pr_auc']:.4f}, ROC-AUC={heldout_metrics['roc_auc']:.4f}, F1={heldout_metrics['f1']:.4f}, MCC={heldout_metrics['mcc']:.4f}")
    print(f"\nAll metrics saved to: {metrics_dir}/")
    print(f"Checkpoints saved to: {checkpoint_dir}/")

    # Clean up GPU memory
    del model, encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_metrics, test_metrics
