"""Training entry point for Gated Bilinear architecture.

Implements Sections 7 (Training Recipe):
- Separate learning rate groups for deep vs. fixed families
- Learning rate scheduling and early stopping based on PR-AUC
- Wires up the GBCompatibilityDataset, GatedBilinearModel, and AsymmetricFocalLoss
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from experiments.gated_bilinear.config import GBConfig
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from experiments.gated_bilinear.dataset import build_gb_dataloaders, build_gb_heldout_dataloader
from experiments.gated_bilinear.features import compute_gb_descriptor_norm_stats
from experiments.gated_bilinear.loss import AsymmetricFocalLoss
from experiments.gated_bilinear.model import GatedBilinearModel, FIXED_VECTOR_FAMILIES
from experiments.gated_bilinear.prior import ExcipientPriorTable
from experiments.gated_bilinear.evaluate_gb import (
    evaluate_gb_model,
    evaluate_heldout_gb,
    save_gb_metrics,
    get_val_pr_auc_gb,
)

from src.encoders import ENCODER_REGISTRY
from src.utils import seed_everything, get_device

# For GNN offline check / initialization
GNN_ENCODERS = {"dmpnn_chemprop", "dmpnn_scratch", "attentivefp", "gine", "gatv2", "pna", "pretrained_gat"}


def compute_pos_weight(train_df: pd.DataFrame) -> float:
    """Compute n_neg / n_pos from training dataframe."""
    labels = train_df["Outcome1"].values
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    return float(n_neg / max(n_pos, 1))


def train_one_epoch_gb(model, loader, optimizer, loss_fn, device, config):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        labels = batch["label"]
        sample_weight = batch["sample_weight"]

        optimizer.zero_grad()
        logits = model(batch)

        loss = loss_fn(logits, labels, sample_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gb_grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def build_optimizer_and_scheduler(model, config):
    """Build optimizer with appropriate LR groups."""
    if config.gb_family in FIXED_VECTOR_FAMILIES:
        # Single LR for fixed-vector families
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.gb_lr_fixed,
            weight_decay=config.gb_weight_decay,
        )
    else:
        # Transformer / Graph families: separate LR for new vs kept layers
        new_params = []
        kept_params = []
        
        # Everything in 'head', 'api_tower', 'exc_tower' is new
        # Everything in 'encoder', 'api_proj', 'exc_proj', cross_attn, pools is kept
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("head.") or name.startswith("api_tower.") or name.startswith("exc_tower."):
                new_params.append(param)
            else:
                kept_params.append(param)
                
        param_groups = [
            {"params": new_params, "lr": config.gb_lr_new},
            {"params": kept_params, "lr": config.gb_lr_kept},
        ]
        
        optimizer = AdamW(
            param_groups,
            weight_decay=config.gb_weight_decay,
        )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.gb_lr_factor,
        patience=config.gb_lr_patience,
    )
    
    return optimizer, scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Train Gated Bilinear Model")
    parser.add_argument("--family", choices=[
        "maccs", "morgan", "pubchem", "mol2vec",
        "molformer", "chemberta",
        "gin", "gat", "dmpnn"
    ], required=True, help="Model family to train")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name identifier (defaults to family name, e.g. gin)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = GBConfig()
    config.gb_family = args.family
    model_name = args.model_name or config.gb_family
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
        
    # Mapping family to encoder type
    family_to_encoder = {
        "maccs": "fixed_vector",
        "morgan": "fixed_vector",
        "pubchem": "fixed_vector",
        "mol2vec": "fixed_vector",
        "molformer": "molformer",
        "chemberta": "chemberta",
        "gin": "pretrained_gin",
        "gat": "pretrained_gat",
        "dmpnn": "dmpnn_chemprop",
    }
    config.encoder = family_to_encoder[config.gb_family]
    
    config.resolve_checkpoint_paths()
    config.resolve_csv_paths()
    config.resolve_gb_paths()
    if args.model_name:
        config.gb_metrics_dir = f"experiments/results/metrics_bilinear/{args.model_name}"

    seed_everything(config.seed)
    device = get_device(config.device)

    print(f"==================================================")
    print(f"GATED BILINEAR MODEL TRAINING")
    print(f"Family: {config.gb_family.upper()}")
    print(f"Encoder: {config.encoder}")
    print(f"Device: {device}")
    print(f"==================================================")

    # 1. Load training dataframe to build prior table and stats
    train_df = pd.read_csv(config.train_csv)
    train_exc_smiles = train_df["Excipient_Smiles"].fillna("").astype(str).tolist()
    train_api_smiles = train_df["API_Smiles"].fillna("").astype(str).tolist()
    train_labels = train_df["Outcome1"].astype(int).tolist()
    
    # Try to extract cluster info
    cluster_col = next((c for c in ["API_Cluster", "api_cluster", "Cluster"] if c in train_df.columns), None)
    train_clusters = train_df[cluster_col].tolist() if cluster_col else None
    
    config.base_rate = float(np.mean(train_labels))
    print(f"Global positive rate: {config.base_rate:.4f}")
    
    # Build prior table
    print("Building excipient prior table...")
    prior_table = ExcipientPriorTable(
        train_exc_smiles=train_exc_smiles,
        train_labels=train_labels,
        train_api_clusters=train_clusters,
        global_rate=config.base_rate,
    )
    
    # Compute descriptor normalization stats
    print("Computing descriptor normalization stats...")
    desc_stats = compute_gb_descriptor_norm_stats(train_api_smiles + train_exc_smiles)

    # 2. Build DataLoaders
    print("Building DataLoaders...")
    train_loader, val_loader, test_loader = build_gb_dataloaders(config, prior_table, desc_stats)
    heldout_loader = build_gb_heldout_dataloader(config, prior_table, desc_stats)
    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test, {len(heldout_loader.dataset)} held-out")

    # 3. Build Encoder
    print("Building Encoder...")
    encoder_cls = ENCODER_REGISTRY[config.encoder]
    if config.encoder == "molformer":
        encoder = encoder_cls(config.molformer_model_path, device=str(device))
    elif config.encoder == "pretrained_gin":
        encoder = encoder_cls(config.gin_pretrained_name, device=str(device))
    elif config.encoder == "chemberta":
        encoder = encoder_cls(config.chemberta_model_path, device=str(device))
    elif config.encoder == "fixed_vector":
        encoder = encoder_cls(source="mol2vec", device=str(device)) # Dummy source, not used in PGB
    elif config.encoder in GNN_ENCODERS:
        encoder = encoder_cls(config, device=str(device))
    else:
        raise ValueError(f"Unknown encoder: {config.encoder}")
    
    encoder.to(device)

    # 4. Build Model
    print("Building Gated Bilinear Model...")
    model = GatedBilinearModel(config, encoder, config.gb_family)
    model.to(device)
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    # 5. Build Loss, Optimizer, Scheduler
    pos_weight = compute_pos_weight(train_df)
    loss_fn = AsymmetricFocalLoss(
        gamma_pos=config.gb_afl_gamma_pos,
        gamma_neg=config.gb_afl_gamma_neg,
        pos_weight=pos_weight,
    )
    
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    
    max_epochs = config.gb_max_epochs_fixed if config.gb_family in FIXED_VECTOR_FAMILIES else config.gb_max_epochs_deep

    # 6. Training Loop
    os.makedirs(config.gb_checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(config.gb_checkpoint_dir, "best_model.pt")
    
    best_pr_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_pr_auc": [], "lr": []}
    
    print("\nStarting Training...")
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        avg_loss = train_one_epoch_gb(model, train_loader, optimizer, loss_fn, device, config)
        val_pr_auc = get_val_pr_auc_gb(model, val_loader, device)

        scheduler.step(val_pr_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{max_epochs} | Loss: {avg_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

        if val_pr_auc > best_pr_auc + config.gb_early_stop_min_delta:
            best_pr_auc = val_pr_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_pr_auc": val_pr_auc,
                "config": config,
            }, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.gb_early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={config.gb_early_stop_patience})")
                break

    # 7. Evaluation
    print(f"\nRestoring best model from epoch {best_epoch} (PR-AUC={best_pr_auc:.4f})")
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print("\nEvaluating on Validation and Test sets...")
    val_metrics, test_metrics, best_thresh, screen_thresh = evaluate_gb_model(model, val_loader, test_loader, device, config)

    print("\nEvaluating on Held-Out 24-Pair Set...")
    heldout_metrics, heldout_df = evaluate_heldout_gb(
        model=model,
        heldout_loader=heldout_loader,
        device=device,
        threshold=best_thresh,
        model_name=model_name,
        output_csv_path=config.gb_heldout_predictions_csv,
        heldout_source_csv=config.gb_heldout_csv,
    )

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (PGB - {model_name.upper()})")
    print(f"{'='*60}")
    print(f"Main threshold (MCC w/ TNR >= {config.gb_tnr_floor}): {best_thresh:.4f}")
    print(f"Screen threshold (Recall >= {config.gb_screen_min_recall}): {screen_thresh:.4f}")
    
    print(f"\nTest Metrics (at Main threshold {best_thresh:.4f}):")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  confusion_matrix: {test_metrics['confusion_matrix']}")

    print(f"\nHeld-Out 24-Pair Metrics (at Main threshold {best_thresh:.4f}):")
    for k, v in heldout_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  confusion_matrix: {heldout_metrics['confusion_matrix']}")

    # Save metrics
    os.makedirs(config.gb_metrics_dir, exist_ok=True)
    save_gb_metrics(val_metrics, os.path.join(config.gb_metrics_dir, "val_metrics.json"))
    save_gb_metrics(test_metrics, os.path.join(config.gb_metrics_dir, "test_metrics.json"))
    save_gb_metrics(heldout_metrics, os.path.join(config.gb_metrics_dir, "heldout_metrics.json"))
    save_gb_metrics({"history": history}, os.path.join(config.gb_metrics_dir, "training_history.json"))
    
    print(f"\nMetrics saved to: {config.gb_metrics_dir}/")
    print(f"Held-out predictions accumulated in: {config.gb_heldout_predictions_csv}")


if __name__ == "__main__":
    main()
