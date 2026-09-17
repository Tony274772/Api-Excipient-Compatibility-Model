"""Training loop – Section 11."""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.loss import get_loss_fn
from src.evaluate import get_val_pr_auc, collect_predictions


def compute_pos_weight(train_loader) -> torch.Tensor:
    """Compute pos_weight = num_neg / num_pos from training data."""
    labels = train_loader.dataset.df["Outcome1"].values
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    return torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, loss_fn, device, config, pos_weight=None):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch_device = {
            "api_smiles": batch["api_smiles"],
            "exc_smiles": batch["exc_smiles"],
            "api_desc": batch["api_desc"].to(device),
            "exc_desc": batch["exc_desc"].to(device),
            "exc_available": batch["exc_available"].to(device),
        }
        labels = batch["label"].to(device)
        sample_weight = batch["sample_weight"].to(device)

        optimizer.zero_grad()
        logits = model(batch_device)

        if config.loss == "weighted_bce":
            pw = pos_weight.to(device) if pos_weight is not None else None
            loss = loss_fn(logits, labels, sample_weight, pw=pw)
        else:
            loss = loss_fn(logits, labels, sample_weight)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(model, train_loader, val_loader, config, device):
    """Full training loop with early stopping and LR scheduling.
    
    Returns:
        model (best checkpoint restored), training history dict
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.lr_factor,
        patience=config.lr_patience,
    )

    loss_fn = get_loss_fn(config)
    pos_weight = compute_pos_weight(train_loader) if config.loss == "weighted_bce" else None

    best_pr_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_ckpt_path = os.path.join(config.checkpoint_dir, "best_model.pt")

    history = {
        "train_loss": [],
        "val_pr_auc": [],
        "lr": [],
    }

    print(f"\nTraining with encoder={config.encoder}, fusion={config.fusion}, loss={config.loss}")
    print(f"Device: {device}, Epochs: {config.max_epochs}, Batch size: {config.batch_size}")
    print(f"Checkpoint dir: {config.checkpoint_dir}\n")

    for epoch in range(1, config.max_epochs + 1):
        t0 = time.time()

        # Train
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, config, pos_weight
        )

        # Validate
        val_pr_auc = get_val_pr_auc(model, val_loader, device)

        # Scheduler step
        scheduler.step(val_pr_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{config.max_epochs} | "
            f"Loss: {avg_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | "
            f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
        )

        # Early stopping check
        if val_pr_auc > best_pr_auc + config.early_stop_min_delta:
            best_pr_auc = val_pr_auc
            best_epoch = epoch
            patience_counter = 0
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_pr_auc": val_pr_auc,
                "config": config,
            }, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={config.early_stop_patience})")
                break

    # Restore best checkpoint
    print(f"\nRestoring best model from epoch {best_epoch} (PR-AUC={best_pr_auc:.4f})")
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, history
