"""Training loop – Section 11."""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

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

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
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


def pgb_train(model, train_loader, val_loader, config, device):
    """
    Training loop for PGBCompatibilityModel.
    Uses two LR groups: new PGB layers at pgb_lr, kept encoder layers at pgb_encoder_lr.
    Loss: asym_focal with single pos_weight. No balanced sampler.
    Returns: model (best ckpt restored), history dict.
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Identify parameter groups by name
    pgb_params   = []
    enc_params   = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Kept layers: cross-attention, projection, pooling (carried from CompatibilityModel)
        if any(k in name for k in [
            "api_proj", "exc_proj", "cross_attn", "ln_exc", "ln_api",
            "api_pool", "exc_pool", "api_pool_fusion", "exc_pool_fusion",
            "exc_placeholder", "exc_global_placeholder",
        ]):
            enc_params.append(param)
        else:
            pgb_params.append(param)

    optimizer = AdamW([
        {"params": pgb_params, "lr": config.pgb_lr},
        {"params": enc_params, "lr": config.pgb_encoder_lr if enc_params else config.pgb_lr},
    ], weight_decay=config.pgb_weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=config.pgb_lr_patience)

    # Loss function
    loss_fn = get_loss_fn(config)
    pos_weight = compute_pos_weight(train_loader)

    best_pr_auc = -1.0
    best_epoch  = 0
    patience_counter = 0
    best_ckpt_path = os.path.join(config.checkpoint_dir, "best_model.pt")
    history = {"train_loss": [], "val_pr_auc": [], "lr": []}

    for epoch in range(1, config.pgb_max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        # Initialize tqdm progress bar for the epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.pgb_max_epochs}", leave=False)
        for batch in pbar:
            batch_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                            for k, v in batch.items()}
            labels = batch_device.pop("label")
            sw     = batch_device.pop("sample_weight")

            optimizer.zero_grad()
            logits = model(batch_device)
            loss   = loss_fn(logits, labels, sw, pw=pos_weight.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss   = total_loss / max(n_batches, 1)
        val_pr_auc = _pgb_val_pr_auc(model, val_loader, device)
        scheduler.step(val_pr_auc)
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["lr"].append(current_lr)

        print(f"PGB Epoch {epoch:3d}/{config.pgb_max_epochs} | "
              f"Loss: {avg_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | LR: {current_lr:.2e}")

        if val_pr_auc > best_pr_auc + config.early_stop_min_delta:
            best_pr_auc = val_pr_auc
            best_epoch  = epoch
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_pr_auc": val_pr_auc, "config": config}, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.pgb_early_stop_patience:
                print(f"\nPGB early stopping at epoch {epoch}")
                break

    print(f"\nPGB restoring best model from epoch {best_epoch} (PR-AUC={best_pr_auc:.4f})")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, history


def _pgb_val_pr_auc(model, val_loader, device):
    """Quick PR-AUC for PGB models — batch keys differ from baseline."""
    from sklearn.metrics import average_precision_score
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            batch_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                            for k, v in batch.items()}
            labels = batch_device.pop("label")
            _ = batch_device.pop("sample_weight", None)
            logits = model(batch_device)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    return float(average_precision_score(labels.astype(int), probs))
