"""CLI entry point – single train/eval run.

Usage:
    python main.py                                   # MoLFormer + cross_attn + ASL (default)
    python main.py --encoder pretrained_gin           # GIN encoder
    python main.py --encoder fixed_vector --fusion concat --fixed_vector_path data/mol2vec.csv
    python main.py --encoder molformer --loss focal   # loss ablation
    python main.py --mode cv                          # 5-fold cross-validation
    python main.py --encoder gine --fusion cross_attn --loss asl
    python main.py --encoder dmpnn_chemprop --fusion cross_attn --loss asl
"""

import argparse
import json
import os
import sys

import torch
import pandas as pd

from src.config import Config
from src.utils import seed_everything, get_device
from src.encoders import ENCODER_REGISTRY
from src.model import CompatibilityModel
from src.dataset import build_dataloaders
from src.train import train
from src.evaluate import evaluate_model, save_metrics
from src.cross_validate import cross_validate


# GNN encoder names that use the new trainable encoder path
GNN_ENCODERS = {"dmpnn_chemprop", "dmpnn_scratch", "attentivefp", "gine", "gatv2", "pna"}


def build_encoder(config, device):
    """Instantiate the selected encoder."""
    encoder_cls = ENCODER_REGISTRY[config.encoder]

    if config.encoder == "molformer":
        encoder = encoder_cls(config.molformer_model_path, device=str(device))
    elif config.encoder == "pretrained_gin":
        encoder = encoder_cls(config.gin_pretrained_name, device=str(device))
    elif config.encoder == "chemberta":
        encoder = encoder_cls(config.chemberta_model_path, device=str(device))
    elif config.encoder == "fixed_vector":
        encoder = encoder_cls(
            source=config.fixed_vector_source,
            vector_path=config.fixed_vector_path,
            device=str(device),
        )
    elif config.encoder in GNN_ENCODERS:
        # All new GNN encoders take (config, device)
        encoder = encoder_cls(config, device=str(device))
    else:
        raise ValueError(f"Unknown encoder: {config.encoder}")

    encoder.to(device)
    config.encoder_output_dim = encoder.output_dim
    return encoder


def parse_args():
    parser = argparse.ArgumentParser(description="API-Excipient Compatibility Model")

    # Mode
    parser.add_argument("--mode", choices=["train", "cv"], default="train",
                        help="'train' for single train/eval, 'cv' for 5-fold CV")
    parser.add_argument("--split_type", choices=["cluster", "random"], default=None,
                        help="Type of data split: 'cluster' (default) or 'random'")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Explicit folder name for checkpoint and metrics (e.g. molformer_cross_attn_asl)")

    # Encoder (Axis A)
    parser.add_argument("--encoder", choices=list(ENCODER_REGISTRY.keys()), default=None)
    parser.add_argument("--fixed_vector_source", choices=["mol2vec", "pubchemfp", "rdkit_descriptors", "morgan", "maccs"], default=None)
    parser.add_argument("--fixed_vector_path", default=None)

    # Fusion (Axis B)
    parser.add_argument("--fusion", choices=["cross_attn", "concat"], default=None)
    parser.add_argument("--pooling", choices=["global_gated_attention", "cls", "explicit_pairwise"], default=None,
                        help="Pooling mode for cross-attention: 'global_gated_attention', 'cls', or 'explicit_pairwise'")
    parser.add_argument("--pairwise", action="store_true",
                        help="Use explicit API-token × Excipient-token pairwise pooling (sets --pooling explicit_pairwise)")

    # Loss (Axis C)
    parser.add_argument("--loss", choices=["bce", "weighted_bce", "focal", "asl"], default=None)

    # Training
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)

    # Flags
    parser.add_argument("--no_descriptors", action="store_true")
    parser.add_argument("--no_balanced_sampler", action="store_true")

    return parser.parse_args()


def apply_args_to_config(args, config):
    """Override config defaults with CLI arguments."""
    if args.encoder is not None:
        config.encoder = args.encoder
    if args.fixed_vector_source is not None:
        config.fixed_vector_source = args.fixed_vector_source
    if args.fixed_vector_path is not None:
        config.fixed_vector_path = args.fixed_vector_path
    if args.fusion is not None:
        config.fusion = args.fusion
    if getattr(args, "pairwise", False):
        config.pooling = "explicit_pairwise"
    elif args.pooling is not None:
        config.pooling = args.pooling
    if args.loss is not None:
        config.loss = args.loss
    if args.lr is not None:
        config.lr = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    if args.no_descriptors:
        config.use_descriptors = False
    if args.no_balanced_sampler:
        config.use_balanced_sampler = False
    if args.split_type is not None:
        config.split_type = args.split_type


def main():
    args = parse_args()
    config = Config()
    apply_args_to_config(args, config)
    config.resolve_checkpoint_paths()
    config.resolve_csv_paths()

    if args.run_name:
        prefix = "random_split/" if config.split_type == "random" else ""
        config.checkpoint_dir = f"checkpoints/{prefix}{args.run_name}"
        config.metrics_dir = f"metrics/{prefix}{args.run_name}"

    seed_everything(config.seed)
    device = get_device(config.device)

    print(f"Config: encoder={config.encoder}, fusion={config.fusion}, loss={config.loss}")
    print(f"Device: {device}")
    print(f"Metrics dir: {config.metrics_dir}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")

    if args.mode == "cv":
        cross_validate(config)
        return

    # --- Single train/eval run ---

    # Compute positive prior from training data (needed before encoder for PNA)
    train_df = pd.read_csv(config.train_csv)
    config.positive_prior = float(train_df["Outcome1"].mean())
    print(f"Positive prior (from train): {config.positive_prior:.4f}")

    # PNA requires degree histogram computed from training data BEFORE encoder construction
    if config.encoder == "pna":
        print("Computing PNA degree histogram from training data...")
        from src.encoders.pna_utils import degree_histogram_from_dataframe
        deg_hist = degree_histogram_from_dataframe(train_df)
        config.pna_degree_hist = deg_hist.tolist()
        print(f"PNA degree histogram: {len(config.pna_degree_hist)} bins, max degree = {len(config.pna_degree_hist) - 1}")

    # Record CheMeleon provenance
    if config.encoder == "dmpnn_chemprop":
        config.chemeleon_checkpoint_path = config.dmpnn_pretrained_checkpoint
        config.chemeleon_checkpoint_md5 = "6a80b54fdb7de37ef0374d302f01e8ce"

    # Build encoder
    encoder = build_encoder(config, device)

    # Force concat fusion for non-sequence encoders
    if not encoder.is_sequence_capable:
        config.fusion = "concat"
        # Re-resolve paths since fusion may have changed
        config.resolve_checkpoint_paths()

    # Build model
    model = CompatibilityModel(config, encoder)
    model.init_bias(config.positive_prior)
    model.to(device)

    # Count trainable parameters
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_trainable:,} trainable / {n_total:,} total")

    # Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(config)
    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

    # Train — use separate LR groups for CheMeleon pretrained encoder
    if config.encoder == "dmpnn_chemprop":
        model, history = _train_with_separate_lr(model, train_loader, val_loader, config, device)
    else:
        model, history = train(model, train_loader, val_loader, config, device)

    # Evaluate
    val_metrics, test_metrics, best_thresh = evaluate_model(
        model, val_loader, test_loader, device, config
    )

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best threshold: {best_thresh:.4f}")
    print(f"\nValidation:")
    for k, v in val_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  confusion_matrix: {val_metrics['confusion_matrix']}")

    print(f"\nTest:")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  confusion_matrix: {test_metrics['confusion_matrix']}")

    # Save metrics
    os.makedirs(config.metrics_dir, exist_ok=True)
    save_metrics(val_metrics, os.path.join(config.metrics_dir, "val_metrics.json"))
    save_metrics(test_metrics, os.path.join(config.metrics_dir, "test_metrics.json"))
    save_metrics({"history": history}, os.path.join(config.metrics_dir, "training_history.json"))

    print(f"\nMetrics saved to {config.metrics_dir}/")


def _train_with_separate_lr(model, train_loader, val_loader, config, device):
    """Train with separate learning-rate groups for CheMeleon encoder vs downstream.

    CheMeleon message-passing: lr = dmpnn_encoder_lr (1e-5 default)
    Everything else:            lr = config.lr (1.5e-4 default)
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from src.loss import get_loss_fn
    from src.evaluate import get_val_pr_auc
    from src.train import compute_pos_weight, train_one_epoch
    import time

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Separate encoder and downstream parameters
    encoder_params = []
    downstream_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            encoder_params.append(param)
        else:
            downstream_params.append(param)

    param_groups = [
        {"params": encoder_params, "lr": config.dmpnn_encoder_lr},
        {"params": downstream_params, "lr": config.lr},
    ]

    optimizer = AdamW(param_groups, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=config.lr_factor, patience=config.lr_patience)

    loss_fn = get_loss_fn(config)
    pos_weight = compute_pos_weight(train_loader) if config.loss == "weighted_bce" else None

    best_pr_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_ckpt_path = os.path.join(config.checkpoint_dir, "best_model.pt")

    history = {"train_loss": [], "val_pr_auc": [], "lr": []}

    print(f"\nTraining with encoder={config.encoder}, fusion={config.fusion}, loss={config.loss}")
    print(f"CheMeleon encoder LR: {config.dmpnn_encoder_lr}, Downstream LR: {config.lr}")
    print(f"Device: {device}, Epochs: {config.max_epochs}, Batch size: {config.batch_size}")
    print(f"Checkpoint dir: {config.checkpoint_dir}\n")

    for epoch in range(1, config.max_epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, config, pos_weight)
        val_pr_auc = get_val_pr_auc(model, val_loader, device)
        scheduler.step(val_pr_auc)
        current_lr = optimizer.param_groups[1]["lr"]  # downstream LR

        history["train_loss"].append(avg_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{config.max_epochs} | "
            f"Loss: {avg_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | "
            f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
        )

        if val_pr_auc > best_pr_auc + config.early_stop_min_delta:
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
            if patience_counter >= config.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={config.early_stop_patience})")
                break

    print(f"\nRestoring best model from epoch {best_epoch} (PR-AUC={best_pr_auc:.4f})")
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, history


if __name__ == "__main__":
    main()
