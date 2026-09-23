"""CLI entry point – single train/eval run.

Usage:
    python main.py                                   # MoLFormer + cross_attn + ASL (default)
    python main.py --encoder pretrained_gin           # GIN encoder
    python main.py --encoder fixed_vector --fusion concat --fixed_vector_path data/mol2vec.csv
    python main.py --encoder molformer --loss focal   # loss ablation
    python main.py --mode cv                          # 5-fold cross-validation
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

    # Build encoder
    encoder = build_encoder(config, device)

    # Force concat fusion for non-sequence encoders
    if not encoder.is_sequence_capable:
        config.fusion = "concat"
        # Re-resolve paths since fusion may have changed
        config.resolve_checkpoint_paths()

    # Compute positive prior from training data
    train_df = pd.read_csv(config.train_csv)
    config.positive_prior = float(train_df["Outcome1"].mean())
    print(f"Positive prior (from train): {config.positive_prior:.4f}")

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

    # Train
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


if __name__ == "__main__":
    main()
