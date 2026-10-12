"""Single-run training for ChemBERTa & DMPNN with global_gated_attention pooling.

Uses the exact same group-split data as earlier:
  - Train:   data/train.csv (2,030 rows)
  - Val:     data/val.csv (842 rows)
  - Test:    data/test.csv (672 rows)
  - Held-out: held_out_testset/held_out_test_set.csv (24 pairs)

Saves metrics to experiments/results/metrics_pgb/global_gated/{family}/:
  val_metrics.json, test_metrics.json, heldout_metrics.json, training_history.json

Saves checkpoints to checkpoints/pgb/global_gated/{family}/:
  best_model.pt, pgb_norm_stats.npz, prior_table.json

CLI Usage:
  python -m experiments.pgb.global_gated.single_run --family chemberta
  python -m experiments.pgb.global_gated.single_run --family dmpnn
  python -m experiments.pgb.global_gated.single_run --family all
"""

import argparse
from experiments.pgb.global_gated.config import GATED_FAMILIES, PGBGlobalGatedConfig
from src.single_run import pgb_single_run


def run_family(name: str, args):
    config = PGBGlobalGatedConfig()
    config.apply_family_spec(name)
    config.seed = args.seed

    if args.epochs:
        config.pgb_max_epochs = args.epochs
    if args.device:
        config.device = args.device

    print(f"\n{'='*70}")
    print(f"Training PGB model with GLOBAL_GATED_ATTENTION pooling: {name.upper()}")
    print(f"  Encoder: {config.encoder}")
    print(f"  Fusion:  {config.fusion}")
    print(f"  Pooling: {config.pooling}")
    print(f"  Metrics dir:     {config.metrics_dir}")
    print(f"  Checkpoint dir:  {config.checkpoint_dir}")
    print(f"{'='*70}")

    val_metrics, test_metrics = pgb_single_run(config)
    return val_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train ChemBERTa / DMPNN PGB models with global_gated_attention pooling."
    )
    parser.add_argument(
        "--family",
        default="all",
        choices=list(GATED_FAMILIES.keys()) + ["all"],
        help="Family to train: 'chemberta', 'dmpnn', or 'all'",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. 'cuda', 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.family == "all":
        for name in GATED_FAMILIES:
            run_family(name, args)
    else:
        run_family(args.family, args)


if __name__ == "__main__":
    main()
