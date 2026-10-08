"""Single-run training for PGB architecture on 60/20/20 train/val/test data.

Saves metrics to experiments/results/metrics_pgb/{family}/:
    val_metrics.json, test_metrics.json, heldout_metrics.json, training_history.json

Supports direct CLI execution:
    python -m experiments.pgb.single_run --family maccs
    python -m experiments.pgb.single_run --family molformer
    python -m experiments.pgb.single_run --family all
"""

import argparse
from experiments.pgb.config import PGBConfig, FAMILIES
from src.single_run import pgb_single_run


def run_family(name: str, args):
    config = PGBConfig()
    config.apply_family_spec(name)
    config.seed = args.seed

    if args.epochs:
        config.pgb_max_epochs = args.epochs
    if args.device:
        config.device = args.device

    print(f"\n{'='*60}")
    print(f"Training PGB model SINGLE RUN (60/20/20): {name}")
    print(f"{'='*60}")
    val_metrics, test_metrics = pgb_single_run(config)
    return val_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser(description="PGB single run on data/ 60/20/20 split.")
    parser.add_argument("--family", default="maccs",
                        choices=list(FAMILIES.keys()) + ["all"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    if args.family == "all":
        for name in FAMILIES:
            run_family(name, args)
    else:
        run_family(args.family, args)


if __name__ == "__main__":
    main()
