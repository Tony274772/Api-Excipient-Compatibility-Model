"""
run_pgb_single.py
Entry point for training the 9 Prior-Gated Bilinear (PGB) models exactly once (no CV).
It uses an 80/20 train/val split on start_dataset.csv, and evaluates on the 24-pairs held_out_testset.csv.

Run one family at a time:
    python run_pgb_single.py --family maccs
    python run_pgb_single.py --family molformer
    python run_pgb_single.py --family all
"""

import argparse
from src.config import Config
from src.single_run import pgb_single_run
from run_pgb import FAMILIES

def run_family(name: str, args):
    spec = FAMILIES[name]
    config = Config()
    config.use_pgb_head = True
    config.split_type   = "cluster"
    config.seed         = args.seed

    for k, v in spec.items():
        setattr(config, k, v)

    if args.epochs:
        config.pgb_max_epochs = args.epochs
    if args.device:
        config.device = args.device

    config.resolve_paths()

    print(f"\n{'='*60}")
    print(f"Training PGB model SINGLE RUN: {name}")
    print(f"{'='*60}")
    val_metrics, test_metrics = pgb_single_run(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
