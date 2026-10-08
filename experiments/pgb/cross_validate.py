"""5-fold cross-validation for PGB architecture.

Supports direct CLI execution:
    python -m experiments.pgb.cross_validate --family maccs
    python -m experiments.pgb.cross_validate --family molformer
    python -m experiments.pgb.cross_validate --family all
"""

import argparse
from experiments.pgb.config import PGBConfig, FAMILIES
from src.cross_validate import pgb_cross_validate


def run_family(name: str, args):
    spec = FAMILIES[name]
    config = PGBConfig()
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
    print(f"Training PGB model CV: {name}")
    print(f"{'='*60}")
    summary = pgb_cross_validate(config)

    print(f"\nPGB {name} CV summary:")
    for k in ["pr_auc", "f1", "mcc"]:
        print(f"  val  {k}: {summary[f'val_{k}_mean']:.4f} ± {summary[f'val_{k}_std']:.4f}")
        print(f"  test {k}: {summary[f'test_{k}_mean']:.4f} ± {summary[f'test_{k}_std']:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="PGB 5-fold cross-validation.")
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
