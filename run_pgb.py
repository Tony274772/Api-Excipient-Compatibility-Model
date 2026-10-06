"""
run_pgb.py
Entry point for training the 9 Prior-Gated Bilinear (PGB) models.
Run one family at a time:
    python run_pgb.py --family maccs
    python run_pgb.py --family molformer
    python run_pgb.py --family all
"""

import argparse
from src.config import Config
from src.cross_validate import pgb_cross_validate

FAMILIES = {
    "maccs":      {"encoder": "fixed_vector", "fixed_vector_source": "maccs",     "fusion": "concat", "pgb_max_epochs": 40},
    "morgan":     {"encoder": "fixed_vector", "fixed_vector_source": "morgan",    "fusion": "concat", "pgb_max_epochs": 40},
    "pubchemfp":  {"encoder": "fixed_vector", "fixed_vector_source": "pubchemfp", "fusion": "concat", "pgb_max_epochs": 40},
    "mol2vec":    {"encoder": "fixed_vector", "fixed_vector_source": "mol2vec",   "fusion": "concat", "pgb_max_epochs": 40},
    "molformer":  {"encoder": "molformer",  "fusion": "cross_attn", "pooling": "global_gated_attention", "pgb_max_epochs": 60},
    "chemberta":  {"encoder": "chemberta",  "fusion": "cross_attn", "pooling": "cls",                    "pgb_max_epochs": 60},
    "gin":        {"encoder": "pretrained_gin", "fusion": "cross_attn", "pooling": "global_gated_attention", "pgb_max_epochs": 60},
    "gat":        {"encoder": "pretrained_gat", "fusion": "cross_attn", "pooling": "global_gated_attention", "pgb_max_epochs": 60},
    "dmpnn":      {"encoder": "dmpnn_chemprop", "fusion": "cross_attn", "pooling": "cls",                "pgb_max_epochs": 60},
}

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
    print(f"Training PGB model: {name}")
    print(f"{'='*60}")
    summary = pgb_cross_validate(config)

    print(f"\nPGB {name} CV summary:")
    for k in ["pr_auc", "f1", "mcc"]:
        print(f"  val  {k}: {summary[f'val_{k}_mean']:.4f} ± {summary[f'val_{k}_std']:.4f}")
        print(f"  test {k}: {summary[f'test_{k}_mean']:.4f} ± {summary[f'test_{k}_std']:.4f}")


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
