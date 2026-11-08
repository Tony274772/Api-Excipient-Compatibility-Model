"""Retrain the top 5 grid models (by validation PR-AUC on the 80/20 split) with 5 seeds each.

The five configurations are the top of experiments/results/full_grid/summary.csv, ranked by
validation PR-AUC. Every run uses exactly the grid recipe (data/grid_8020/, same early stopping,
same threshold tuning on validation) and only the seed changes. Seed 42 is retrained too, so all
five seeds come from the same code in the same place.

Results go to experiments/results/top5_seeds/seed_<s>/<run>/ (val_metrics.json, test_metrics.json
= the 22 held-out pairs, training_history.json, run_info.json). A run whose run_info.json exists
is skipped, so the script can be re-launched after an interruption.

    python experiments/top5_seeds/run_seeds.py --list
    python experiments/top5_seeds/run_seeds.py --device cuda
    python experiments/top5_seeds/run_seeds.py --device cuda --models chemberta_cross_attn_cls_bce
    python experiments/top5_seeds/run_seeds.py --device cuda --seeds 43 44
    python experiments/top5_seeds/collect.py
"""

import argparse
import os
import sys
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch  # noqa: E402

from experiments.full_grid.grid import GRID  # noqa: E402
from experiments.full_grid.run_grid import DATA, run_one  # noqa: E402

# top 5 of experiments/results/full_grid/summary.csv by val PR-AUC (seed 42)
TOP5 = [
    "chemberta_cross_attn_cls_bce",                 # 0.7627
    "pgb_dmpnn_chemprop_cross_attn_cls_asl",        # 0.7545
    "chemberta_cross_attn_global_gated_bce",        # 0.7398
    "pgb_molformer_cross_attn_global_gated_asl",    # 0.7324
    "dmpnn_chemprop_cross_attn_cls_asl",            # 0.7317
]
SEEDS = [42, 43, 44, 45, 46]

OUT = os.path.join(PROJECT_ROOT, "experiments", "results", "top5_seeds")
CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "top5_seeds")


def seed_dirs(out, ckpt, seed):
    return os.path.join(out, f"seed_{seed}"), os.path.join(ckpt, f"seed_{seed}")


def is_done(out, name, seed):
    # run_info.json is written last, after both metric files
    return os.path.exists(os.path.join(out, f"seed_{seed}", name, "run_info.json"))


def main():
    p = argparse.ArgumentParser(description="Top 5 grid models x 5 seeds on the 80/20 split.")
    p.add_argument("--models", nargs="+", default=TOP5, help="run names (default: the top 5)")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=None, help="cap max epochs (smoke tests only)")
    p.add_argument("--patience", type=int, default=None, help="default: the grid recipe (6 baseline / 7 PGB)")
    p.add_argument("--min-delta", type=float, default=None, help="default: the grid recipe (0.002)")
    p.add_argument("--out", default=OUT, help="results folder")
    p.add_argument("--ckpt", default=CKPT, help="checkpoint folder")
    p.add_argument("--overwrite", action="store_true", help="retrain runs that already finished")
    p.add_argument("--list", action="store_true", help="show what is done and stop")
    args = p.parse_args()

    unknown = [m for m in args.models if m not in GRID]
    if unknown:
        raise SystemExit(f"unknown run(s): {unknown}. See python experiments/full_grid/grid.py")

    planned = [(m, s) for m in args.models for s in args.seeds]
    if args.list:
        for m, s in planned:
            print(f"{'done ' if is_done(args.out, m, s) else '     '}seed {s}  {m}")
        print(f"\n{sum(is_done(args.out, m, s) for m, s in planned)} of {len(planned)} finished")
        return

    if not os.path.exists(os.path.join(DATA, "train.csv")):
        raise SystemExit("data/grid_8020/ is missing. Run: python experiments/full_grid/prepare_data.py")
    if args.epochs:
        print(f"WARNING: --epochs {args.epochs} caps training; use only for a smoke test")

    todo = [(m, s) for m, s in planned if args.overwrite or not is_done(args.out, m, s)]
    print(f"{len(args.models)} models x {len(args.seeds)} seeds = {len(planned)} runs, "
          f"{len(todo)} to do | out {os.path.relpath(args.out, PROJECT_ROOT)}", flush=True)

    t0, done, failed = time.time(), [], []
    for i, (name, seed) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] seed {seed}  {name}  ({(time.time() - t0) / 60:.0f} min spent)", flush=True)
        res, ck = seed_dirs(args.out, args.ckpt, seed)
        try:
            run_one(name, seed, args.device, args.epochs, args.patience, args.min_delta,
                    results_root=res, ckpt_root=ck)
            done.append((name, seed))
        except Exception as e:
            failed.append((name, seed, f"{type(e).__name__}: {e}"))
            print(f"[{name} seed {seed}] FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    missing = sum(not is_done(args.out, m, s) for m, s in planned)
    print(f"\nfinished {len(done)}, failed {len(failed)}, {missing} still missing, "
          f"{(time.time() - t0) / 60:.0f} min")
    for n, s, e in failed:
        print(f"  failed: seed {s} {n} — {e}")
    print("\nCollect with: python experiments/top5_seeds/collect.py")


if __name__ == "__main__":
    main()
