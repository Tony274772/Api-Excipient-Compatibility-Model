"""Train one configuration of the grid on the 80/20 split, or list / drive the whole grid.

Every run uses data/grid_8020/ (train 2770 / val 774, tested on the 22 external held-out pairs),
the same recipe the configuration originally had, and seed 42 unless told otherwise. Results go to
experiments/results/full_grid/<run name>/ with val_metrics.json, test_metrics.json,
training_history.json and run_info.json, so a finished run is skipped on a rerun.

    python experiments/full_grid/prepare_data.py              # once, builds data/grid_8020/
    python experiments/full_grid/run_grid.py --list
    python experiments/full_grid/run_grid.py --run fixed_vector_morgan_concat_asl
    python experiments/full_grid/run_grid.py --all --device cuda
    python experiments/full_grid/run_grid.py --all --time-budget 300   # stop after 300 minutes

--time-budget makes this usable on Colab: it stops cleanly before the session is cut, and the next
invocation picks up the runs that are still missing.
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)          # the library resolves data paths relative to the project root

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rdkit import RDLogger  # noqa: E402

from experiments.full_grid.grid import GRID, ORDER, spec  # noqa: E402
from src.config import Config  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATA = os.path.join(PROJECT_ROOT, "data", "grid_8020")
RESULTS = os.path.join(PROJECT_ROOT, "experiments", "results", "full_grid")
CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "full_grid")


def make_config(g, seed, device, epochs, patience=None, min_delta=None, results_root=RESULTS, ckpt_root=CKPT):
    """The configuration this run name stands for, pointed at the 80/20 grid data."""
    c = Config()
    c.encoder, c.fusion, c.pooling, c.loss = g["encoder"], g["fusion"], g["pooling"], g["loss"]
    c.use_pgb_head = g["pgb"]
    if g["source"]:
        c.fixed_vector_source = g["source"]
    c.seed, c.device = seed, device
    c.data_dir = DATA
    c.train_csv = os.path.join(DATA, "train.csv")
    c.val_csv = os.path.join(DATA, "val.csv")
    c.test_csv = os.path.join(DATA, "test.csv")
    c.api_descriptors_path = os.path.join(DATA, "api_descriptors.csv")
    c.excipient_descriptors_path = os.path.join(DATA, "excipient_descriptors.csv")
    c.descriptor_norm_stats_path = os.path.join(DATA, "descriptor_norm_stats.json")
    c.resolve_paths()
    # resolve_paths() rewrites the csv paths from data_dir; put ours back
    c.train_csv = os.path.join(DATA, "train.csv")
    c.val_csv = os.path.join(DATA, "val.csv")
    c.test_csv = os.path.join(DATA, "test.csv")
    if g["source"] in ("mol2vec", "pubchemfp"):
        c.fixed_vector_path = os.path.join(PROJECT_ROOT, "data",
                                           "mol2vec_embeddings.csv" if g["source"] == "mol2vec" else "pubchem_fps.csv")
    c.checkpoint_dir = os.path.join(ckpt_root, g["name"])
    c.metrics_dir = os.path.join(results_root, g["name"])
    if epochs:
        c.max_epochs = epochs
        c.pgb_max_epochs = epochs
    if patience:
        c.early_stop_patience = patience        # baseline trainer
        c.pgb_early_stop_patience = patience    # PGB trainer
    if min_delta is not None:
        c.early_stop_min_delta = min_delta
    c.positive_prior = float(pd.read_csv(c.train_csv)["Outcome1"].mean())
    return c


def is_done(name):
    return os.path.exists(os.path.join(RESULTS, name, "val_metrics.json"))


def run_one(name, seed=42, device="auto", epochs=None, patience=None, min_delta=None,
            results_root=RESULTS, ckpt_root=CKPT):
    """Train and evaluate one configuration. Returns the run_info dict.

    results_root / ckpt_root default to the grid folders; the multi-seed runs point them elsewhere
    so a second seed never overwrites the first.
    """
    g = spec(name)
    out = os.path.join(results_root, name)
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(ckpt_root, name), exist_ok=True)

    t0 = time.time()
    seed_everything(seed)
    config = make_config(g, seed, device, epochs, patience, min_delta, results_root, ckpt_root)
    dev = get_device(config.device)

    from main import build_encoder
    encoder = build_encoder(config, dev)

    if g["pgb"]:
        # pgb_single_run writes to <pgb_metrics_root>/<family>/ and <pgb_checkpoint_root>/<family>/,
        # so point both at this run's folders rather than its own defaults
        from src.single_run import pgb_single_run
        config.heldout_csv = config.test_csv
        config.family = name
        config.pgb_metrics_root = results_root
        config.pgb_checkpoint_root = ckpt_root
        pgb_single_run(config)
        val_m = json.load(open(os.path.join(out, "val_metrics.json")))
        test_m = json.load(open(os.path.join(out, "test_metrics.json")))
    else:
        from src.dataset import build_dataloaders
        from src.evaluate import evaluate_model, save_metrics
        from src.model import CompatibilityModel
        from src.train import train as train_baseline

        train_loader, val_loader, test_loader = build_dataloaders(config)
        model = CompatibilityModel(config, encoder).to(dev)
        model, history = train_baseline(model, train_loader, val_loader, config, dev)
        val_m, test_m, thr = evaluate_model(model, val_loader, test_loader, dev, config)
        save_metrics(val_m, os.path.join(out, "val_metrics.json"))
        save_metrics(test_m, os.path.join(out, "test_metrics.json"))
        save_metrics({"history": history}, os.path.join(out, "training_history.json"))

    info = dict(run=name, **{k: g[k] for k in ("encoder", "source", "fusion", "pooling", "loss", "pgb", "legacy")},
                seed=seed, device=str(dev),
                max_epochs=config.pgb_max_epochs if g["pgb"] else config.max_epochs,
                early_stop_patience=config.pgb_early_stop_patience if g["pgb"] else config.early_stop_patience,
                early_stop_min_delta=config.early_stop_min_delta,
                parameters=sum(p.numel() for p in model.parameters()) if not g["pgb"] else None,
                data={k: os.path.relpath(v, PROJECT_ROOT) for k, v in
                      [("train", config.train_csv), ("val", config.val_csv), ("heldout", config.test_csv)]},
                rows={"train": 2770, "val": 774, "heldout": 22},
                minutes=round((time.time() - t0) / 60, 2), finished=datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(out, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[{name}] val PR-AUC {val_m['pr_auc']:.4f} | held-out PR-AUC {test_m['pr_auc']:.4f} "
          f"| {info['minutes']} min")
    return info


def main():
    p = argparse.ArgumentParser(description="Run the model grid on the 80/20 split.")
    p.add_argument("--run", help="one run name")
    p.add_argument("--all", action="store_true", help="every configuration not yet finished")
    p.add_argument("--list", action="store_true")
    p.add_argument("--only", nargs="+", help="limit --all to these encoders, e.g. --only fixed_vector chemberta")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None, help="cap max epochs (default 150 baseline / 40 PGB)")
    p.add_argument("--patience", type=int, default=None,
                   help="early-stopping patience in epochs (default 6 baseline / 7 PGB); lower stops sooner")
    p.add_argument("--min-delta", type=float, default=None,
                   help="val PR-AUC gain that counts as improvement (default 0.002); higher stops sooner")
    p.add_argument("--time-budget", type=float, default=None, help="stop starting new runs after N minutes")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.list:
        for i, n in enumerate(ORDER, 1):
            print(f"{i:3d}. {'done ' if is_done(n) else '     '}{n}")
        print(f"\n{sum(is_done(n) for n in ORDER)} of {len(ORDER)} finished")
        return

    if not os.path.exists(os.path.join(DATA, "train.csv")):
        raise SystemExit("data/grid_8020/ is missing. Run: python experiments/full_grid/prepare_data.py")

    device = args.device or "auto"
    if args.run:
        run_one(args.run, args.seed, device, args.epochs, args.patience, args.min_delta)
        return
    if not args.all:
        raise SystemExit("pass --run <name>, --all or --list")

    todo = [n for n in ORDER if args.overwrite or not is_done(n)]
    if args.only:
        todo = [n for n in todo if GRID[n]["encoder"] in args.only]
    print(f"{len(todo)} runs to do (of {len(ORDER)}) | device {get_device(device)}")
    t0, done, failed = time.time(), [], []
    for i, name in enumerate(todo, 1):
        spent = (time.time() - t0) / 60
        if args.time_budget and spent > args.time_budget:
            print(f"\ntime budget of {args.time_budget} min reached; {len(todo) - i + 1} runs left for the next call")
            break
        print(f"\n[{i}/{len(todo)}] {name}  ({spent:.0f} min spent)", flush=True)
        try:
            run_one(name, args.seed, device, args.epochs, args.patience, args.min_delta)
            done.append(name)
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"[{name}] FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nfinished {len(done)}, failed {len(failed)}, "
          f"{sum(not is_done(n) for n in ORDER)} still missing, {(time.time() - t0) / 60:.0f} min")
    for n, e in failed:
        print(f"  failed: {n} — {e}")
    print("\nCollect with: python experiments/full_grid/collect.py")


if __name__ == "__main__":
    main()
