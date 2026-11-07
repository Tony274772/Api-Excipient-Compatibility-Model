"""Recompute metrics for finished runs from their saved checkpoints. No training.

The baseline evaluation path did not record ROC-AUC and saved no predictions, so those columns are
missing for the 49 baseline runs. Every run did save best_model.pt with its weights and its config,
so the models can simply be rebuilt and re-scored: inference over 774 validation and 22 held-out
pairs, a few seconds each.

Each run's val_metrics.json and test_metrics.json are rewritten with the full metric set (ROC-AUC
included), and predictions_val.csv / predictions_heldout.csv are written alongside so this never
has to be redone. The decision threshold is taken from the existing val_metrics.json, so every
other number reproduces exactly what the run already reported — if one does not, the script says so
rather than quietly overwriting it.

Run it where the checkpoints are (the machine that trained them):

    python experiments/full_grid/rescore.py --dry-run
    python experiments/full_grid/rescore.py --device cuda
    python experiments/full_grid/rescore.py --device cuda --only fixed_vector
"""

import argparse
import json
import os
import shutil
import sys

import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from experiments.full_grid.grid import GRID, ORDER  # noqa: E402
from experiments.full_grid.run_grid import CKPT, RESULTS, make_config  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

CHECKED = ["pr_auc", "mcc", "f1", "precision", "recall", "accuracy"]


def predictions_table(csv_path, probs, labels, threshold):
    df = pd.read_csv(csv_path)
    cols = [c for c in ["Api_name", "Excipient_name", "API_CID", "Excipient_CID"] if c in df]
    out = df[cols].copy() if cols else pd.DataFrame(index=range(len(probs)))
    out["label"] = labels.astype(int)
    out["probability"] = probs
    out["prediction"] = (probs >= threshold).astype(int)
    out["correct"] = (out["prediction"] == out["label"]).astype(int)
    return out


def rescore_one(name, device_str, check_tol=1e-6):
    """Rebuild this run's model from its checkpoint and re-score val + held-out."""
    from src.dataset import build_dataloaders
    from src.evaluate import collect_predictions, compute_metrics
    from src.model import CompatibilityModel
    from main import build_encoder

    g = GRID[name]
    out = os.path.join(RESULTS, name)
    ckpt_path = os.path.join(CKPT, name, "best_model.pt")
    old_val = json.load(open(os.path.join(out, "val_metrics.json")))
    threshold = float(old_val["threshold"])

    seed_everything(42)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    # the checkpoint's config came from the machine that trained it; re-point the data at this one
    fresh = make_config(g, 42, device_str, None)
    for f in ("train_csv", "val_csv", "test_csv", "data_dir", "api_descriptors_path",
              "excipient_descriptors_path", "descriptor_norm_stats_path", "fixed_vector_path"):
        setattr(config, f, getattr(fresh, f))
    config.device = device_str

    dev = get_device(device_str)
    encoder = build_encoder(config, dev)
    model = CompatibilityModel(config, encoder).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, val_loader, test_loader = build_dataloaders(config)
    scored = {}
    for split, loader, csv in (("val", val_loader, config.val_csv), ("heldout", test_loader, config.test_csv)):
        probs, labels = collect_predictions(model, loader, dev)
        m = compute_metrics(probs, labels, threshold)
        scored[split] = (m, predictions_table(csv, probs, labels, threshold))

    # the rebuilt model must reproduce what the run already reported
    drift = [k for k in CHECKED
             if k in old_val and abs(float(old_val[k]) - float(scored["val"][0][k])) > check_tol]
    if drift:
        detail = ", ".join(f"{k} {old_val[k]:.4f} -> {scored['val'][0][k]:.4f}" for k in drift)
        raise RuntimeError(f"rebuilt model does not reproduce the saved metrics ({detail})")

    for split, fname in (("val", "val_metrics.json"), ("heldout", "test_metrics.json")):
        m, table = scored[split]
        path = os.path.join(out, fname)
        shutil.copy2(path, path + ".bak")
        with open(path, "w") as f:
            json.dump(m, f, indent=2)
        table.to_csv(os.path.join(out, f"predictions_{split}.csv"), index=False)
    return scored["val"][0]["roc_auc"], scored["heldout"][0]["roc_auc"]


def main():
    p = argparse.ArgumentParser(description="Recompute metrics from saved checkpoints (no training).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", nargs="+", help="limit to these encoders")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    todo = []
    for name in ORDER:
        if GRID[name]["pgb"]:
            continue                               # the PGB path already records ROC-AUC
        if args.only and GRID[name]["encoder"] not in args.only:
            continue
        has_val = os.path.exists(os.path.join(RESULTS, name, "val_metrics.json"))
        has_ckpt = os.path.exists(os.path.join(CKPT, name, "best_model.pt"))
        if not has_val:
            continue
        if not has_ckpt:
            print(f"  no checkpoint, skipped: {name}")
            continue
        if "roc_auc" in json.load(open(os.path.join(RESULTS, name, "val_metrics.json"))):
            print(f"  already has roc_auc:    {name}")
            continue
        todo.append(name)

    print(f"\n{len(todo)} run(s) to re-score (inference only)")
    if args.dry_run:
        for n in todo:
            print("   ", n)
        return
    if not todo:
        print("nothing to do.")
        return

    failed = []
    for i, name in enumerate(todo, 1):
        try:
            v, h = rescore_one(name, args.device)
            print(f"[{i}/{len(todo)}] {name:48s} val ROC-AUC {v:.4f} | held-out {h:.4f}", flush=True)
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"[{i}/{len(todo)}] {name:48s} FAILED {type(e).__name__}: {e}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nre-scored {len(todo) - len(failed)} of {len(todo)}")
    for n, e in failed:
        print(f"  failed: {n} — {e}")
    print("\nThe previous metrics files are kept as *.json.bak.")
    print("Collect with: python experiments/full_grid/collect.py")


if __name__ == "__main__":
    main()
