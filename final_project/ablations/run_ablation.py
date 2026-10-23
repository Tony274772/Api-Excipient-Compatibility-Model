"""
Ablation runs of the Morgan-PGB 80/20 model.

Same data, recipe, seed and threshold rule as `python run.py train`; only the model is changed by the
ablation named with --ablation (see variants.py). Checkpoints and results are kept apart from the
final model:

    ablations/checkpoints/<ablation>_seed<seed>/   best_model.pt, prior_table.json, pgb_norm_stats.npz, threshold.json
    ablations/results/<ablation>_seed<seed>/       metrics JSON, predictions, training history, run info
    ablations/results/summary.csv                  one row per ablation + seed (updated by every evaluate)

Usage (from any directory):
    python ablations/run_ablation.py list
    python ablations/run_ablation.py train --ablation no_w
    python ablations/run_ablation.py train --ablation prior1
    python ablations/run_ablation.py train --ablation prior1+no_w   # ablations combine with "+"
    python ablations/run_ablation.py train --ablation full          # baseline through the same harness
    python ablations/run_ablation.py evaluate --ablation no_w
    python ablations/run_ablation.py train --ablation full --mech v2   # mechanism flags v2 (mech_v2.py)

--mech v2 swaps the 5 mechanism flags of src/lookup.py for the 13 rules of ablations/mech_v2.py (gate and
MLP inputs grow by 8). Runs with v2 are named <ablation>_mechv2_seed<seed>.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime

FINAL_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FINAL_PROJECT)

import run  # noqa: E402  (also chdirs into final_project/, so the paths below are relative to it)
from run import (HELDOUT_CSV, SPLITS, TRAIN_CSV, VAL_CSV, build_encoder, frame_for,  # noqa: E402
                 heldout_pairs_in_train, make_config, md5, metric_suite, prediction_table,
                 print_heldout, print_metrics)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ablations.mech_v2 import NUM_MECH_V2, MechV2PGBDataset  # noqa: E402
from ablations.variants import ABLATIONS, DESCRIPTIONS, build_model, parse_ablation  # noqa: E402
from ablations.variants import describe as describe_ablation  # noqa: E402
from src.dataset import PGBDataset, PGBNormStats, pgb_collate_fn  # noqa: E402
from src.evaluate import collect_predictions, save_metrics, tune_threshold  # noqa: E402
from src.pgb_prior import ExcipientPriorTable, build_leave_cluster_out_prior_vectors  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

CKPT_ROOT = "ablations/checkpoints"
RESULTS_ROOT = "ablations/results"
SUMMARY_CSV = os.path.join(RESULTS_ROOT, "summary.csv")


MECH_SETS = ("v1", "v2")


def run_name(ablation, seed, mech="v1"):
    return f"{ablation}_mechv2_seed{seed}" if mech == "v2" else f"{ablation}_seed{seed}"


def describe(ablation, mech="v1"):
    text = describe_ablation(ablation)
    return f"{text} | mechanism flags v2 ({NUM_MECH_V2} rules)" if mech == "v2" else text


def make_loader(csv, config, prior_table, pgb_norm, encoder, row_priors=None, shuffle=False):
    """run.make_loader, with the v2 mechanism flags when config.mech_set == 'v2'."""
    dataset_cls = MechV2PGBDataset if getattr(config, "mech_set", "v1") == "v2" else PGBDataset
    ds = dataset_cls(csv, config.encoder, config.fixed_vector_source, prior_table, pgb_norm,
                     fixed_vector_encoder=encoder, row_priors=row_priors)
    return DataLoader(ds, batch_size=config.pgb_batch_size, shuffle=shuffle, collate_fn=pgb_collate_fn, num_workers=0)


def score_frame(model, df, config, prior_table, pgb_norm, encoder, device):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pairs.csv")
        df.to_csv(path, index=False)
        return collect_predictions(model, make_loader(path, config, prior_table, pgb_norm, encoder), device)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(ckpt_dir, device):
    ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device, weights_only=False)
    config = ckpt["config"]
    encoder = build_encoder(config, device)
    model = build_model(config, encoder)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    prior_table = ExcipientPriorTable.load(os.path.join(ckpt_dir, "prior_table.json"))
    pgb_norm = PGBNormStats.load(os.path.join(ckpt_dir, "pgb_norm_stats.npz"))
    return model, config, encoder, prior_table, pgb_norm, ckpt


# --------------------------------------------------------------------------- train
def train(ablation, seed, device_str=None, epochs=None, overwrite=False, mech="v1"):
    """Same order of operations as run.train(); the model comes from build_model()."""
    from src.train import pgb_train

    name = run_name(ablation, seed, mech)
    ckpt_dir, out_dir = os.path.join(CKPT_ROOT, name), os.path.join(RESULTS_ROOT, name)
    if os.path.exists(os.path.join(ckpt_dir, "best_model.pt")) and not overwrite:
        raise SystemExit(f"{ckpt_dir} already exists; pass --overwrite to retrain")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    seed_everything(seed)
    config = make_config(seed, device_str or "auto", epochs)
    config.ablation = ablation
    config.mech_set = mech
    if mech == "v2":
        config.pgb_num_mechanisms = NUM_MECH_V2   # sizes the gate and MLP inputs of the head
    device = get_device(config.device)
    config.checkpoint_dir, config.metrics_dir = ckpt_dir, out_dir
    train_df, val_df = pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)

    encoder = build_encoder(config, device)

    all_train_smiles = list(set(train_df["API_Smiles"].dropna().tolist() + train_df["Excipient_Smiles"].dropna().tolist()))
    pgb_norm = PGBNormStats.from_smiles_list(all_train_smiles)
    pgb_norm.save(os.path.join(ckpt_dir, "pgb_norm_stats.npz"))

    prior_table = ExcipientPriorTable()
    prior_table.fit(train_df, leave_out_cluster_id=None)
    prior_table.save(os.path.join(ckpt_dir, "prior_table.json"))
    train_priors = build_leave_cluster_out_prior_vectors(train_df, cluster_col="cluster_id")

    train_loader = make_loader(TRAIN_CSV, config, prior_table, pgb_norm, encoder, row_priors=train_priors, shuffle=True)
    val_loader = make_loader(VAL_CSV, config, prior_table, pgb_norm, encoder)

    model = build_model(config, encoder)
    model.to(device)
    print(f"[ablation {ablation}: {describe(ablation, mech)}]")
    print(f"[seed {seed} | {device}] train {len(train_df)} | val {len(val_df)} | params {n_params(model):,} | "
          f"max epochs {config.pgb_max_epochs} | checkpoint -> {os.path.join(FINAL_PROJECT, ckpt_dir)}")
    model, history = pgb_train(model, train_loader, val_loader, config, device)

    val_probs, val_labels = collect_predictions(model, val_loader, device)
    threshold, _ = tune_threshold(val_probs, val_labels, config.threshold_step, tnr_floor=config.pgb_tnr_floor)
    with open(os.path.join(ckpt_dir, "threshold.json"), "w") as f:
        json.dump({"threshold": float(threshold), "rule": f"best val MCC with TNR >= {config.pgb_tnr_floor}"}, f, indent=2)

    save_metrics({"history": history}, os.path.join(out_dir, "training_history.json"))
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump({
            "model": "Morgan-PGB", "ablation": ablation, "mech_set": mech, "description": describe(ablation, mech),
            "parameters": n_params(model),
            "split": "80/20 (Butina API clusters, GroupShuffleSplit random_state=42)",
            "seed": seed, "device": str(device),
            "data": {s: {"path": p, "rows": len(pd.read_csv(p)), "md5": md5(p)}
                     for s, p in [("train", TRAIN_CSV), ("val", VAL_CSV), ("heldout", HELDOUT_CSV)]},
            "epochs_trained": len(history["val_pr_auc"]),
            "best_epoch": int(pd.Series(history["val_pr_auc"]).idxmax()) + 1,
            "threshold_from_val": float(threshold), "tnr_floor": config.pgb_tnr_floor,
            "finished": datetime.now().isoformat(timespec="seconds"), "minutes": round((time.time() - t0) / 60, 2),
        }, f, indent=2)

    print()
    evaluate(ablation, seed, device_str, mech)


# --------------------------------------------------------------------------- evaluate
def evaluate(ablation, seed, device_str=None, mech="v1"):
    name = run_name(ablation, seed, mech)
    ckpt_dir, out_dir = os.path.join(CKPT_ROOT, name), os.path.join(RESULTS_ROOT, name)
    device = get_device(device_str or "auto")
    model, config, encoder, prior_table, pgb_norm, ckpt = load_checkpoint(ckpt_dir, device)

    data, probs, labels = {}, {}, {}
    for s in SPLITS:
        data[s] = frame_for(s)
        probs[s], labels[s] = score_frame(model, data[s], config, prior_table, pgb_norm, encoder, device)
    threshold = json.load(open(os.path.join(ckpt_dir, "threshold.json")))["threshold"]

    metrics = {s: metric_suite(probs[s], labels[s], threshold) for s in SPLITS}
    unseen = ~heldout_pairs_in_train(data["heldout"])
    metrics["heldout_unseen"] = metric_suite(probs["heldout"][unseen], labels["heldout"][unseen], threshold)
    tables = {s: prediction_table(data[s], probs[s], labels[s], threshold) for s in SPLITS}

    os.makedirs(out_dir, exist_ok=True)
    for s in metrics:
        save_metrics(metrics[s], os.path.join(out_dir, f"{s}_metrics.json"))
    for s in SPLITS:
        tables[s].to_csv(os.path.join(out_dir, f"predictions_{s}.csv"), index=False)
    update_summary(ablation, seed, mech, model, ckpt, threshold, metrics)

    print("=" * 100)
    print(f"Ablation {ablation}: {describe(ablation, mech)}")
    print(f"checkpoint {ckpt_dir} | device {device} | params {n_params(model):,} | best epoch {ckpt.get('epoch')} | "
          f"val PR-AUC at best epoch {ckpt.get('val_pr_auc', float('nan')):.4f} | threshold {threshold:.3f}")
    print("=" * 100)
    print_metrics(metrics)
    print(f"(heldout_unseen = the {int(unseen.sum())} held-out pairs that are not also in train.csv)")
    print(f"\nHeld-out predictions ({metrics['heldout']['correct']}/{metrics['heldout']['n']} correct):")
    print_heldout(tables["heldout"])
    print(f"\nSaved metrics and predictions to {os.path.join(FINAL_PROJECT, out_dir)}")
    print(f"Summary of all ablations: {os.path.join(FINAL_PROJECT, SUMMARY_CSV)}")
    return metrics


def update_summary(ablation, seed, mech, model, ckpt, threshold, metrics):
    row = {"ablation": ablation, "mech": mech, "seed": seed, "description": describe(ablation, mech),
           "parameters": n_params(model),
           "best_epoch": ckpt.get("epoch"), "threshold": round(float(threshold), 3)}
    for s in ["val", "heldout", "heldout_unseen"]:
        m = metrics[s]
        row[f"{s}_correct"] = f"{m['correct']}/{m['n']}"
        for k in ["pr_auc", "roc_auc", "mcc", "f1", "precision", "recall", "tp", "tn", "fp", "fn"]:
            row[f"{s}_{k}"] = round(float(m[k]), 4) if isinstance(m[k], float) else m[k]
    df = pd.read_csv(SUMMARY_CSV) if os.path.exists(SUMMARY_CSV) else pd.DataFrame()
    if len(df):
        df["mech"] = df["mech"].fillna("v1") if "mech" in df else "v1"
        df = df[~((df["ablation"] == ablation) & (df["seed"] == seed) & (df["mech"] == mech))]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(SUMMARY_CSV, index=False)


def main():
    p = argparse.ArgumentParser(description="Ablations of the Morgan-PGB 80/20 model.")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("list", help="list the available ablations")

    t = sub.add_parser("train", help="train an ablation, then evaluate it")
    t.add_argument("--ablation", required=True, help="name from `list`, or several joined with +")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default=None, help="cpu, cuda (default: cuda if available)")
    t.add_argument("--epochs", type=int, default=None, help="max epochs (default 40, early stopping on val PR-AUC)")
    t.add_argument("--overwrite", action="store_true")
    t.add_argument("--mech", choices=MECH_SETS, default="v1", help="mechanism flags: v1 (src/lookup.py) or v2")

    e = sub.add_parser("evaluate", help="metrics on val / held-out for a trained ablation")
    e.add_argument("--ablation", required=True, help="name from `list`, or several joined with +")
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--device", default=None)
    e.add_argument("--mech", choices=MECH_SETS, default="v1")

    args = p.parse_args()
    if getattr(args, "ablation", None):
        try:
            parse_ablation(args.ablation)
        except ValueError as err:
            p.error(str(err))
    if args.cmd == "list":
        for k in ABLATIONS:
            print(f"{k:15s} {DESCRIPTIONS[k]}")
        print('combine with "+", applied left to right, e.g. prior1+no_w')
    elif args.cmd == "train":
        train(args.ablation, args.seed, args.device, args.epochs, args.overwrite, args.mech)
    elif args.cmd == "evaluate":
        evaluate(args.ablation, args.seed, args.device, args.mech)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
