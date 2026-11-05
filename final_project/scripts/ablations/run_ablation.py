"""
Ablation runs of the Morgan-PGB 80/20 model.

Same data, recipe, seed and threshold rule as `python run.py train`; only the model is changed by the
ablation named with --ablation (see apiexc/ablations/variants.py). Checkpoints and results are kept
apart from the final model:

    checkpoints/ablations/<ablation>_seed<seed>/    best_model.pt, prior_table.json, pgb_norm_stats.npz, threshold.json
    results/ablations/runs/<ablation>_seed<seed>/   metrics JSON, predictions, training history, run info
    results/ablations/summary.csv                   one row per ablation + seed (updated by every evaluate)

Usage (from any directory):
    python scripts/ablations/run_ablation.py list
    python scripts/ablations/run_ablation.py train --ablation no_w
    python scripts/ablations/run_ablation.py train --ablation prior1+no_w     # ablations combine with "+"
    python scripts/ablations/run_ablation.py train --ablation full            # baseline through the same harness
    python scripts/ablations/run_ablation.py evaluate --ablation no_w
    python scripts/ablations/run_ablation.py train --ablation full --mech v2  # mechanism flags v2

--mech v2 swaps the 5 mechanism flags of apiexc/features/mechanisms.py for the 13 rules of
apiexc/ablations/mech_v2.py (gate and MLP inputs grow by 8). Runs with v2 are named <ablation>_mechv2_seed<seed>.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # project root

import pandas as pd  # noqa: E402

from apiexc import paths  # noqa: E402
from apiexc.ablations.mech_v2 import NUM_MECH_V2, MechV2PGBDataset  # noqa: E402
from apiexc.ablations.prior_lookup import (build_leave_cluster_out_prior_vectors, load_prior_table,  # noqa: E402
                                           new_prior_table)
from apiexc.ablations.variants import ABLATIONS, DESCRIPTIONS, DESIGNS, build_model, data_settings, parse_ablation  # noqa: E402
from apiexc.ablations.variants import describe as describe_ablation  # noqa: E402
from apiexc.dataset import PGBDataset, PGBNormStats  # noqa: E402
from apiexc.evaluation.metrics import (collect_predictions, metric_suite, prediction_table, print_metrics,  # noqa: E402
                                       print_predictions, save_metrics, tune_threshold)
from apiexc.features.excipient_prior import GLOBAL_RATE  # noqa: E402
from apiexc.pipeline import build_encoder, check_heldout_unseen, get_split, load_checkpoint, make_config, make_loader, md5  # noqa: E402
from apiexc.utils import get_device, seed_everything  # noqa: E402

SPLITS = ["val", "heldout"]
MECH_SETS = ("v1", "v2")


def run_name(ablation, seed, mech="v1"):
    return f"{ablation}_mechv2_seed{seed}" if mech == "v2" else f"{ablation}_seed{seed}"


def describe(ablation, mech="v1"):
    text = describe_ablation(ablation)
    return f"{text} | mechanism flags v2 ({NUM_MECH_V2} rules)" if mech == "v2" else text


def dataset_class(config):
    """PGBDataset, or the v2 mechanism-flag dataset when config.mech_set == 'v2'."""
    return MechV2PGBDataset if getattr(config, "mech_set", "v1") == "v2" else PGBDataset


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def load_ablation(ckpt_dir, device):
    """load_checkpoint with the ablation model, prior table and dataset of the run."""
    lm = load_checkpoint(ckpt_dir, device, build_model=build_model, load_prior_table=load_prior_table)
    lm.dataset_cls = dataset_class(lm.config)
    return lm


# --------------------------------------------------------------------------- train
def train(ablation, seed, device_str=None, epochs=None, overwrite=False, mech="v1"):
    """Same order of operations as run.py train; the model comes from build_model()."""
    from apiexc.training.trainer import pgb_train

    split = get_split()
    check_heldout_unseen(split)
    name = run_name(ablation, seed, mech)
    ckpt_dir, out_dir = paths.ABLATION_CHECKPOINTS / name, paths.ABLATION_RUNS / name
    if (ckpt_dir / "best_model.pt").exists() and not overwrite:
        raise SystemExit(f"{ckpt_dir} already exists; pass --overwrite to retrain")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    seed_everything(seed)
    config = make_config(split, seed, device_str or "auto", epochs)
    config.ablation = ablation
    for key, value in data_settings(ablation).items():   # data ablations, e.g. no_family -> prior_lookup
        setattr(config, key, value)
    # prior global rate: the train positive rate (from make_config), or the old 0.094 with old_global
    legacy = getattr(config, "prior_global_rate", "train") == "legacy"
    config.prior_global_rate_value = GLOBAL_RATE if legacy else config.positive_prior
    config.mech_set = mech
    if mech == "v2":
        config.pgb_num_mechanisms = NUM_MECH_V2   # sizes the gate and MLP inputs of the head
    device = get_device(config.device)
    config.checkpoint_dir, config.metrics_dir = str(ckpt_dir), str(out_dir)
    train_df, val_df = pd.read_csv(split.train_csv), pd.read_csv(split.val_csv)

    encoder = build_encoder(config, device)

    all_train_smiles = list(set(train_df["API_Smiles"].dropna().tolist() + train_df["Excipient_Smiles"].dropna().tolist()))
    pgb_norm = PGBNormStats.from_smiles_list(all_train_smiles)
    pgb_norm.save(str(ckpt_dir / "pgb_norm_stats.npz"))

    prior_table = new_prior_table(config)
    prior_table.fit(train_df, leave_out_cluster_id=None)
    prior_table.save(str(ckpt_dir / "prior_table.json"))
    train_priors = build_leave_cluster_out_prior_vectors(train_df, config, cluster_col="cluster_id")

    ds_cls = dataset_class(config)
    train_loader = make_loader(split.train_csv, config, prior_table, pgb_norm, encoder, row_priors=train_priors,
                               shuffle=True, dataset_cls=ds_cls)
    val_loader = make_loader(split.val_csv, config, prior_table, pgb_norm, encoder, dataset_cls=ds_cls)

    model = build_model(config, encoder)
    model.to(device)
    print(f"[ablation {ablation}: {describe(ablation, mech)}]")
    print(f"[seed {seed} | {device}] train {len(train_df)} | val {len(val_df)} | params {n_params(model):,} | "
          f"max epochs {config.pgb_max_epochs} | checkpoint -> {ckpt_dir}")
    model, history = pgb_train(model, train_loader, val_loader, config, device)

    val_probs, val_labels = collect_predictions(model, val_loader, device)
    threshold, _ = tune_threshold(val_probs, val_labels, config.threshold_step, tnr_floor=config.pgb_tnr_floor)
    (ckpt_dir / "threshold.json").write_text(json.dumps(
        {"threshold": float(threshold), "rule": f"best val MCC with TNR >= {config.pgb_tnr_floor}"}, indent=2))

    save_metrics({"history": history}, str(out_dir / "training_history.json"))
    (out_dir / "run_info.json").write_text(json.dumps({
        "model": "Morgan-PGB", "ablation": ablation, "mech_set": mech, "description": describe(ablation, mech),
        "parameters": n_params(model),
        "prior_table": {"lookup": getattr(config, "prior_lookup", "full"),
                        "global_rate": float(prior_table.global_rate)},
        "split": split.description,
        "seed": seed, "device": str(device),
        "data": {s: {"path": paths.rel(p), "rows": len(pd.read_csv(p)), "md5": md5(p)}
                 for s, p in [("train", split.train_csv), ("val", split.val_csv), ("heldout", split.heldout_csv)]},
        "epochs_trained": len(history["val_pr_auc"]),
        "best_epoch": int(pd.Series(history["val_pr_auc"]).idxmax()) + 1,
        "threshold_from_val": float(threshold), "tnr_floor": config.pgb_tnr_floor,
        "finished": datetime.now().isoformat(timespec="seconds"), "minutes": round((time.time() - t0) / 60, 2),
    }, indent=2))

    print()
    evaluate(ablation, seed, device_str, mech)


# --------------------------------------------------------------------------- evaluate
def evaluate(ablation, seed, device_str=None, mech="v1"):
    split = get_split()
    check_heldout_unseen(split)
    name = run_name(ablation, seed, mech)
    ckpt_dir, out_dir = paths.ABLATION_CHECKPOINTS / name, paths.ABLATION_RUNS / name
    lm = load_ablation(ckpt_dir, get_device(device_str or "auto"))

    data, probs, labels = {}, {}, {}
    for s in SPLITS:
        data[s] = split.frame(s)
        probs[s], labels[s] = lm.score(data[s])
    threshold = lm.threshold

    metrics = {s: metric_suite(probs[s], labels[s], threshold) for s in SPLITS}
    tables = {s: prediction_table(data[s], probs[s], labels[s], threshold) for s in SPLITS}

    os.makedirs(out_dir, exist_ok=True)
    for s in SPLITS:
        save_metrics(metrics[s], str(out_dir / f"{s}_metrics.json"))
        tables[s].to_csv(out_dir / f"predictions_{s}.csv", index=False)
    update_summary(ablation, seed, mech, lm.model, lm.info, threshold, metrics)

    print("=" * 100)
    print(f"Ablation {ablation}: {describe(ablation, mech)}")
    print(f"checkpoint {paths.rel(ckpt_dir)} | device {lm.device} | params {n_params(lm.model):,} | best epoch "
          f"{lm.info.get('epoch')} | val PR-AUC at best epoch {lm.info.get('val_pr_auc', float('nan')):.4f} | "
          f"threshold {threshold:.3f}")
    print("=" * 100)
    print_metrics(metrics)
    print(f"\nHeld-out predictions ({metrics['heldout']['correct']}/{metrics['heldout']['n']} correct):")
    print_predictions(tables["heldout"])
    print(f"\nSaved metrics and predictions to {out_dir}")
    print(f"Summary of all ablations: {paths.ABLATION_SUMMARY}")
    return metrics


def update_summary(ablation, seed, mech, model, info, threshold, metrics):
    row = {"ablation": ablation, "mech": mech, "seed": seed, "description": describe(ablation, mech),
           "parameters": n_params(model),
           "best_epoch": info.get("epoch"), "threshold": round(float(threshold), 3)}
    for s in SPLITS:
        m = metrics[s]
        row[f"{s}_correct"] = f"{m['correct']}/{m['n']}"
        for k in ["pr_auc", "roc_auc", "mcc", "f1", "precision", "recall", "tp", "tn", "fp", "fn"]:
            row[f"{s}_{k}"] = round(float(m[k]), 4) if isinstance(m[k], float) else m[k]
    df = pd.read_csv(paths.ABLATION_SUMMARY) if paths.ABLATION_SUMMARY.exists() else pd.DataFrame()
    if len(df):
        df["mech"] = df["mech"].fillna("v1") if "mech" in df else "v1"
        df = df[~((df["ablation"] == ablation) & (df["seed"] == seed) & (df["mech"] == mech))]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(paths.ABLATION_RESULTS, exist_ok=True)
    df.to_csv(paths.ABLATION_SUMMARY, index=False)


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
    t.add_argument("--mech", choices=MECH_SETS, default="v1", help="mechanism flags: v1 (apiexc/features/mechanisms.py) or v2")

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
        for k, parts in DESIGNS.items():
            print(f"{k:15s} {'+'.join(parts)}")
        print('combine with "+", applied left to right, e.g. prior1+no_w')
    elif args.cmd == "train":
        train(args.ablation, args.seed, args.device, args.epochs, args.overwrite, args.mech)
    elif args.cmd == "evaluate":
        evaluate(args.ablation, args.seed, args.device, args.mech)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
