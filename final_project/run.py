"""
Morgan-PGB: API-excipient compatibility model (Prior-Gated Bilinear head on Morgan + MACCS fingerprints),
trained on the 80/20 API group split. The shipped model scores 18/22 on the held-out pairs.

Main entry point. Works from any directory; the project folder can be moved anywhere.

    python run.py train                                  # train from scratch -> checkpoints/morgan_pgb_8020_retrain_seed42/
    python run.py train --seed 43 --device cpu
    python run.py train --split random                   # other split -> checkpoints/morgan_pgb_random_seed42/
    python run.py evaluate                               # metrics of the shipped model (val + held-out)
    python run.py evaluate --checkpoint checkpoints/morgan_pgb_random_seed42
    python run.py predict --csv my_pairs.csv             # score new pairs (needs API_Smiles, Excipient_Smiles)
    python run.py split                                  # rebuild the default split from the raw dataset and check it

Splits are the folders in data/splits/ (api_cluster = default, random, excipient_new, api_excipient_new,
api_cluster06). Every split is tested on the same 22 held-out pairs in data/heldout/.

Recipe:
  - PGB descriptor norm stats and excipient prior table from the train split only
  - leave-own-cluster-out excipient priors for train rows (cluster_id column)
  - asymmetric focal loss (gamma+ 1.2, gamma- 2.5) with pos_weight, no balanced sampler
  - early stopping and LR schedule on val PR-AUC
  - threshold on val: best MCC with true-negative rate >= 0.97
  - the 22 held-out pairs are only scored, never used for any choice
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # make `apiexc` importable from any directory

import pandas as pd  # noqa: E402
from rdkit import RDLogger  # noqa: E402

from apiexc import paths  # noqa: E402
from apiexc.dataset import PGBNormStats  # noqa: E402
from apiexc.evaluation.metrics import (collect_predictions, metric_suite, prediction_table, print_metrics,  # noqa: E402
                                       print_predictions, save_metrics, tune_threshold)
from apiexc.features.excipient_prior import ExcipientPriorTable, build_leave_cluster_out_prior_vectors  # noqa: E402
from apiexc.models.pgb_model import pgb_model_from_config  # noqa: E402
from apiexc.pipeline import (SPLIT_FILE, build_encoder, check_heldout_unseen, checkpoint_split, get_split,  # noqa: E402
                             load_checkpoint, make_config, make_loader, md5)
from apiexc.utils import get_device, seed_everything  # noqa: E402

RDLogger.DisableLog("rdApp.*")   # silence "not removing hydrogen atom without neighbors" on inorganic salts

SPLITS = ["val", "heldout"]


# --------------------------------------------------------------------------- evaluate
def evaluate(ckpt_dir, split=None, device_str=None):
    """Score a checkpoint on val and the held-out pairs; write metrics and predictions to results/evaluation/."""
    split = split or checkpoint_split(ckpt_dir)
    check_heldout_unseen(split)
    lm = load_checkpoint(ckpt_dir, get_device(device_str or "auto"))

    data, probs, labels = {}, {}, {}
    for s in SPLITS:
        data[s] = split.frame(s)
        probs[s], labels[s] = lm.score(data[s])

    tuned, _ = tune_threshold(probs["val"], labels["val"], lm.config.threshold_step, tnr_floor=lm.config.pgb_tnr_floor)
    if lm.threshold is not None:
        threshold = lm.threshold
        thr_note = f"{threshold:.3f} (saved with the model; re-derived from val now: {tuned:.3f})"
    else:
        threshold = float(tuned)
        thr_note = f"{threshold:.3f} (derived from val: best MCC with TNR >= {lm.config.pgb_tnr_floor})"

    metrics = {s: metric_suite(probs[s], labels[s], threshold) for s in SPLITS}
    tables = {s: prediction_table(data[s], probs[s], labels[s], threshold) for s in SPLITS}

    out_dir = paths.EVALUATION_RESULTS / lm.folder.name
    os.makedirs(out_dir, exist_ok=True)
    for s in SPLITS:
        save_metrics(metrics[s], str(out_dir / f"{s}_metrics.json"))
        tables[s].to_csv(out_dir / f"predictions_{s}.csv", index=False)

    print("=" * 100)
    print("Morgan-PGB  |  checkpoint:", paths.rel(lm.folder))
    print(f"split {split.name}: {split.description}")
    print(f"device {lm.device} | best epoch {lm.info.get('epoch')} | val PR-AUC at best epoch "
          f"{lm.info.get('val_pr_auc', float('nan')):.4f}")
    print(f"rows: train {len(split.frame('train'))} | val {len(data['val'])} | held-out {len(data['heldout'])} "
          f"(no held-out pair in train or val)")
    print(f"threshold {thr_note}")
    print("=" * 100)
    print_metrics(metrics)
    print(f"\nHeld-out predictions ({metrics['heldout']['correct']}/{metrics['heldout']['n']} correct, "
          f"label 1 = incompatible, prediction = probability >= {threshold:.3f}):")
    print_predictions(tables["heldout"])

    if lm.folder.resolve() == paths.FINAL_CHECKPOINT.resolve():
        print(f"\nCheck against the original run ({paths.rel(paths.REFERENCE_RESULTS)}/):")
        for s in SPLITS:
            ref = json.loads((paths.REFERENCE_RESULTS / f"{s}_metrics.json").read_text())
            print(f"  {s:8s} confusion matrix now {metrics[s]['confusion_matrix']} vs original {ref['confusion_matrix']} | "
                  f"PR-AUC now {metrics[s]['pr_auc']:.4f} vs original {ref['pr_auc']:.4f}")
    print(f"\nSaved metrics and predictions to {out_dir}")
    return metrics


# --------------------------------------------------------------------------- predict
def predict(csv, ckpt_dir, device_str=None, out=None):
    """Score a CSV of pairs (API_Smiles, Excipient_Smiles; label column Outcome1 or ground_truth optional)."""
    lm = load_checkpoint(ckpt_dir, get_device(device_str or "auto"))
    df = pd.read_csv(csv)
    missing = [c for c in ["API_Smiles", "Excipient_Smiles"] if c not in df]
    if missing:
        raise SystemExit(f"{csv} is missing columns {missing}")
    has_label = "Outcome1" in df or "ground_truth" in df
    df = df.rename(columns={"ground_truth": "Outcome1"})
    if not has_label:
        df["Outcome1"] = 0   # PGBDataset needs a label column; the value is not used for scoring
    probs, labels = lm.score(df)
    threshold = lm.threshold if lm.threshold is not None else 0.5
    table = prediction_table(df, probs, labels, threshold)
    if not has_label:
        table = table.drop(columns=["label", "correct"])
    out = out or paths.PREDICTION_RESULTS / f"predictions_{os.path.splitext(os.path.basename(csv))[0]}.csv"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    table.to_csv(out, index=False)
    print(table.drop(columns=[c for c in ["API_Smiles", "Excipient_Smiles"] if c in table]).round(4).to_string())
    if has_label:
        print()
        print(pd.Series(metric_suite(probs, labels, threshold)).to_string())
    print(f"\nthreshold {threshold:.3f} | saved {os.path.abspath(out)}")


# --------------------------------------------------------------------------- train
def train(split, seed, device_str=None, epochs=None, overwrite=False):
    """Train from scratch on a split, save the checkpoint, then evaluate it."""
    from apiexc.training.trainer import pgb_train

    check_heldout_unseen(split)
    # the default split keeps the original name; another split gets its own, e.g. morgan_pgb_random_seed42
    tag = "8020_retrain" if split.name == paths.DEFAULT_SPLIT else split.name.replace("/", "_")
    ckpt_dir = paths.CHECKPOINTS / f"morgan_pgb_{tag}_seed{seed}"
    if (ckpt_dir / "best_model.pt").exists() and not overwrite:
        raise SystemExit(f"{ckpt_dir} already exists; pass --overwrite to retrain")
    os.makedirs(ckpt_dir, exist_ok=True)
    (ckpt_dir / SPLIT_FILE).write_text(split.name + "\n")
    out_dir = paths.EVALUATION_RESULTS / ckpt_dir.name
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    seed_everything(seed)
    config = make_config(split, seed, device_str or "auto", epochs)
    device = get_device(config.device)
    config.checkpoint_dir, config.metrics_dir = str(ckpt_dir), str(out_dir)
    train_df, val_df = pd.read_csv(split.train_csv), pd.read_csv(split.val_csv)

    encoder = build_encoder(config, device)

    # PGB descriptor norm stats from the unique training SMILES only
    all_train_smiles = list(set(train_df["API_Smiles"].dropna().tolist() + train_df["Excipient_Smiles"].dropna().tolist()))
    pgb_norm = PGBNormStats.from_smiles_list(all_train_smiles)
    pgb_norm.save(str(ckpt_dir / "pgb_norm_stats.npz"))

    # excipient prior table from train rows (global rate = train positive rate); train rows get
    # leave-own-cluster-out priors
    prior_table = ExcipientPriorTable(global_rate=config.positive_prior)
    prior_table.fit(train_df, leave_out_cluster_id=None)
    prior_table.save(str(ckpt_dir / "prior_table.json"))
    train_priors = build_leave_cluster_out_prior_vectors(train_df, cluster_col="cluster_id",
                                                         global_rate=prior_table.global_rate)

    train_loader = make_loader(split.train_csv, config, prior_table, pgb_norm, encoder, row_priors=train_priors, shuffle=True)
    val_loader = make_loader(split.val_csv, config, prior_table, pgb_norm, encoder)

    model = pgb_model_from_config(config, encoder)
    model.to(device)
    print(f"[Morgan-PGB | split {split.name} | seed {seed} | {device}] train {len(train_df)} | val {len(val_df)} | "
          f"max epochs {config.pgb_max_epochs} | checkpoint -> {ckpt_dir}")
    model, history = pgb_train(model, train_loader, val_loader, config, device)

    val_probs, val_labels = collect_predictions(model, val_loader, device)
    threshold, _ = tune_threshold(val_probs, val_labels, config.threshold_step, tnr_floor=config.pgb_tnr_floor)
    (ckpt_dir / "threshold.json").write_text(json.dumps(
        {"threshold": float(threshold), "rule": f"best val MCC with TNR >= {config.pgb_tnr_floor}"}, indent=2))

    save_metrics({"history": history}, str(out_dir / "training_history.json"))
    (out_dir / "run_info.json").write_text(json.dumps({
        "model": "Morgan-PGB", "split_name": split.name, "split": split.description,
        "seed": seed, "device": str(device),
        "data": {name: {"path": paths.rel(p), "rows": len(pd.read_csv(p)), "md5": md5(p)}
                 for name, p in [("train", split.train_csv), ("val", split.val_csv), ("heldout", split.heldout_csv)]},
        "epochs_trained": len(history["val_pr_auc"]),
        "best_epoch": int(pd.Series(history["val_pr_auc"]).idxmax()) + 1,
        "threshold_from_val": float(threshold), "tnr_floor": config.pgb_tnr_floor,
        "finished": datetime.now().isoformat(timespec="seconds"), "minutes": round((time.time() - t0) / 60, 2),
    }, indent=2))

    print()
    evaluate(ckpt_dir, split, device_str)


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Morgan-PGB API-excipient compatibility model (80/20 split, 18/22 held-out).")
    sub = p.add_subparsers(dest="cmd")
    split_help = f"a folder name in data/splits/ ({', '.join(paths.available_splits())}) or a path to a folder with train.csv / val.csv"

    t = sub.add_parser("train", help="train from scratch, save the checkpoint, then evaluate it")
    t.add_argument("--split", default=paths.DEFAULT_SPLIT, help=f"{split_help} (default {paths.DEFAULT_SPLIT})")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default=None, help="cpu, cuda (default: cuda if available)")
    t.add_argument("--epochs", type=int, default=None, help="max epochs (default 40, early stopping on val PR-AUC)")
    t.add_argument("--overwrite", action="store_true", help="retrain into an existing checkpoint folder")

    e = sub.add_parser("evaluate", help="metrics on val / held-out + held-out predictions")
    e.add_argument("--checkpoint", default=str(paths.FINAL_CHECKPOINT))
    e.add_argument("--split", default=None, help=f"{split_help} (default: the split the checkpoint was trained on)")
    e.add_argument("--device", default=None)

    pr = sub.add_parser("predict", help="score a CSV of pairs (API_Smiles, Excipient_Smiles; optional Outcome1/ground_truth)")
    pr.add_argument("--csv", required=True)
    pr.add_argument("--checkpoint", default=str(paths.FINAL_CHECKPOINT))
    pr.add_argument("--out", default=None, help=f"output CSV (default {paths.rel(paths.PREDICTION_RESULTS)}/predictions_<csv name>.csv)")
    pr.add_argument("--device", default=None)

    sub.add_parser("split", help="rebuild the default train / val split from data/raw/start_dataset.csv and check it matches")

    args = p.parse_args()
    if args.cmd == "train":
        train(get_split(args.split), args.seed, args.device, args.epochs, args.overwrite)
    elif args.cmd == "evaluate":
        ckpt = paths.resolve(args.checkpoint)
        evaluate(ckpt, get_split(args.split) if args.split else None, args.device)
    elif args.cmd == "predict":
        predict(paths.resolve(args.csv), paths.resolve(args.checkpoint), args.device,
                os.path.abspath(args.out) if args.out else None)
    elif args.cmd == "split":
        from apiexc.splitting import rebuild_split
        ok = rebuild_split()
        print(f"\nRebuilt split and split_log.txt in {paths.REBUILT_SPLIT} -> "
              + (f"matches data/splits/{paths.DEFAULT_SPLIT}/ exactly" if ok else "DIFFERS from the shipped split"))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
