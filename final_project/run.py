"""
Final project: Morgan-PGB (Prior-Gated Bilinear head on Morgan + MACCS fingerprints),
trained on the 80/20 split (train / val). Scores 20/24 on the held-out pairs.

Everything this script needs is inside this folder; it can be moved anywhere.

Usage (from any directory):
    python run.py train                               # train from scratch -> checkpoints/morgan_pgb_8020_retrain_seed42/
    python run.py train --seed 43 --device cpu
    python run.py evaluate                            # metrics of the shipped model (val + held-out)
    python run.py evaluate --checkpoint checkpoints/morgan_pgb_8020_retrain_seed42
    python run.py predict --csv my_pairs.csv          # score new pairs (needs API_Smiles, Excipient_Smiles)
    python run.py split                               # rebuild the 80/20 split from start_dataset + write the split log

Recipe (same as run_pgb_single.py --family morgan -> pgb_single_run() at git commit f534ef7):
  - split: Butina API clusters, GroupShuffleSplit(test_size=0.2, random_state=42) -> train 2770 / val 774
  - PGB descriptor norm stats and excipient prior table from the train split only
  - leave-own-cluster-out excipient priors for train rows (cluster_id column)
  - asymmetric focal loss (gamma+ 1.2, gamma- 2.5) with pos_weight, no balanced sampler
  - early stopping and LR schedule on val PR-AUC
  - threshold on val: best MCC with true-negative rate >= 0.97
  - the 24 held-out pairs are only scored, never used for any choice
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CALLER_CWD = os.getcwd()
sys.path.insert(0, HERE)   # `src` resolves to final_project/src (the checkpoint pickles src.config.Config)
os.chdir(HERE)             # all relative paths below are inside this folder

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import PGBDataset, PGBNormStats, pgb_collate_fn
from src.encoders import ENCODER_REGISTRY
from src.evaluate import collect_predictions, compute_metrics, save_metrics, tune_threshold
from src.model import pgb_model_from_config
from src.pgb_prior import ExcipientPriorTable, build_leave_cluster_out_prior_vectors
from src.utils import get_device, seed_everything

RDLogger.DisableLog("rdApp.*")   # silence "not removing hydrogen atom without neighbors" on inorganic salts

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
HELDOUT_CSV = "data/held_out_test_set.csv"
START_CSV = "data/start_dataset.csv"
FINAL_CKPT = "checkpoints/morgan_pgb_8020_seed42"
REFERENCE = "reference_results"
METRICS = "metrics"
SPLITS = ["val", "heldout"]


# --------------------------------------------------------------------------- helpers
def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def metric_suite(probs, labels, threshold):
    m = compute_metrics(probs, labels, threshold)
    labels = labels.astype(int)
    m["roc_auc"] = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan")
    cm = m["confusion_matrix"]
    m["tn"], m["fp"], m["fn"], m["tp"] = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    m["n"] = int(len(labels))
    m["correct"] = int(m["tp"] + m["tn"])
    return m


def make_config(seed=42, device="auto", epochs=None):
    """Training config for the Morgan-PGB 80/20 run, built the same way as run_pgb_single.py:
    Config() defaults + FAMILIES["morgan"] + the overrides in pgb_single_run()."""
    config = Config()
    config.use_pgb_head = True
    config.split_type = "cluster"
    config.seed = seed
    # FAMILIES["morgan"]
    config.encoder = "fixed_vector"
    config.fixed_vector_source = "morgan"
    config.fusion = "concat"
    config.encoder_output_dim = 1024   # set by the original build_encoder from data/morgan_fps.csv; metadata only on the concat path
    config.pgb_max_epochs = epochs or 40
    config.device = device
    config.resolve_paths()
    # pgb_single_run() overrides
    config.train_csv, config.val_csv, config.test_csv = TRAIN_CSV, VAL_CSV, HELDOUT_CSV
    config.positive_prior = float(pd.read_csv(TRAIN_CSV)["Outcome1"].mean())
    config.use_balanced_sampler = False
    config.loss = "asym_focal"
    return config


def build_encoder(config, device):
    # Morgan-PGB computes its fingerprints from SMILES in PGBDataset; the encoder holds no
    # weights and only routes the model to the concat (no structural branch) path.
    encoder = ENCODER_REGISTRY[config.encoder](source=config.fixed_vector_source, vector_path=None, device=str(device))
    encoder.to(device)
    return encoder


def make_loader(csv, config, prior_table, pgb_norm, encoder, row_priors=None, shuffle=False):
    ds = PGBDataset(csv, config.encoder, config.fixed_vector_source, prior_table, pgb_norm,
                    fixed_vector_encoder=encoder, row_priors=row_priors)
    return DataLoader(ds, batch_size=config.pgb_batch_size, shuffle=shuffle, collate_fn=pgb_collate_fn, num_workers=0)


def frame_for(split):
    """DataFrame for a split, with the label in Outcome1 (as PGBDataset expects)."""
    if split == "heldout":
        return pd.read_csv(HELDOUT_CSV).rename(columns={"ground_truth": "Outcome1"})
    return pd.read_csv({"train": TRAIN_CSV, "val": VAL_CSV}[split])


def score_frame(model, df, config, prior_table, pgb_norm, encoder, device):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pairs.csv")
        df.to_csv(path, index=False)
        return collect_predictions(model, make_loader(path, config, prior_table, pgb_norm, encoder), device)


def load_checkpoint(ckpt_dir, device):
    ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device, weights_only=False)
    config = ckpt["config"]
    encoder = build_encoder(config, device)
    model = pgb_model_from_config(config, encoder)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    prior_table = ExcipientPriorTable.load(os.path.join(ckpt_dir, "prior_table.json"))
    pgb_norm = PGBNormStats.load(os.path.join(ckpt_dir, "pgb_norm_stats.npz"))
    return model, config, encoder, prior_table, pgb_norm, ckpt


def prediction_table(df, probs, labels, threshold):
    cols = [c for c in ["Api_name", "Excipient_name", "API_CID", "Excipient_CID", "API_Smiles", "Excipient_Smiles"] if c in df]
    out = df[cols].copy()
    out["label"], out["probability"] = labels.astype(int), probs
    out["prediction"] = (probs >= threshold).astype(int)
    out["correct"] = (out["prediction"] == out["label"]).astype(int)
    return out


def heldout_pairs_in_train(heldout_df):
    """Boolean mask: held-out pairs whose (API_CID, Excipient_CID) also occurs in train.csv."""
    train = pd.read_csv(TRAIN_CSV)
    seen = set(zip(train["API_CID"], train["Excipient_CID"]))
    return np.array([(a, e) in seen for a, e in zip(heldout_df["API_CID"], heldout_df["Excipient_CID"])])


def print_metrics(metrics):
    rows = []
    for s, m in metrics.items():
        rows.append({"split": s, "n": m["n"], "correct": f"{m['correct']}/{m['n']}", "TP": m["tp"], "TN": m["tn"],
                     "FP": m["fp"], "FN": m["fn"], "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"], "MCC": m["mcc"],
                     "F1": m["f1"], "precision": m["precision"], "recall": m["recall"], "accuracy": m["accuracy"]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def print_heldout(table):
    t = table.copy()
    t["result"] = np.where(t["correct"] == 1, "ok", "WRONG")
    t["probability"] = t["probability"].round(4)
    show = [c for c in ["Api_name", "Excipient_name", "label", "probability", "prediction", "result"] if c in t]
    print(t[show].to_string(index=True))


# --------------------------------------------------------------------------- evaluate
def evaluate(ckpt_dir, device_str=None):
    device = get_device(device_str or "auto")
    model, config, encoder, prior_table, pgb_norm, ckpt = load_checkpoint(ckpt_dir, device)

    data, probs, labels = {}, {}, {}
    for s in SPLITS:
        data[s] = frame_for(s)
        probs[s], labels[s] = score_frame(model, data[s], config, prior_table, pgb_norm, encoder, device)

    tuned, _ = tune_threshold(probs["val"], labels["val"], config.threshold_step, tnr_floor=config.pgb_tnr_floor)
    thr_file = os.path.join(ckpt_dir, "threshold.json")
    if os.path.exists(thr_file):
        threshold = json.load(open(thr_file))["threshold"]
        thr_note = f"{threshold:.3f} (saved with the model; re-derived from val now: {tuned:.3f})"
    else:
        threshold = float(tuned)
        thr_note = f"{threshold:.3f} (derived from val: best MCC with TNR >= {config.pgb_tnr_floor})"

    metrics = {s: metric_suite(probs[s], labels[s], threshold) for s in SPLITS}
    tables = {s: prediction_table(data[s], probs[s], labels[s], threshold) for s in SPLITS}

    # held-out pairs the model never saw in training (2 of the 24 are also in train.csv)
    unseen = ~heldout_pairs_in_train(data["heldout"])
    metrics["heldout_unseen"] = metric_suite(probs["heldout"][unseen], labels["heldout"][unseen], threshold)

    out_dir = os.path.join(METRICS, os.path.basename(os.path.normpath(ckpt_dir)))
    os.makedirs(out_dir, exist_ok=True)
    for s in metrics:
        save_metrics(metrics[s], os.path.join(out_dir, f"{s}_metrics.json"))
    for s in SPLITS:
        tables[s].to_csv(os.path.join(out_dir, f"predictions_{s}.csv"), index=False)

    print("=" * 100)
    print("Morgan-PGB  |  80/20 split  |  checkpoint:", ckpt_dir)
    print(f"device {device} | best epoch {ckpt.get('epoch')} | val PR-AUC at best epoch {ckpt.get('val_pr_auc', float('nan')):.4f}")
    print(f"rows: train {len(frame_for('train'))} | val {len(data['val'])} | held-out {len(data['heldout'])}")
    print(f"threshold {thr_note}")
    print("=" * 100)
    print_metrics(metrics)
    print(f"(heldout_unseen = the {int(unseen.sum())} held-out pairs that are not also in train.csv)")
    print(f"\nHeld-out predictions ({metrics['heldout']['correct']}/{metrics['heldout']['n']} correct, "
          f"label 1 = incompatible, prediction = probability >= {threshold:.3f}):")
    print_heldout(tables["heldout"])

    if os.path.normpath(ckpt_dir) == os.path.normpath(FINAL_CKPT):
        print("\nCheck against the original run (reference_results/):")
        for s in SPLITS:
            ref_m = json.load(open(os.path.join(REFERENCE, f"{s}_metrics.json")))
            ref_cm = ref_m["confusion_matrix"]
            print(f"  {s:8s} confusion matrix now {metrics[s]['confusion_matrix']} vs original {ref_cm} | "
                  f"PR-AUC now {metrics[s]['pr_auc']:.4f} vs original {ref_m['pr_auc']:.4f}")
    print(f"\nSaved metrics and predictions to {os.path.join(HERE, out_dir)}")
    return metrics


# --------------------------------------------------------------------------- predict
def predict(csv, ckpt_dir, device_str=None, out=None):
    device = get_device(device_str or "auto")
    model, config, encoder, prior_table, pgb_norm, _ = load_checkpoint(ckpt_dir, device)
    df = pd.read_csv(csv)
    missing = [c for c in ["API_Smiles", "Excipient_Smiles"] if c not in df]
    if missing:
        raise SystemExit(f"{csv} is missing columns {missing}")
    has_label = "Outcome1" in df or "ground_truth" in df
    df = df.rename(columns={"ground_truth": "Outcome1"})
    if not has_label:
        df["Outcome1"] = 0
    probs, labels = score_frame(model, df, config, prior_table, pgb_norm, encoder, device)
    thr_file = os.path.join(ckpt_dir, "threshold.json")
    threshold = json.load(open(thr_file))["threshold"] if os.path.exists(thr_file) else 0.5
    table = prediction_table(df, probs, labels, threshold)
    if not has_label:
        table = table.drop(columns=["label", "correct"])
    out = out or os.path.join(METRICS, "predictions_" + os.path.splitext(os.path.basename(csv))[0] + ".csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    table.to_csv(out, index=False)
    print(table.drop(columns=[c for c in ["API_Smiles", "Excipient_Smiles"] if c in table]).round(4).to_string())
    if has_label:
        print()
        print(pd.Series(metric_suite(probs, labels, threshold)).to_string())
    print(f"\nthreshold {threshold:.3f} | saved {os.path.abspath(out)}")


# --------------------------------------------------------------------------- train
def train(seed, device_str=None, epochs=None, overwrite=False):
    from src.train import pgb_train

    ckpt_dir = f"checkpoints/morgan_pgb_8020_retrain_seed{seed}"
    if os.path.exists(os.path.join(ckpt_dir, "best_model.pt")) and not overwrite:
        raise SystemExit(f"{ckpt_dir} already exists; pass --overwrite to retrain")
    os.makedirs(ckpt_dir, exist_ok=True)
    out_dir = os.path.join(METRICS, os.path.basename(ckpt_dir))
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ---- same order of operations as pgb_single_run() ----
    seed_everything(seed)
    config = make_config(seed, device_str or "auto", epochs)
    device = get_device(config.device)
    config.checkpoint_dir, config.metrics_dir = ckpt_dir, out_dir
    train_df, val_df = pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)

    encoder = build_encoder(config, device)

    # PGB descriptor norm stats from the unique training SMILES only
    all_train_smiles = list(set(train_df["API_Smiles"].dropna().tolist() + train_df["Excipient_Smiles"].dropna().tolist()))
    pgb_norm = PGBNormStats.from_smiles_list(all_train_smiles)
    pgb_norm.save(os.path.join(ckpt_dir, "pgb_norm_stats.npz"))

    # excipient prior table from train rows; train rows get leave-own-cluster-out priors
    prior_table = ExcipientPriorTable()
    prior_table.fit(train_df, leave_out_cluster_id=None)
    prior_table.save(os.path.join(ckpt_dir, "prior_table.json"))
    train_priors = build_leave_cluster_out_prior_vectors(train_df, cluster_col="cluster_id")

    train_loader = make_loader(TRAIN_CSV, config, prior_table, pgb_norm, encoder, row_priors=train_priors, shuffle=True)
    val_loader = make_loader(VAL_CSV, config, prior_table, pgb_norm, encoder)

    model = pgb_model_from_config(config, encoder)
    model.to(device)
    print(f"[Morgan-PGB 80/20 | seed {seed} | {device}] train {len(train_df)} | val {len(val_df)} | "
          f"max epochs {config.pgb_max_epochs} | checkpoint -> {os.path.join(HERE, ckpt_dir)}")
    model, history = pgb_train(model, train_loader, val_loader, config, device)

    val_probs, val_labels = collect_predictions(model, val_loader, device)
    threshold, _ = tune_threshold(val_probs, val_labels, config.threshold_step, tnr_floor=config.pgb_tnr_floor)
    with open(os.path.join(ckpt_dir, "threshold.json"), "w") as f:
        json.dump({"threshold": float(threshold), "rule": f"best val MCC with TNR >= {config.pgb_tnr_floor}"}, f, indent=2)

    save_metrics({"history": history}, os.path.join(out_dir, "training_history.json"))
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump({
            "model": "Morgan-PGB", "split": "80/20 (Butina API clusters, GroupShuffleSplit random_state=42)",
            "seed": seed, "device": str(device),
            "data": {name: {"path": p, "rows": len(pd.read_csv(p)), "md5": md5(p)}
                     for name, p in [("train", TRAIN_CSV), ("val", VAL_CSV), ("heldout", HELDOUT_CSV)]},
            "epochs_trained": len(history["val_pr_auc"]),
            "best_epoch": int(pd.Series(history["val_pr_auc"]).idxmax()) + 1,
            "threshold_from_val": float(threshold), "tnr_floor": config.pgb_tnr_floor,
            "finished": datetime.now().isoformat(timespec="seconds"), "minutes": round((time.time() - t0) / 60, 2),
        }, f, indent=2)

    print()
    evaluate(ckpt_dir, device_str)


# --------------------------------------------------------------------------- split
def butina_cluster(unique_smiles, dist_thresh=0.15):
    """Butina clustering of API SMILES on Morgan fingerprints (radius 2, 2048 bits),
    identical to src/cross_validate.py at commit f534ef7."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in unique_smiles]
    dists = []
    for i in range(1, len(fps)):
        dists.extend(1.0 - s for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    clusters = Butina.ClusterData(dists, len(fps), dist_thresh, isDistData=True)
    return {unique_smiles[m]: cid for cid, members in enumerate(clusters) for m in members}


def rebuild_split():
    """Build train / val from data/start_dataset.csv, write a split log, and check the result
    is identical to data/train.csv and data/val.csv.

    1. Butina-cluster the unique API SMILES (Morgan r=2, 2048 bits, distance cutoff 0.15) -> cluster_id
    2. GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) over API clusters -> train / val
       (not stratified on the label)
    The 24 held-out pairs are the external test set.
    """
    import sklearn
    from rdkit import rdBase
    from sklearn.model_selection import GroupShuffleSplit

    out_dir = os.path.join(METRICS, "rebuilt_split")
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    def stats(name, d):
        return (f"{name:7s} rows={len(d):5d} | incompatible={int(d['Outcome1'].sum()):4d} ({d['Outcome1'].mean():5.1%}) | "
                f"compatible={int((d['Outcome1'] == 0).sum()):5d} | unique APIs={d['API_CID'].nunique():4d} | "
                f"unique excipients={d['Excipient_CID'].nunique():4d}")

    log(f"Split rebuilt {datetime.now().isoformat(timespec='seconds')} by `python run.py split`")
    log(f"rdkit {rdBase.rdkitVersion} | scikit-learn {sklearn.__version__} | pandas {pd.__version__}")
    log()

    raw = pd.read_csv(START_CSV)
    unique_apis = raw["API_Smiles"].unique().tolist()
    smiles_to_cluster = butina_cluster(unique_apis)
    raw["cluster_id"] = raw["API_Smiles"].map(smiles_to_cluster)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(raw, y=raw["Outcome1"], groups=raw["cluster_id"]))
    splits = {"train": raw.iloc[train_idx].reset_index(drop=True), "val": raw.iloc[val_idx].reset_index(drop=True)}
    heldout = frame_for("heldout")

    log(f"Input {START_CSV}: {len(raw)} pairs, {raw['API_CID'].nunique()} APIs, {raw['Excipient_CID'].nunique()} excipients "
        f"(md5 {md5(START_CSV)})")
    log(f"Butina (Morgan r=2, 2048 bits, distance cutoff 0.15): {len(unique_apis)} unique API SMILES -> "
        f"{len(set(smiles_to_cluster.values()))} clusters")
    log("GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) on cluster_id")
    log()
    log(stats("train", splits["train"]))
    log(stats("val", splits["val"]))
    log(stats("heldout", heldout))
    log()
    tr, va = splits["train"], splits["val"]
    log(f"API cluster overlap train<->val: {len(set(tr.cluster_id) & set(va.cluster_id))}")
    log(f"API_CID overlap train<->val:     {len(set(tr.API_CID) & set(va.API_CID))}")
    log(f"Excipient_CID overlap train<->val (expected): {len(set(tr.Excipient_CID) & set(va.Excipient_CID))} "
        f"of {va.Excipient_CID.nunique()} val excipients")
    log()
    ho_pairs = set(zip(heldout.API_CID, heldout.Excipient_CID))
    for name, d in splits.items():
        hits = d[[(a, e) in ho_pairs for a, e in zip(d.API_CID, d.Excipient_CID)]]
        for _, r in hits.iterrows():
            log(f"Held-out pair also in {name}: API_CID {r.API_CID} + Excipient_CID {r.Excipient_CID} (label {r.Outcome1})")
    log()

    ok = True
    for name, ref in [("train", TRAIN_CSV), ("val", VAL_CSV)]:
        path = os.path.join(out_dir, f"{name}.csv")
        splits[name].to_csv(path, index=False)
        same_rows = pd.read_csv(path).equals(pd.read_csv(ref))
        ok &= same_rows
        log(f"{name}.csv: {len(splits[name])} rows | identical rows/order to {ref} = {same_rows} | "
            f"identical bytes = {md5(path) == md5(ref)}")

    with open(os.path.join(out_dir, "split_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nRebuilt split and split_log.txt in {os.path.join(HERE, out_dir)} -> "
          + ("matches data/ exactly" if ok else "DIFFERS from data/"))


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Final Morgan-PGB model (80/20 split, 20/24 held-out).")
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("train", help="train from scratch into checkpoints/morgan_pgb_8020_retrain_seed<seed>/, then evaluate")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default=None, help="cpu, cuda (default: cuda if available)")
    t.add_argument("--epochs", type=int, default=None, help="max epochs (default 40, early stopping on val PR-AUC)")
    t.add_argument("--overwrite", action="store_true")

    e = sub.add_parser("evaluate", help="metrics on val / held-out + held-out predictions")
    e.add_argument("--checkpoint", default=FINAL_CKPT)
    e.add_argument("--device", default=None)

    pr = sub.add_parser("predict", help="score a CSV of pairs (API_Smiles, Excipient_Smiles; optional Outcome1/ground_truth)")
    pr.add_argument("--csv", required=True)
    pr.add_argument("--checkpoint", default=FINAL_CKPT)
    pr.add_argument("--out", default=None)
    pr.add_argument("--device", default=None)

    sub.add_parser("split", help="rebuild train/val from data/start_dataset.csv, write the split log, check it matches data/")

    args = p.parse_args()
    # make --checkpoint / --csv paths given relative to the caller's directory work after chdir
    for attr in ("checkpoint", "csv"):
        v = getattr(args, attr, None)
        if v and not os.path.isabs(v) and os.path.exists(os.path.join(CALLER_CWD, v)) and not os.path.exists(v):
            setattr(args, attr, os.path.join(CALLER_CWD, v))
    if args.cmd == "predict" and args.out and not os.path.isabs(args.out):
        args.out = os.path.join(CALLER_CWD, args.out)

    if args.cmd == "train":
        train(args.seed, args.device, args.epochs, args.overwrite)
    elif args.cmd == "evaluate":
        evaluate(args.checkpoint, args.device)
    elif args.cmd == "predict":
        predict(args.csv, args.checkpoint, args.device, args.out)
    elif args.cmd == "split":
        rebuild_split()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
