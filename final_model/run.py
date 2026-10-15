"""
Final model: Morgan-PGB (Prior-Gated Bilinear head on Morgan + MACCS fingerprints),
trained on the 60/20/20 split (train / val / test). Scores 19/24 on the held-out pairs.

Everything this script needs is inside this folder; it can be moved anywhere.

Usage (from any directory):
    python run.py                         # evaluate the shipped model: val, test, held-out metrics + held-out predictions
    python run.py evaluate --checkpoint checkpoints/retrain_seed_43
    python run.py predict --csv my_pairs.csv          # score new pairs (needs API_Smiles, Excipient_Smiles)
    python run.py train --seed 42                     # retrain from scratch -> checkpoints/retrain_seed_42/
    python run.py split                               # rebuild the split from start_dataset + write the split log

Recipe (same as experiments/oct6_resplit/run.py in the original project):
  - PGB descriptor norm stats and excipient prior table from the train split only
  - leave-own-cluster-out excipient priors for train rows (cluster_id column)
  - early stopping and LR schedule on val PR-AUC
  - threshold on val: best MCC with true-negative rate >= 0.97
  - test and held-out are only scored, never used for any choice
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CALLER_CWD = os.getcwd()
sys.path.insert(0, HERE)   # `src` resolves to final_model/src (the checkpoint pickles src.config.Config)
os.chdir(HERE)             # all relative paths below are inside this folder

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import PGBDataset, PGBNormStats, pgb_collate_fn
from src.encoders import ENCODER_REGISTRY
from src.evaluate import collect_predictions, compute_metrics, save_metrics, tune_threshold
from src.model import pgb_model_from_config
from src.pgb_prior import ExcipientPriorTable, build_leave_cluster_out_prior_vectors
from src.utils import get_device, seed_everything

RDLogger.DisableLog("rdApp.*")   # silence "not removing hydrogen atom without neighbors" on inorganic salts

DATA = {s: f"data/{s}.csv" for s in ["train", "val", "test"]}
HELDOUT_CSV = "data/held_out_test_set.csv"
START_CSV = "data/start_dataset.csv"
FINAL_CKPT = "checkpoints/morgan_pgb_seed42"
REFERENCE = "reference_results"
OUTPUTS = "outputs"
SPLITS = ["val", "test", "heldout"]


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
    return pd.read_csv(DATA[split])


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


def print_metrics(metrics):
    rows = []
    for s in SPLITS:
        m = metrics[s]
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
def evaluate(ckpt_dir, device_str=None, quiet_reference=False):
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

    out_dir = os.path.join(OUTPUTS, os.path.basename(os.path.normpath(ckpt_dir)))
    os.makedirs(out_dir, exist_ok=True)
    for s in SPLITS:
        save_metrics(metrics[s], os.path.join(out_dir, f"{s}_metrics.json"))
        tables[s].to_csv(os.path.join(out_dir, f"predictions_{s}.csv"), index=False)

    print("=" * 100)
    print("Morgan-PGB  |  60/20/20 split  |  checkpoint:", ckpt_dir)
    print(f"device {device} | best epoch {ckpt.get('epoch')} | val PR-AUC at best epoch {ckpt.get('val_pr_auc', float('nan')):.4f}")
    print(f"rows: train {len(frame_for('train'))} | val {len(data['val'])} | test {len(data['test'])} | held-out {len(data['heldout'])}")
    print(f"threshold {thr_note}")
    print("=" * 100)
    print_metrics(metrics)
    print(f"\nHeld-out predictions ({metrics['heldout']['correct']}/{metrics['heldout']['n']} correct, "
          f"label 1 = incompatible, prediction = probability >= {threshold:.3f}):")
    print_heldout(tables["heldout"])

    if os.path.normpath(ckpt_dir) == os.path.normpath(FINAL_CKPT) and not quiet_reference:
        print("\nCheck against the original run (reference_results/):")
        for s in SPLITS:
            ref = pd.read_csv(os.path.join(REFERENCE, f"predictions_{s}.csv"))
            ref_m = json.load(open(os.path.join(REFERENCE, f"{s}_metrics.json")))
            diff = float(np.abs(ref["probability"].values - probs[s]).max())
            same = (ref["prediction"].values == tables[s]["prediction"].values).all()
            print(f"  {s:8s} max |prob diff| {diff:.2e} | same predictions: {same} | "
                  f"correct now {metrics[s]['correct']} vs original {ref_m['correct']}")
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
    out = out or os.path.join(OUTPUTS, "predictions_" + os.path.splitext(os.path.basename(csv))[0] + ".csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    table.to_csv(out, index=False)
    print(table.drop(columns=[c for c in ["API_Smiles", "Excipient_Smiles"] if c in table]).round(4).to_string())
    if has_label:
        print()
        print(pd.Series(metric_suite(probs, labels, threshold)).to_string())
    print(f"\nthreshold {threshold:.3f} | saved {os.path.abspath(out)}")


# --------------------------------------------------------------------------- train
def train(seed, device_str=None, overwrite=False):
    from src.train import pgb_train

    ckpt_dir = f"checkpoints/retrain_seed_{seed}"
    if os.path.normpath(ckpt_dir) == os.path.normpath(FINAL_CKPT):
        raise SystemExit("refusing to overwrite the shipped model")
    if os.path.exists(os.path.join(ckpt_dir, "best_model.pt")) and not overwrite:
        raise SystemExit(f"{ckpt_dir} already exists; pass --overwrite to retrain")
    os.makedirs(ckpt_dir, exist_ok=True)
    t0 = time.time()

    # The training config is the one stored in the shipped checkpoint; only seed, paths and device change.
    config = copy.deepcopy(torch.load(os.path.join(FINAL_CKPT, "best_model.pt"), map_location="cpu", weights_only=False)["config"])
    config.seed = seed
    config.device = device_str or "auto"

    # ---- same order of operations as the original run ----
    seed_everything(config.seed)
    device = get_device(config.device)
    data = {s: pd.read_csv(p) for s, p in DATA.items()}

    config.train_csv, config.val_csv, config.test_csv = DATA["train"], DATA["val"], DATA["test"]
    config.checkpoint_dir, config.metrics_dir = ckpt_dir, os.path.join(OUTPUTS, os.path.basename(ckpt_dir))
    config.positive_prior = float(data["train"]["Outcome1"].mean())
    config.use_balanced_sampler = False
    config.loss = "asym_focal"

    encoder = build_encoder(config, device)

    train_df = data["train"]
    all_train_smiles = list(set(train_df["API_Smiles"].dropna().tolist() + train_df["Excipient_Smiles"].dropna().tolist()))
    pgb_norm = PGBNormStats.from_smiles_list(all_train_smiles)
    pgb_norm.save(os.path.join(ckpt_dir, "pgb_norm_stats.npz"))

    prior_table = ExcipientPriorTable()
    prior_table.fit(train_df, leave_out_cluster_id=None)
    prior_table.save(os.path.join(ckpt_dir, "prior_table.json"))
    train_priors = build_leave_cluster_out_prior_vectors(train_df, cluster_col="cluster_id")

    train_loader = make_loader(DATA["train"], config, prior_table, pgb_norm, encoder, row_priors=train_priors, shuffle=True)
    val_loader = make_loader(DATA["val"], config, prior_table, pgb_norm, encoder)

    model = pgb_model_from_config(config, encoder)
    model.to(device)
    print(f"[morgan-PGB | seed {seed} | {device}] train {len(train_df)} | val {len(data['val'])} | test {len(data['test'])}")
    model, history = pgb_train(model, train_loader, val_loader, config, device)

    val_probs, val_labels = collect_predictions(model, val_loader, device)
    threshold, _ = tune_threshold(val_probs, val_labels, config.threshold_step, tnr_floor=config.pgb_tnr_floor)
    with open(os.path.join(ckpt_dir, "threshold.json"), "w") as f:
        json.dump({"threshold": float(threshold), "rule": f"best val MCC with TNR >= {config.pgb_tnr_floor}"}, f, indent=2)

    out_dir = os.path.join(OUTPUTS, os.path.basename(ckpt_dir))
    os.makedirs(out_dir, exist_ok=True)
    save_metrics({"history": history}, os.path.join(out_dir, "training_history.json"))
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump({
            "model": "morgan-PGB", "seed": seed, "device": str(device),
            "split": {s: {"path": p, "rows": len(data[s]), "md5": md5(p)} for s, p in DATA.items()},
            "heldout": {"path": HELDOUT_CSV, "md5": md5(HELDOUT_CSV)},
            "epochs_trained": len(history["val_pr_auc"]),
            "best_epoch": int(pd.Series(history["val_pr_auc"]).idxmax()) + 1,
            "threshold_from_val": float(threshold), "tnr_floor": config.pgb_tnr_floor,
            "finished": datetime.now().isoformat(timespec="seconds"), "minutes": round((time.time() - t0) / 60, 2),
        }, f, indent=2)

    print()
    evaluate(ckpt_dir, device_str)


# --------------------------------------------------------------------------- split
def parent_key(smiles):
    """Largest fragment, stereo removed, canonical SMILES (salt / hydrate / stereo blind)."""
    from rdkit import Chem
    smiles = "" if pd.isna(smiles) else str(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    frag = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumHeavyAtoms())
    Chem.RemoveStereochemistry(frag)
    return Chem.MolToSmiles(frag)


def butina_clusters(smiles_list, sim_cutoff=0.85):
    """Butina clustering of API SMILES on Morgan fingerprints (radius 2, 2048 bits)."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]
    dists = []
    for i in range(1, len(fps)):
        dists.extend(1.0 - s for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    clusters = Butina.ClusterData(dists, len(fps), 1.0 - sim_cutoff, isDistData=True)
    return {smiles_list[m]: cid for cid, members in enumerate(clusters) for m in members}


def rebuild_split():
    """Build train / val / test from data/start_dataset.csv, write a split log, and check the result
    is byte-identical to data/{train,val,test}.csv.

    1. Butina-cluster the unique API SMILES (Morgan r=2, 2048 bits, Tanimoto >= 0.85) -> cluster_id
    2. val  : GroupShuffleSplit(test_size=0.2, random_state=42) over API clusters
    3. test : StratifiedGroupKFold(4, shuffle=True, random_state=42) over the remaining clusters,
              stratified on Outcome1; the first fold is test, the other three are train
    No API cluster is in two splits.
    """
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    out_dir = os.path.join(OUTPUTS, "rebuilt_split")
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    def header(title):
        log("=" * 78)
        log(title)
        log("=" * 78)

    raw = pd.read_csv(START_CSV)
    n, pos = len(raw), int(raw["Outcome1"].sum())
    header("FINAL MODEL (Morgan-PGB) - DATA SPLIT LOG")
    log(f"Source dataset        : {START_CSV}  (md5 {md5(START_CSV)})")
    log(f"Built                 : {datetime.now().isoformat(timespec='seconds')} by `python run.py split`")
    log("Split type            : API-cluster grouped, label-stratified, ~60 / 20 / 20 (train / val / test)")
    log("Random seed           : 42")
    log("External test set     : data/held_out_test_set.csv (24 pairs, scored only, not part of the split)")
    log()

    header("STEP 1: RAW DATA (start_dataset.csv)")
    log(f"Total pairs            : {n}")
    log(f"Compatible (0)         : {n - pos} ({(n - pos) / n:.1%})")
    log(f"Incompatible (1)       : {pos} ({pos / n:.1%})")
    log(f"Unique API_CID         : {raw['API_CID'].nunique()}")
    log(f"Unique API SMILES      : {raw['API_Smiles'].nunique()}")
    log(f"Unique Excipient_CID   : {raw['Excipient_CID'].nunique()}")
    log()

    header("STEP 2: NEAR-DUPLICATE API CLUSTERING (Butina, Morgan r=2 2048 bits, Tanimoto >= 0.85)")
    apis = raw["API_Smiles"].unique().tolist()
    smi2c = butina_clusters(apis)
    raw["cluster_id"] = raw["API_Smiles"].map(smi2c)
    sizes = pd.Series(smi2c).value_counts()
    merged = sizes[sizes > 1]
    log(f"Unique API SMILES clustered      : {len(apis)}")
    log(f"Butina clusters                  : {sizes.size}")
    log(f"Clusters with > 1 API            : {merged.size} (largest: {int(sizes.max())} APIs)")
    log(f"APIs in a multi-API cluster      : {int(merged.sum())} ({merged.sum() / len(apis):.1%} of all APIs)")
    log()
    log("-> cluster_id is the grouping key: near-duplicate APIs (salts, stereoisomers, close analogues)")
    log("   always land in the same split.")
    log()

    header("STEP 3: GROUPED STRATIFIED SPLIT (~60 / 20 / 20)")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    rest_idx, val_idx = next(gss.split(raw, y=raw["Outcome1"], groups=raw["cluster_id"]))
    rest = raw.iloc[rest_idx].reset_index(drop=True)
    val = raw.iloc[val_idx].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(rest, y=rest["Outcome1"], groups=rest["cluster_id"]))
    splits = {"train": rest.iloc[train_idx].reset_index(drop=True), "val": val,
              "test": rest.iloc[test_idx].reset_index(drop=True)}
    log("(a) VAL  : GroupShuffleSplit(test_size=0.2, random_state=42) draws 20% of the API clusters")
    log(f"           -> {val['cluster_id'].nunique()} clusters, {len(val)} rows")
    log("(b) TEST : the remaining clusters go into StratifiedGroupKFold(n_splits=4, shuffle=True,")
    log("           random_state=42), stratified on Outcome1; fold 1 is TEST, folds 2-4 are TRAIN")
    log(f"           -> test {splits['test']['cluster_id'].nunique()} clusters / {len(splits['test'])} rows, "
        f"train {splits['train']['cluster_id'].nunique()} clusters / {len(splits['train'])} rows")
    log()

    header("STEP 4: LEAKAGE VERIFICATION")
    names = list(splits)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    leak = False
    for a, b in pairs:
        g = len(set(splits[a]["cluster_id"]) & set(splits[b]["cluster_id"]))
        leak |= g > 0
        log(f"API cluster overlap {a + '<->' + b:12s}: {g}")
    log()
    for a, b in pairs:
        c = len(set(splits[a]["API_CID"]) & set(splits[b]["API_CID"]))
        s = len(set(splits[a]["API_Smiles"]) & set(splits[b]["API_Smiles"]))
        leak |= (c + s) > 0
        log(f"API_CID / API SMILES overlap {a + '<->' + b:12s}: {c} / {s}")
    log()
    log("LEAKAGE FOUND - see counts above" if leak else
        "No leakage: every API (and every near-duplicate API cluster) is in exactly one split.")
    log()
    tr_exc = set(splits["train"]["Excipient_CID"])
    for s in ["val", "test"]:
        e = set(splits[s]["Excipient_CID"])
        log(f"Excipient_CID overlap train<->{s:4s} (expected, NOT leakage): {len(e & tr_exc)} / {len(e)} {s} excipients also in train")
    log()

    header("STEP 5: OVERLAP WITH THE 24 HELD-OUT PAIRS (same CIDs, or same parent molecules)")
    ho = pd.read_csv(HELDOUT_CSV)
    ho["api_key"], ho["exc_key"] = ho["API_Smiles"].map(parent_key), ho["Excipient_Smiles"].map(parent_key)
    found = 0
    for name, d in splits.items():
        for _, r in d.iterrows():
            ak, ek = parent_key(r["API_Smiles"]), parent_key(r["Excipient_Smiles"])
            hit = ho[((ho["API_CID"] == r["API_CID"]) & (ho["Excipient_CID"] == r["Excipient_CID"]))
                     | ((ho["api_key"] == ak) & (ho["exc_key"] == ek))]
            if len(hit):
                found += 1
                log(f"{name:5s}: {hit.iloc[0]['Api_name']} + {hit.iloc[0]['Excipient_name']} "
                    f"(API_CID {r['API_CID']}, Excipient_CID {r['Excipient_CID']}, label {r['Outcome1']})")
    log(f"-> {found} of the 24 held-out pairs also occur in start_dataset; they were kept in their split."
        if found else "None.")
    log()

    header("STEP 6: SPLIT SUMMARY")
    for name, d in splits.items():
        p = int(d["Outcome1"].sum())
        log(f"{name.upper():5s} | rows={len(d):5d} ({len(d) / n:5.1%}) | pos={p:4d} ({p / len(d):5.1%}) | "
            f"neg={len(d) - p:5d} ({1 - p / len(d):5.1%}) | API clusters={d['cluster_id'].nunique():4d} | "
            f"unique APIs={d['API_CID'].nunique():4d} | unique excipients={d['Excipient_CID'].nunique():4d}")
    log(f"TOTAL | rows={sum(len(d) for d in splits.values()):5d} (every start_dataset row is used exactly once)")
    log()

    header("STEP 7: FILES WRITTEN")
    ok = True
    for name, d in splits.items():
        path = os.path.join(out_dir, f"{name}.csv")
        d.to_csv(path, index=False)
        same = md5(path) == md5(DATA[name])
        ok &= same
        log(f"{name + '.csv':9s} : {len(d):5d} rows | md5 {md5(path)} | identical to {DATA[name]}: {same}")
    log("columns   : " + ", ".join(splits["train"].columns))
    log("grouping key used: cluster_id (Butina API cluster)")
    log("=" * 78)

    with open(os.path.join(out_dir, "split_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nRebuilt split and split_log.txt in {os.path.join(HERE, out_dir)} -> "
          + ("matches data/ exactly" if ok else "DIFFERS from data/"))


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Final Morgan-PGB model (60/20/20 split, 19/24 held-out).")
    sub = p.add_subparsers(dest="cmd")

    e = sub.add_parser("evaluate", help="metrics on val / test / held-out + held-out predictions (default)")
    e.add_argument("--checkpoint", default=FINAL_CKPT)
    e.add_argument("--device", default=None, help="cpu, cuda (default: cuda if available)")

    pr = sub.add_parser("predict", help="score a CSV of pairs (API_Smiles, Excipient_Smiles; optional Outcome1/ground_truth)")
    pr.add_argument("--csv", required=True)
    pr.add_argument("--checkpoint", default=FINAL_CKPT)
    pr.add_argument("--out", default=None)
    pr.add_argument("--device", default=None)

    t = sub.add_parser("train", help="retrain from scratch into checkpoints/retrain_seed_<seed>/")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default=None)
    t.add_argument("--overwrite", action="store_true")

    sub.add_parser("split", help="rebuild train/val/test from data/start_dataset.csv, write the split log, check it matches data/")

    args = p.parse_args()
    # make --checkpoint / --csv paths given relative to the caller's directory work after chdir
    for attr in ("checkpoint", "csv", "out"):
        v = getattr(args, attr, None)
        if v and not os.path.isabs(v) and os.path.exists(os.path.join(CALLER_CWD, v)) and not os.path.exists(v):
            setattr(args, attr, os.path.join(CALLER_CWD, v))
    if args.cmd == "predict" and args.out and not os.path.isabs(args.out):
        args.out = os.path.join(CALLER_CWD, args.out)

    if args.cmd in (None, "evaluate"):
        evaluate(getattr(args, "checkpoint", FINAL_CKPT), getattr(args, "device", None))
    elif args.cmd == "predict":
        predict(args.csv, args.checkpoint, args.device, args.out)
    elif args.cmd == "train":
        train(args.seed, args.device, args.overwrite)
    elif args.cmd == "split":
        rebuild_split()


if __name__ == "__main__":
    main()
