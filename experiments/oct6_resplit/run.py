"""
Retrain the Oct-6 PGB models on data/oct6_resplit (train ~60% / Oct-6 val / test ~20%) and
evaluate on val, test and the 24 held-out pairs.

Recipe: identical to the Oct-6 80/20 run (src/single_run.py at commit f534ef7). The training
config is copied from each family's Oct-6 checkpoint
(checkpoints/pgb_*/single_run_80_20split/best_model.pt); only data paths, output paths and seed change.
  - PGB norm stats and excipient prior from the train split only
  - leave-own-cluster-out priors for train rows (cluster_id column from the Oct-6 split)
  - early stopping and LR schedule on val; threshold on val (best MCC with TNR >= pgb_tnr_floor)
  - test and held-out are only scored

Nothing outside data/oct6_resplit, checkpoints/oct6_resplit and metrics/oct6_resplit is read for
writing; existing models and results are not touched.

Outputs per family and seed (finished runs are skipped unless --overwrite):
  checkpoints/oct6_resplit/<family>/seed_<s>/best_model.pt, prior_table.json, pgb_norm_stats.npz
  metrics/oct6_resplit/<family>/seed_<s>/
      val_metrics.json, test_metrics.json, heldout_metrics.json, training_history.json,
      predictions_{val,test,heldout}.csv, run_info.json
  metrics/oct6_resplit/summary.csv

Usage (from the project root):
  python -m experiments.oct6_resplit.make_split                # once
  python -m experiments.oct6_resplit.run --families morgan
  python -m experiments.oct6_resplit.run --families morgan --seeds 42 43 44
  python -m experiments.oct6_resplit.run --families all
  python -m experiments.oct6_resplit.run --summary
"""

import argparse
import copy
import json
import os
import time
import traceback
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.split_8020.run import _git_commit, _md5, metric_suite
from experiments.split_8020.vectors import assert_coverage, fill_heldout_vectors
from src.evaluate import collect_predictions, save_metrics, tune_threshold
from src.utils import seed_everything, get_device

SPLIT_DIR = "data/oct6_resplit"
CSV = {s: f"{SPLIT_DIR}/{s}.csv" for s in ["train", "val", "test"]}
HELDOUT_CSV = "held_out_testset/held_out_test_set.csv"
CKPT_ROOT = "checkpoints/oct6_resplit"
METRICS_ROOT = "metrics/oct6_resplit"

OCT6_FOLDERS = {
    "maccs": "pgb_fixed_vector_maccs_concat_asl", "morgan": "pgb_fixed_vector_morgan_concat_asl",
    "pubchemfp": "pgb_fixed_vector_pubchemfp_concat_asl", "mol2vec": "pgb_fixed_vector_mol2vec_concat_asl",
    "molformer": "pgb_molformer_cross_attn_global_gated_asl", "chemberta": "pgb_chemberta_cross_attn_cls_asl",
    "gin": "pgb_pretrained_gin_cross_attn_global_gated_asl", "gat": "pgb_pretrained_gat_cross_attn_global_gated_asl",
    "dmpnn": "pgb_dmpnn_chemprop_cross_attn_cls_asl",
}


def oct6_config(family):
    path = f"checkpoints/{OCT6_FOLDERS[family]}/single_run_80_20split/best_model.pt"
    return copy.deepcopy(torch.load(path, map_location="cpu", weights_only=False)["config"]), path


def train_family(family, seed, device_str):
    from src.cross_validate import build_encoder
    from src.dataset import PGBDataset, pgb_collate_fn, PGBNormStats
    from src.model import pgb_model_from_config
    from src.pgb_prior import ExcipientPriorTable, build_leave_cluster_out_prior_vectors
    from src.train import pgb_train

    t0 = time.time()
    ckpt_dir = os.path.join(CKPT_ROOT, family, f"seed_{seed}")
    metrics_dir = os.path.join(METRICS_ROOT, family, f"seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    config, source_ckpt = oct6_config(family)
    config.seed = seed
    if device_str:
        config.device = device_str

    # ---- same order of operations as pgb_single_run at f534ef7 ----
    seed_everything(config.seed)
    device = get_device(config.device)
    data = {s: pd.read_csv(p) for s, p in CSV.items()}

    run_config = copy.deepcopy(config)
    run_config.train_csv, run_config.val_csv, run_config.test_csv = CSV["train"], CSV["val"], CSV["test"]
    run_config.checkpoint_dir, run_config.metrics_dir = ckpt_dir, metrics_dir
    run_config.positive_prior = float(data["train"]["Outcome1"].mean())
    run_config.use_balanced_sampler = False
    run_config.loss = "asym_focal"

    encoder = build_encoder(run_config, device)
    if not encoder.is_sequence_capable:
        run_config.fusion = "concat"
    fv_enc = encoder if run_config.encoder == "fixed_vector" else None
    fvs = getattr(run_config, "fixed_vector_source", "maccs")
    if fv_enc is not None and fvs in ("pubchemfp", "mol2vec"):
        assert_coverage(fv_enc, pd.concat([d[c] for d in data.values() for c in ("API_Smiles", "Excipient_Smiles")]).dropna().astype(str).tolist(),
                        "train/val/test")

    train_df = data["train"]
    all_train_smiles = list(set(train_df["API_Smiles"].dropna().tolist() + train_df["Excipient_Smiles"].dropna().tolist()))
    pgb_norm = PGBNormStats.from_smiles_list(all_train_smiles)
    pgb_norm.save(os.path.join(ckpt_dir, "pgb_norm_stats.npz"))

    prior_table = ExcipientPriorTable()
    prior_table.fit(train_df, leave_out_cluster_id=None)
    prior_table.save(os.path.join(ckpt_dir, "prior_table.json"))
    train_priors = build_leave_cluster_out_prior_vectors(train_df, cluster_col="cluster_id")

    def loader(csv, row_priors=None, shuffle=False):
        ds = PGBDataset(csv, run_config.encoder, fvs, prior_table, pgb_norm, fixed_vector_encoder=fv_enc, row_priors=row_priors)
        return DataLoader(ds, batch_size=run_config.pgb_batch_size, shuffle=shuffle, collate_fn=pgb_collate_fn, num_workers=0)

    train_loader = loader(CSV["train"], row_priors=train_priors, shuffle=True)
    val_loader, test_loader = loader(CSV["val"]), loader(CSV["test"])

    model = pgb_model_from_config(run_config, encoder)
    model.to(device)
    print(f"[{family} | seed {seed}] train {len(train_df)} | val {len(data['val'])} | test {len(data['test'])}")

    model, history = pgb_train(model, train_loader, val_loader, run_config, device)

    # threshold on val, exactly as evaluate_model() does for PGB
    val_probs, val_labels = collect_predictions(model, val_loader, device)
    threshold, _ = tune_threshold(val_probs, val_labels, run_config.threshold_step, tnr_floor=run_config.pgb_tnr_floor)
    results = {"val": (val_probs, val_labels, data["val"])}
    test_probs, test_labels = collect_predictions(model, test_loader, device)
    results["test"] = (test_probs, test_labels, data["test"])

    ho = pd.read_csv(HELDOUT_CSV).rename(columns={"ground_truth": "Outcome1"})
    vec_report = fill_heldout_vectors(fv_enc, ho) if fv_enc is not None and fvs in ("pubchemfp", "mol2vec") else None
    tmp = os.path.join(ckpt_dir, "_heldout_tmp.csv")
    ho.to_csv(tmp, index=False)
    try:
        ho_probs, ho_labels = collect_predictions(model, loader(tmp), device)
    finally:
        os.remove(tmp)
    results["heldout"] = (ho_probs, ho_labels, ho)

    metrics = {}
    for split, (probs, labels, df) in results.items():
        metrics[split] = metric_suite(probs, labels, threshold)
        save_metrics(metrics[split], os.path.join(metrics_dir, f"{split}_metrics.json"))
        cols = [c for c in ["Api_name", "Excipient_name", "API_CID", "Excipient_CID", "API_Smiles", "Excipient_Smiles"] if c in df]
        out = df[cols].copy()
        out["label"], out["probability"] = labels.astype(int), probs
        out["prediction"] = (probs >= threshold).astype(int)
        out.to_csv(os.path.join(metrics_dir, f"predictions_{split}.csv"), index=False)
    save_metrics({"history": history}, os.path.join(metrics_dir, "training_history.json"))

    with open(os.path.join(metrics_dir, "run_info.json"), "w") as f:
        json.dump({
            "family": family, "seed": seed, "device": str(device), "config_copied_from": source_ckpt,
            "split": {s: {"path": p, "rows": len(data[s]), "md5": _md5(p)} for s, p in CSV.items()},
            "heldout": {"path": HELDOUT_CSV, "md5": _md5(HELDOUT_CSV)},
            "git_commit": _git_commit(),
            "epochs_trained": len(history["val_pr_auc"]),
            "best_epoch": int(pd.Series(history["val_pr_auc"]).idxmax()) + 1,
            "threshold_from_val": float(threshold), "tnr_floor": run_config.pgb_tnr_floor,
            "heldout_fixed_vectors": vec_report,
            "finished": datetime.now().isoformat(timespec="seconds"), "minutes": round((time.time() - t0) / 60, 2),
        }, f, indent=2, default=str)

    del model, encoder
    return metrics, threshold


def summarize():
    rows = []
    if os.path.isdir(METRICS_ROOT):
        for fam in sorted(os.listdir(METRICS_ROOT)):
            fdir = os.path.join(METRICS_ROOT, fam)
            if not os.path.isdir(fdir):
                continue
            for sd in sorted(os.listdir(fdir)):
                d = os.path.join(fdir, sd)
                if not os.path.exists(os.path.join(d, "heldout_metrics.json")):
                    continue
                m = {s: json.load(open(os.path.join(d, f"{s}_metrics.json"))) for s in ["val", "test", "heldout"]}
                rows.append({"family": fam, "seed": sd.replace("seed_", ""), "threshold": m["val"]["threshold"],
                             "heldout_correct": m["heldout"]["correct"], "heldout_TP": m["heldout"]["tp"], "heldout_TN": m["heldout"]["tn"],
                             "heldout_pr_auc": m["heldout"]["pr_auc"], "heldout_roc_auc": m["heldout"]["roc_auc"],
                             "val_pr_auc": m["val"]["pr_auc"], "val_mcc": m["val"]["mcc"], "val_f1": m["val"]["f1"],
                             "test_pr_auc": m["test"]["pr_auc"], "test_roc_auc": m["test"]["roc_auc"],
                             "test_mcc": m["test"]["mcc"], "test_f1": m["test"]["f1"],
                             "test_cm": m["test"]["confusion_matrix"], "heldout_cm": m["heldout"]["confusion_matrix"]})
    if not rows:
        print(f"No finished runs under {METRICS_ROOT}/")
        return
    df = pd.DataFrame(rows).sort_values(["heldout_correct", "val_pr_auc"], ascending=False)
    df.to_csv(os.path.join(METRICS_ROOT, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
    print(f"\nSaved {METRICS_ROOT}/summary.csv")


def main():
    p = argparse.ArgumentParser(description="Retrain Oct-6 PGB models on train/val/test from data/oct6_resplit.")
    p.add_argument("--families", nargs="+", default=None, help=f"{list(OCT6_FOLDERS)} or all")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()

    if args.summary:
        summarize()
        return
    if not args.families:
        raise SystemExit("Give --families (e.g. morgan, or all) or --summary.")
    for s, path in CSV.items():
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found. Build the split first:  python -m experiments.oct6_resplit.make_split")
    families = list(OCT6_FOLDERS) if args.families == ["all"] else args.families
    unknown = [f for f in families if f not in OCT6_FOLDERS]
    if unknown:
        raise SystemExit(f"Unknown families {unknown}; choose from {list(OCT6_FOLDERS)}")

    failures = []
    for seed in args.seeds:
        for fam in families:
            if os.path.exists(os.path.join(METRICS_ROOT, fam, f"seed_{seed}", "heldout_metrics.json")) and not args.overwrite:
                print(f"skip {fam} seed {seed} (done; --overwrite to retrain)")
                continue
            print(f"\n{'=' * 70}\n{fam} | seed {seed}\n{'=' * 70}")
            try:
                m, thr = train_family(fam, seed, args.device)
                print(f"[{fam} | seed {seed}] threshold {thr:.3f} | val PR-AUC {m['val']['pr_auc']:.4f} | "
                      f"test PR-AUC {m['test']['pr_auc']:.4f} MCC {m['test']['mcc']:.4f} | held-out {m['heldout']['correct']}/24")
            except Exception as e:
                failures.append((fam, seed, repr(e)))
                os.makedirs(METRICS_ROOT, exist_ok=True)
                with open(os.path.join(METRICS_ROOT, "failures.log"), "a") as f:
                    f.write(f"{datetime.now().isoformat(timespec='seconds')} {fam} seed {seed}\n{traceback.format_exc()}\n")
                print(f"FAILED {fam} seed {seed}: {e!r}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    for f in failures:
        print("FAILED:", *f)
    summarize()


if __name__ == "__main__":
    main()
