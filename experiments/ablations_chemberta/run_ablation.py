"""Train and evaluate ONE ablation of `chemberta__x_pubchemfp` at one seed.

The recipe is run_grid.run_cell, step for step and in the same order (seed -> config ->
representations -> encoder -> norm stats -> prior -> loaders -> model), so `full` here reproduces
the grid's chemberta__x_pubchemfp run at the same seed on the same device. Only the ablation hooks
differ:

    build settings (variants.BUILD_SETTINGS)  applied before data and model are made
    apply_ablation(model, name)               cuts parts out of the built full model

Outputs (a run is finished only when COMPLETE exists; it is written last):

    experiments/results/ablations_chemberta/runs/<ablation>_seed<seed>/
        val_metrics.json  run_info.json  training_history.json  predictions_val.csv  COMPLETE
    experiments/results/ablations_chemberta/_heldout_sealed/<ablation>_seed<seed>/
        heldout_metrics.json  predictions_heldout.csv            (not used for any decision)
    checkpoints/ablations_chemberta/<ablation>_seed<seed>/
        best_model.pt (trainable part)  prior_table.json  pgb_norm_stats.npz

Usage (from the project root):
    python -m experiments.ablations_chemberta.run_ablation --ablation no_w --seed 42
    python -m experiments.ablations_chemberta.run_ablation --ablation prior1+no_w --seed 43
    python -m experiments.ablations_chemberta.run_ablation --list          # variants + parameters
    python -m experiments.ablations_chemberta.run_ablation --ablation no_w --smoke   # 1 epoch, _smoke/
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime

import pandas as pd
import torch
from rdkit import RDLogger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.ablations_chemberta import variants as V  # noqa: E402
from experiments.ablations_chemberta.prior_lookup import leave_cluster_out_priors, table_class  # noqa: E402
from experiments.encoder_comparison.data import (GridPGBDataset, SplitPaths,  # noqa: E402
                                                 build_cell_representations,
                                                 check_heldout_is_external, sanity_check_batch)
from experiments.encoder_comparison.families import Cell  # noqa: E402
from experiments.encoder_comparison.run_grid import (build_encoder, build_model,  # noqa: E402
                                                     make_config, metric_suite, prediction_table)
from src.dataset import PGBNormStats, pgb_collate_fn  # noqa: E402
from src.evaluate import collect_predictions, tune_threshold  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

RDLogger.DisableLog("rdApp.*")

MODEL_CELL = "chemberta__x_pubchemfp"
PRIMARY, SECONDARY = "chemberta", "pubchemfp"
SEEDS = [42, 43, 44, 45, 46]

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "results", "ablations_chemberta")
RUNS_ROOT = os.path.join(RESULTS_ROOT, "runs")
SEALED_ROOT = os.path.join(RESULTS_ROOT, "_heldout_sealed")
SMOKE_ROOT = os.path.join(RESULTS_ROOT, "_smoke")
LOOKUP_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "encoder_grid", "_lookup")
CKPT_ROOT = os.path.join(PROJECT_ROOT, "checkpoints", "ablations_chemberta")
SENTINEL = "COMPLETE"


def run_name(ablation: str, seed: int) -> str:
    return f"{ablation}_seed{seed}"


def is_complete(ablation: str, seed: int) -> bool:
    return os.path.exists(os.path.join(RUNS_ROOT, run_name(ablation, seed), SENTINEL))


def cell_for(settings: dict) -> Cell:
    """The grid cell this run is built from: chemberta x pubchemfp, with pooling / secondary
    changed where the ablation says so."""
    return Cell(PRIMARY, settings.get("secondary", SECONDARY), pooling=settings.get("pooling"),
                block="fp_secondary")


def configure(config, settings: dict):
    """Build settings that live on the config (read before the model is built)."""
    config.ablation_prior_lookup = settings.get("prior_lookup", "full")
    config.mech_set = settings.get("mech_set", "v1")
    if config.mech_set == "v2":
        from experiments.ablations_chemberta.mech_v2 import NUM_MECH_V2
        config.pgb_num_mechanisms = NUM_MECH_V2      # sizes the head's gate and MLP inputs
    return config


def dataset_class(config):
    if getattr(config, "mech_set", "v1") == "v2":
        from experiments.ablations_chemberta.mech_v2 import MechV2GridDataset
        return MechV2GridDataset
    return GridPGBDataset


def make_loader(csv, config, prior_table, pgb_norm, primary_rep, secondary_rep, row_priors=None,
                shuffle=False):
    """experiments.encoder_comparison.data.make_grid_loader with a choice of dataset class."""
    from torch.utils.data import DataLoader
    ds = dataset_class(config)(csv, prior_table, pgb_norm, primary_rep=primary_rep,
                               secondary_rep=secondary_rep, row_priors=row_priors)
    return DataLoader(ds, batch_size=config.pgb_batch_size, shuffle=shuffle,
                      collate_fn=pgb_collate_fn, num_workers=0)


def score(model, df, config, prior_table, pgb_norm, primary_rep, secondary_rep, device):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pairs.csv")
        df.to_csv(path, index=False)
        loader = make_loader(path, config, prior_table, pgb_norm, primary_rep, secondary_rep)
        return collect_predictions(model, loader, device)


def head_switches(model) -> dict:
    h = model.head
    if not isinstance(h, V.AblationPGBHead):
        return {"ablation_head": False}
    return {"ablation_head": True, "prior_cols": [V.PRIOR_NAMES[c] for c in h.prior_cols],
            "use_prior_trust": h.use_prior_trust, "use_gate": h.use_gate,
            "use_bilinear": h.use_bilinear, "mech_cols": [h.mech_names[c] for c in h.mech_cols],
            "api_flags": len(h.api_flag_cols), "exc_flags": len(h.exc_flag_cols)}


def tower_layout(tower) -> dict:
    out = {"mix_in": tower.mix[0].in_features, "has_struct": tower.has_struct,
           "has_secondary": tower.has_secondary, "use_salt_context": tower.use_salt_context}
    if isinstance(tower, V.AblationComparisonTower):
        out.update({"segments": tower.segments, "flags": len(tower.flag_cols),
                    "use_desc": tower.use_desc})
    return out


# --------------------------------------------------------------------------- one run
def run(ablation: str, seed: int, device_str: str = "auto", epochs: int = 40,
        overwrite: bool = False, smoke: bool = False, data_root: str = None) -> dict:
    V.parse_ablation(ablation)                       # fail fast on a typo
    settings = V.build_settings(ablation)
    name = run_name(ablation, seed)
    runs_root = os.path.join(SMOKE_ROOT, "runs") if smoke else RUNS_ROOT
    sealed_root = os.path.join(SMOKE_ROOT, "_heldout_sealed") if smoke else SEALED_ROOT
    ckpt_root = os.path.join(SMOKE_ROOT, "checkpoints") if smoke else CKPT_ROOT
    out_dir, sealed_dir = os.path.join(runs_root, name), os.path.join(sealed_root, name)
    ckpt_dir = os.path.join(ckpt_root, name)
    sentinel_path = os.path.join(out_dir, SENTINEL)
    if os.path.exists(sentinel_path) and not (overwrite or smoke):
        raise SystemExit(f"{out_dir} is already COMPLETE; pass --overwrite to rerun")
    if os.path.exists(sentinel_path):
        os.remove(sentinel_path)
    for d in (out_dir, sealed_dir, ckpt_dir, LOOKUP_DIR):
        os.makedirs(d, exist_ok=True)

    paths = SplitPaths(data_root, "api_cluster")
    check_heldout_is_external(paths)
    cell = cell_for(settings)

    # ---- identical order to run_grid.run_cell from here on
    t0 = time.time()
    seed_everything(seed)
    config = make_config(cell, paths, seed, device_str, epochs)
    configure(config, settings)
    config.ablation = ablation
    device = get_device(config.device)
    config.checkpoint_dir, config.metrics_dir = ckpt_dir, out_dir

    train_df, val_df = pd.read_csv(paths.train), pd.read_csv(paths.val)

    primary_rep, secondary_rep = build_cell_representations(cell, paths, LOOKUP_DIR)
    for rep, slot in ((primary_rep, "primary"), (secondary_rep, "secondary")):
        if rep is not None:
            rep.save_scaler(os.path.join(ckpt_dir, f"{slot}_scaler.npz"))

    encoder = build_encoder(config, cell, device)
    parent_report = None
    if settings.get("parent_smiles"):
        from experiments.ablations_chemberta.parent_smiles import (ParentSmilesEncoder,
                                                                   parent_change_report)
        encoder = ParentSmilesEncoder(encoder).to(device)
        parent_report = parent_change_report(paths.all_molecules())
        print(f"[{name}] ChemBERTa reads the parent molecule for {parent_report['changed']} of "
              f"{parent_report['molecules']} molecules")

    train_smiles = list(set(train_df["API_Smiles"].dropna())
                        | set(train_df["Excipient_Smiles"].dropna()))
    pgb_norm = PGBNormStats.from_smiles_list(train_smiles, with_salt=config.pgb_normalize_salt)
    pgb_norm.save(os.path.join(ckpt_dir, "pgb_norm_stats.npz"))

    lookup = config.ablation_prior_lookup
    prior_table = table_class(lookup)(global_rate=config.positive_prior)
    prior_table.fit(train_df, leave_out_cluster_id=None)
    prior_table.save(os.path.join(ckpt_dir, "prior_table.json"))
    train_priors = leave_cluster_out_priors(train_df, prior_table.global_rate, lookup)

    train_loader = make_loader(paths.train, config, prior_table, pgb_norm, primary_rep,
                               secondary_rep, row_priors=train_priors, shuffle=True)
    val_loader = make_loader(paths.val, config, prior_table, pgb_norm, primary_rep, secondary_rep)
    first = next(iter(val_loader))
    sanity_check_batch(first, cell, primary_rep, secondary_rep)
    if int(first["mech_flags"].shape[1]) != config.pgb_num_mechanisms:
        raise RuntimeError(f"batch has {first['mech_flags'].shape[1]} mechanism flags, head expects "
                           f"{config.pgb_num_mechanisms}")

    model = build_model(config, cell, encoder, primary_rep, secondary_rep)
    full_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = V.apply_ablation(model, ablation)
    model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # one forward pass on the first batch: a wrong cut fails here, not after 40 epochs
    model.eval()
    with torch.no_grad():
        out = model({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in first.items()})
    if out.shape[0] != first["mech_flags"].shape[0] or not torch.isfinite(out).all():
        raise RuntimeError(f"[{name}] forward check failed: output {tuple(out.shape)}")

    print(f"\n{'=' * 100}\n[{name}] {V.describe(ablation)}")
    print(f"[{name}] {cell.label()}")
    print(f"[{name}] seed {seed} | device {device} | build settings {settings or 'none'} | "
          f"weight-paired with full: {V.weight_paired(ablation)}")
    print(f"[{name}] trainable params {n_train:,} (full {full_trainable:,}, "
          f"{n_train - full_trainable:+,}) | max epochs {config.pgb_max_epochs}")
    print("=" * 100)

    from src.train import pgb_train
    model, history = pgb_train(model, train_loader, val_loader, config, device)

    val_probs, val_labels = collect_predictions(model, val_loader, device)
    threshold, _ = tune_threshold(val_probs, val_labels, config.threshold_step,
                                  tnr_floor=config.pgb_tnr_floor)
    threshold = float(threshold)
    val_metrics = metric_suite(val_probs, val_labels, threshold)
    ho_df = paths.frame("heldout")
    ho_probs, ho_labels = score(model, ho_df, config, prior_table, pgb_norm, primary_rep,
                                secondary_rep, device)
    ho_metrics = metric_suite(ho_probs, ho_labels, threshold)

    minutes = round((time.time() - t0) / 60, 2)
    run_info = {
        "model": MODEL_CELL, "ablation": ablation, "parts": V.parse_ablation(ablation),
        "description": V.describe(ablation), "build_settings": settings,
        "weight_paired_with_full": V.weight_paired(ablation),
        "reference": V.REFERENCE.get(ablation, "full"),
        "seed_run": name, "seed": seed, "device": str(device), "smoke": smoke,
        "secondary": cell.secondary, "pooling": cell.pooling,
        "secondary_shuffle_sha1": secondary_rep.shuffle_fingerprint() if secondary_rep else "",
        "prior_lookup": lookup, "prior_global_rate": prior_table.global_rate,
        "mech_set": config.mech_set, "n_mech": config.pgb_num_mechanisms,
        "parent_smiles": parent_report,
        "cross_attention": bool(model.use_cross_attn),
        "attention_layers": bool(getattr(model, "use_attention_layers", True)) and model.use_cross_attn,
        "head": head_switches(model),
        "api_tower": tower_layout(model.api_tower), "exc_tower": tower_layout(model.exc_tower),
        "parameters_total": n_total, "parameters_trainable": n_train,
        "parameters_trainable_full": full_trainable,
        "epochs_trained": len(history["val_pr_auc"]),
        "best_epoch": int(pd.Series(history["val_pr_auc"]).idxmax()) + 1,
        "max_epochs": config.pgb_max_epochs,
        "threshold_from_val": threshold, "tnr_floor": config.pgb_tnr_floor,
        "data": {"train": paths.train, "val": paths.val, "heldout": paths.heldout,
                 "rows": {"train": len(train_df), "val": len(val_df), "heldout": len(ho_df)}},
        "minutes": minutes, "finished": datetime.now().isoformat(timespec="seconds"),
    }

    with open(os.path.join(out_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    prediction_table(val_df, val_probs, val_labels, threshold).to_csv(
        os.path.join(out_dir, "predictions_val.csv"), index=False)
    with open(os.path.join(sealed_dir, "heldout_metrics.json"), "w") as f:
        json.dump(ho_metrics, f, indent=2)
    prediction_table(ho_df, ho_probs, ho_labels, threshold).to_csv(
        os.path.join(sealed_dir, "predictions_heldout.csv"), index=False)

    for d, files in ((out_dir, ["val_metrics.json", "run_info.json", "training_history.json",
                                "predictions_val.csv"]),
                     (sealed_dir, ["heldout_metrics.json", "predictions_heldout.csv"])):
        for f in files:
            if not os.path.exists(os.path.join(d, f)):
                raise RuntimeError(f"{name}: {f} was not written; refusing to mark COMPLETE")
    with open(sentinel_path, "w") as f:
        f.write(json.dumps({"run": name, "val_pr_auc": val_metrics["pr_auc"],
                            "finished": run_info["finished"]}) + "\n")

    print(f"\n[{name}] val PR-AUC {val_metrics['pr_auc']:.4f} | val MCC {val_metrics['mcc']:.4f} | "
          f"threshold {threshold:.3f} | {minutes} min")
    print(f"[{name}] results -> {out_dir}  (held-out sealed in {sealed_dir})")
    return val_metrics


# --------------------------------------------------------------------------- parameter table
class _StubSeqEncoder(torch.nn.Module):
    """Stands in for ChemBERTa when only parameter counts are needed (--list): same output width
    (384), sequence-capable, so the model builds the same projection / attention layers."""
    is_sequence_capable = True
    output_dim = 384


def parameter_table(names) -> pd.DataFrame:
    paths = SplitPaths(None, "api_cluster")
    rows = []
    for name in names:
        settings = V.build_settings(name)
        cell = cell_for(settings)
        config = configure(make_config(cell, paths, 42, "cpu", 1), settings)
        from experiments.encoder_comparison.reps import rep_dim

        class _Rep:                                   # only .dim is read by build_model
            dim = rep_dim(cell.secondary)
        model = build_model(config, cell, _StubSeqEncoder(), None, _Rep())
        full_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model = V.apply_ablation(model, name)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rows.append({"ablation": name, "trainable": n, "vs_full_build": n - full_n,
                     "weight_paired": V.weight_paired(name), "description": V.describe(name)})
    df = pd.DataFrame(rows)
    full_n = int(df.loc[df["ablation"] == "full", "trainable"].iloc[0]) if "full" in names else None
    if full_n is not None:
        df.insert(2, "vs_full", df["trainable"] - full_n)
    return df


def main():
    p = argparse.ArgumentParser(description="Train one ablation of chemberta__x_pubchemfp.")
    p.add_argument("--ablation", help="e.g. no_w, prior1+no_w, no_flags_metal+no_salt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--data-root", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, outputs under results/ablations_chemberta/_smoke (never collected)")
    p.add_argument("--list", action="store_true", help="every planned variant with its parameters")
    args = p.parse_args()

    if args.list:
        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 90)
        print(parameter_table(V.PLAN).to_string(index=False))
        return
    if not args.ablation:
        raise SystemExit("pass --ablation <name>, or --list")
    run(args.ablation, args.seed, args.device or "auto", 1 if args.smoke else args.epochs,
        args.overwrite, args.smoke, args.data_root)


if __name__ == "__main__":
    main()
