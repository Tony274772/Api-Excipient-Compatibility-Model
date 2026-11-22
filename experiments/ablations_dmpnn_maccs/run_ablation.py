"""Train and evaluate ONE ablation of `dmpnn__x_maccs` at one seed.

The recipe is run_grid.run_cell, step for step and in the same order (seed -> config ->
representations -> encoder -> norm stats -> prior -> loaders -> model), so `full` here reproduces
the grid's dmpnn__x_maccs run at the same seed on the same device. Only the ablation hooks
differ:

    build settings (variants.BUILD_SETTINGS)  applied before data and model are made
    apply_ablation(model, name)               cuts parts out of the built full model

Outputs (a run is finished only when COMPLETE exists; it is written last):

    experiments/results/ablations_dmpnn_maccs/runs/<ablation>_seed<seed>/
        val_metrics.json  run_info.json  training_history.json  predictions_val.csv  COMPLETE
    experiments/results/ablations_dmpnn_maccs/_heldout_sealed/<ablation>_seed<seed>/
        heldout_metrics.json  predictions_heldout.csv            (not used for any decision)
    checkpoints/ablations_dmpnn_maccs/<ablation>_seed<seed>/
        best_model.pt (trainable part)  prior_table.json  pgb_norm_stats.npz

Usage (from the project root):
    python -m experiments.ablations_dmpnn_maccs.run_ablation --ablation no_w --seed 42
    python -m experiments.ablations_dmpnn_maccs.run_ablation --ablation prior1+no_w --seed 43
    python -m experiments.ablations_dmpnn_maccs.run_ablation --list          # variants + parameters
    python -m experiments.ablations_dmpnn_maccs.run_ablation --ablation no_w --smoke   # 1 epoch, _smoke/
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
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    RDLogger = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.ablations_dmpnn_maccs import variants as V  # noqa: E402
from experiments.ablations_dmpnn_maccs.prior_lookup import leave_cluster_out_priors, table_class  # noqa: E402
from experiments.encoder_comparison.data import (GridPGBDataset, SplitPaths,  # noqa: E402
                                                 build_cell_representations,
                                                 check_heldout_is_external, sanity_check_batch)
from experiments.encoder_comparison.families import Cell  # noqa: E402
from experiments.encoder_comparison.run_grid import (build_encoder, build_model,  # noqa: E402
                                                     make_config, metric_suite, prediction_table)
from src.dataset import PGBNormStats, pgb_collate_fn  # noqa: E402
from src.evaluate import collect_predictions, tune_threshold  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

MODEL_CELL = "dmpnn__x_maccs"
PRIMARY, SECONDARY = "dmpnn", "maccs"
SEEDS = [42, 43, 44, 45, 46]

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "results", "ablations_dmpnn_maccs")
RUNS_ROOT = os.path.join(RESULTS_ROOT, "runs")
SEALED_ROOT = os.path.join(RESULTS_ROOT, "_heldout_sealed")
SMOKE_ROOT = os.path.join(RESULTS_ROOT, "_smoke")
LOOKUP_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "encoder_grid", "_lookup")
CKPT_ROOT = os.path.join(PROJECT_ROOT, "checkpoints", "ablations_dmpnn_maccs")
SENTINEL = "COMPLETE"


def run_name(ablation: str, seed: int) -> str:
    return f"{ablation}_seed{seed}"


def is_complete(ablation: str, seed: int) -> bool:
    return os.path.exists(os.path.join(RUNS_ROOT, run_name(ablation, seed), SENTINEL))


def cell_for(settings: dict) -> Cell:
    """The grid cell this run is built from: dmpnn x maccs, with pooling / secondary
    changed where the ablation says so."""
    return Cell(PRIMARY, settings.get("secondary", SECONDARY), pooling=settings.get("pooling"),
                block="fp_secondary")


def configure(config, settings: dict):
    """Build settings that live on the config (read before the model is built)."""
    config.ablation_prior_lookup = settings.get("prior_lookup", "full")
    config.mech_set = settings.get("mech_set", "v1")
    if config.mech_set == "v2":
        from experiments.ablations_dmpnn_maccs.mech_v2 import NUM_MECH_V2
        config.pgb_num_mechanisms = NUM_MECH_V2      # sizes the head's gate and MLP inputs
    return config


def dataset_class(config):
    if getattr(config, "mech_set", "v1") == "v2":
        from experiments.ablations_dmpnn_maccs.mech_v2 import MechV2GridDataset
        return MechV2GridDataset
    return GridPGBDataset


def make_loader(csv, config, prior_table, pgb_norm, primary_rep, secondary_rep, row_priors=None,
                shuffle=False):
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
        from experiments.ablations_dmpnn_maccs.parent_smiles import (ParentSmilesEncoder,
                                                                    parent_change_report)
        encoder = ParentSmilesEncoder(encoder).to(device)
        parent_report = parent_change_report(paths.all_molecules())
        print(f"[{name}] D-MPNN reads the parent molecule for {parent_report['changed']} of "
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
        "parameters_total": n_total, "parameters_trainable": n_train,
        "parameters_full_trainable": full_trainable,
        "parameters_cut": full_trainable - n_train,
        "epochs_trained": len(history), "best_epoch": int(val_metrics.get("best_epoch", 0)),
        "max_epochs": config.pgb_max_epochs,
        "threshold_from_val": threshold, "tnr_floor": config.pgb_tnr_floor,
        "val_pr_auc": val_metrics["pr_auc"], "val_mcc": val_metrics["mcc"],
        "head_layout": head_switches(model),
        "api_tower_layout": tower_layout(model.api_tower),
        "exc_tower_layout": tower_layout(model.exc_tower),
        "parent_report": parent_report,
        "data": {"train": paths.train, "val": paths.val, "heldout": paths.heldout,
                 "rows": {"train": len(train_df), "val": len(val_df), "heldout": len(ho_df)}},
        "minutes": minutes, "finished": datetime.utcnow().isoformat(timespec="seconds")
    }

    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    with open(os.path.join(out_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    val_preds_df = prediction_table(paths.val, val_probs, threshold)
    val_preds_df.to_csv(os.path.join(out_dir, "predictions_val.csv"), index=False)

    with open(os.path.join(sealed_dir, "heldout_metrics.json"), "w") as f:
        json.dump(ho_metrics, f, indent=2)
    ho_preds_df = prediction_table(paths.heldout, ho_probs, threshold)
    ho_preds_df.to_csv(os.path.join(sealed_dir, "predictions_heldout.csv"), index=False)

    trainable_state = {k: v.cpu() for k, v in model.state_dict().items()
                       if not k.startswith("encoder.")}
    torch.save(trainable_state, os.path.join(ckpt_dir, "best_model.pt"))

    with open(sentinel_path, "w") as f:
        f.write(f"COMPLETE {datetime.utcnow().isoformat(timespec='seconds')}\n"
                f"minutes={minutes}\nval_pr_auc={val_metrics['pr_auc']:.4f}\n"
                f"val_mcc={val_metrics['mcc']:.4f}\n")

    print(f"[{name}] DONE in {minutes} min | val PR-AUC {val_metrics['pr_auc']:.4f} | "
          f"val MCC {val_metrics['mcc']:.4f} | held-out correct {ho_metrics['correct']}/{len(ho_df)}")
    return val_metrics


# --------------------------------------------------------------------------- CLI
def list_variants():
    paths = SplitPaths(None, "api_cluster")
    dummy_cell = Cell(PRIMARY, SECONDARY, block="fp_secondary")
    cfg = make_config(dummy_cell, paths, 42, "cpu", 1)
    train_df = pd.read_csv(paths.train)
    norm = PGBNormStats.from_smiles_list(train_df["API_Smiles"].dropna().unique()[:10], with_salt=False)
    tbl = table_class("full")(global_rate=0.1)
    tbl.fit(train_df.head(20))
    ldr = make_loader(paths.val, cfg, tbl, norm, None, None)
    batch = next(iter(ldr))

    rows = []
    for a in V.PLAN:
        s = V.build_settings(a)
        cell = cell_for(s)
        c = make_config(cell, paths, 42, "cpu", 1)
        configure(c, s)
        from src.encoders.fixed_vector_encoder import IdentityEncoder
        enc = IdentityEncoder(128)
        model = build_model(c, cell, enc, None, None)
        full_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model = V.apply_ablation(model, a)
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rows.append({"ablation": a, "trainable_params": train_p, "delta_params": train_p - full_train,
                     "weight_paired": V.weight_paired(a), "description": V.describe(a)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ablation", default="full", help="ablation name or combination with +")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--data-root", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 epoch, write under _smoke/")
    p.add_argument("--list", action="store_true", help="print the planned variants with param counts")
    args = p.parse_args()

    if args.list:
        list_variants()
        return

    epochs = 1 if args.smoke else args.epochs
    run(args.ablation, args.seed, device_str=args.device, epochs=epochs,
        overwrite=args.overwrite, smoke=args.smoke, data_root=args.data_root)


if __name__ == "__main__":
    main()
