"""Inference script for Gated Bilinear models on the 24-pair held-out test set.

Usage:
    python -m experiments.gated_bilinear.inference_gb --family gin
    python -m experiments.gated_bilinear.inference_gb --all       # runs all trained checkpoints
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from experiments.gated_bilinear.config import GBConfig
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from experiments.gated_bilinear.dataset import build_gb_heldout_dataloader
from experiments.gated_bilinear.evaluate_gb import evaluate_heldout_gb
from experiments.gated_bilinear.features import compute_gb_descriptor_norm_stats
from experiments.gated_bilinear.model import GatedBilinearModel
from experiments.gated_bilinear.prior import ExcipientPriorTable

from src.encoders import ENCODER_REGISTRY
from src.utils import seed_everything, get_device

GNN_ENCODERS = {"dmpnn_chemprop", "dmpnn_scratch", "attentivefp", "gine", "gatv2", "pna", "pretrained_gat"}
ALL_FAMILIES = ["maccs", "morgan", "pubchem", "mol2vec", "molformer", "chemberta", "gin", "gat", "dmpnn"]

FAMILY_TO_ENCODER = {
    "maccs": "fixed_vector",
    "morgan": "fixed_vector",
    "pubchem": "fixed_vector",
    "mol2vec": "fixed_vector",
    "molformer": "molformer",
    "chemberta": "chemberta",
    "gin": "pretrained_gin",
    "gat": "pretrained_gat",
    "dmpnn": "dmpnn_chemprop",
}


def load_model_and_threshold(family: str, device, checkpoint_path: str = None):
    config = GBConfig()
    config.gb_family = family
    config.encoder = FAMILY_TO_ENCODER[family]
    config.resolve_checkpoint_paths()
    config.resolve_csv_paths()
    config.resolve_gb_paths()

    ckpt_file = checkpoint_path or os.path.join(config.gb_checkpoint_dir, "best_model.pt")
    if not os.path.isfile(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found for family '{family}' at: {ckpt_file}")

    encoder_cls = ENCODER_REGISTRY[config.encoder]
    if config.encoder == "molformer":
        encoder = encoder_cls(config.molformer_model_path, device=str(device))
    elif config.encoder == "pretrained_gin":
        encoder = encoder_cls(config.gin_pretrained_name, device=str(device))
    elif config.encoder == "chemberta":
        encoder = encoder_cls(config.chemberta_model_path, device=str(device))
    elif config.encoder == "fixed_vector":
        encoder = encoder_cls(source="mol2vec", device=str(device))
    elif config.encoder in GNN_ENCODERS:
        encoder = encoder_cls(config, device=str(device))
    else:
        raise ValueError(f"Unknown encoder: {config.encoder}")

    encoder.to(device)
    model = GatedBilinearModel(config, encoder, family)
    model.to(device)

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load threshold from val_metrics.json if available, else fallback to 0.5
    metrics_path = os.path.join(config.gb_metrics_dir, "val_metrics.json")
    threshold = 0.5
    if os.path.isfile(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            v_met = json.load(f)
            threshold = float(v_met.get("threshold", 0.5))

    return model, config, threshold


def run_inference_for_family(family: str, prior_table, desc_stats, device, args):
    print(f"\n--- Running Inference for: {family.upper()} ---")
    try:
        model, config, threshold = load_model_and_threshold(family, device, args.checkpoint)
    except FileNotFoundError as e:
        print(f"Skipping {family}: {e}")
        return None

    heldout_loader = build_gb_heldout_dataloader(config, prior_table, desc_stats, args.heldout_csv)
    model_name = args.model_name or family

    metrics, _ = evaluate_heldout_gb(
        model=model,
        heldout_loader=heldout_loader,
        device=device,
        threshold=threshold,
        model_name=model_name,
        output_csv_path=args.output or config.gb_heldout_predictions_csv,
        heldout_source_csv=args.heldout_csv or config.gb_heldout_csv,
    )

    print(f"Main threshold: {threshold:.4f}")
    print(f"PR-AUC:   {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"MCC:      {metrics['mcc']:.4f}")
    print(f"Recall:   {metrics['recall']:.4f}")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run Held-Out Inference on Gated Bilinear models")
    parser.add_argument("--family", choices=ALL_FAMILIES, default=None, help="Single family to evaluate")
    parser.add_argument("--all", action="store_true", help="Run across all trained checkpoints")
    parser.add_argument("--model_name", type=str, default=None, help="Custom column prefix")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pt")
    parser.add_argument("--heldout_csv", type=str, default="held_out_testset/held_out_test_set.csv")
    parser.add_argument("--output", type=str, default="held_out_testset/held_out_predictions_gatedbilinear.csv")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)

    # Prepare shared prior table and stats from training set
    cfg_base = GBConfig()
    cfg_base.resolve_csv_paths()
    train_df = pd.read_csv(cfg_base.train_csv)
    train_exc = train_df["Excipient_Smiles"].fillna("").astype(str).tolist()
    train_api = train_df["API_Smiles"].fillna("").astype(str).tolist()
    train_labels = train_df["Outcome1"].astype(int).tolist()
    cluster_col = next((c for c in ["API_Cluster", "api_cluster", "Cluster"] if c in train_df.columns), None)
    train_clusters = train_df[cluster_col].tolist() if cluster_col else None

    prior_table = ExcipientPriorTable(
        train_exc_smiles=train_exc,
        train_labels=train_labels,
        train_api_clusters=train_clusters,
        global_rate=float(np.mean(train_labels)),
    )
    desc_stats = compute_gb_descriptor_norm_stats(train_api + train_exc)

    families_to_run = []
    if args.family:
        families_to_run = [args.family]
    elif args.all:
        families_to_run = ALL_FAMILIES
    else:
        print("Please specify --family <name> or --all")
        return

    summary = {}
    for fam in families_to_run:
        met = run_inference_for_family(fam, prior_table, desc_stats, device, args)
        if met:
            summary[fam] = met

    if summary:
        print(f"\n{'='*70}")
        print("SUMMARY OF HELD-OUT 24-PAIR INFERENCE")
        print(f"{'='*70}")
        print(f"{'Family':<12} {'PR-AUC':>8} {'ROC-AUC':>8} {'MCC':>8} {'Recall':>8} {'Acc':>8} {'TP':>4} {'FP':>4}")
        print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4} {'-'*4}")
        for fam, m in summary.items():
            print(f"{fam:<12} {m['pr_auc']:>8.4f} {m['roc_auc']:>8.4f} {m['mcc']:>8.4f} {m['recall']:>8.4f} {m['accuracy']:>8.4f} {m['tp']:>4d} {m['fp']:>4d}")
        print(f"\nAll predictions accumulated in: {args.output}")


if __name__ == "__main__":
    main()
