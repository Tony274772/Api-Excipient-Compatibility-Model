"""Inference on the held-out test set across multiple trained architectures.

Loads each checkpoint, runs inference on the 24 held-out pairs,
and produces a single CSV with per-model logit, probability, and prediction columns.

Usage:
    python inference.py                         # run all available checkpoints
    python inference.py --models molformer_cross_attn_global_gated_asl molformer_cross_attn_global_gated_bce
    python inference.py --device cpu            # force CPU
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.utils import seed_everything, get_device
from src.encoders import ENCODER_REGISTRY
from src.model import CompatibilityModel
from src.dataset import CompatibilityDataset, collate_fn
from src.descriptors import DESCRIPTOR_NAMES

# ── Registry: checkpoint dir name → (encoder, fusion, pooling, loss) ──────────
# Maps every known checkpoint directory name to the CLI-level config needed
# to reconstruct the model architecture.

MODEL_REGISTRY = {
    # ── MoLFormer variants ──
    # Note: all cross_attn checkpoints trained with global_gated_attention pooling
    "molformer_cross_attn_asl":             ("molformer", "cross_attn", "global_gated_attention", "asl"),
    "molformer_cross_attn_bce":             ("molformer", "cross_attn", "global_gated_attention", "bce"),
    "molformer_cross_attn_focal":           ("molformer", "cross_attn", "global_gated_attention", "focal"),
    "molformer_cross_attn_weighted_bce":    ("molformer", "cross_attn", "global_gated_attention", "weighted_bce"),
    "molformer_cross_attn_global_gated_asl":("molformer", "cross_attn", "global_gated_attention", "asl"),
    "molformer_cross_attn_global_gated_bce":("molformer", "cross_attn", "global_gated_attention", "bce"),
    "molformer_cross_attn_pairwise_asl":    ("molformer", "cross_attn", "explicit_pairwise",      "asl"),
    "molformer_cross_attn_pairwise_bce":    ("molformer", "cross_attn", "explicit_pairwise",      "bce"),
    "molformer_concat_asl":                 ("molformer", "concat",     "global_gated_attention", "asl"),
    # ── ChemBERTa variants ──
    "chemberta_concat_asl":                 ("chemberta", "concat",     "global_gated_attention", "asl"),
    "chemberta_cross_attn_asl":             ("chemberta", "cross_attn", "global_gated_attention", "asl"),
    "chemberta_cross_attn_bce":             ("chemberta", "cross_attn", "global_gated_attention", "bce"),
    "chemberta_cross_attn_focal":           ("chemberta", "cross_attn", "global_gated_attention", "focal"),
    "chemberta_cross_attn_weighted_bce":    ("chemberta", "cross_attn", "global_gated_attention", "weighted_bce"),
    "chemberta_cross_attn_global_gated_asl":("chemberta", "cross_attn", "global_gated_attention", "asl"),
    "chemberta_cross_attn_global_gated_bce":("chemberta", "cross_attn", "global_gated_attention", "bce"),
    "chemberta_cross_attn_pairwise_asl":    ("chemberta", "cross_attn", "explicit_pairwise",      "asl"),
    "chemberta_cross_attn_pairwise_bce":    ("chemberta", "cross_attn", "explicit_pairwise",      "bce"),
    # ── GIN variants ──
    "pretrained_gin_cross_attn_asl":        ("pretrained_gin", "cross_attn", "global_gated_attention", "asl"),
    "pretrained_gin_concat_asl":            ("pretrained_gin", "concat",     "global_gated_attention", "asl"),
    # ── Fixed-vector (pubchemfp) variants ──
    "fixed_vector_pubchemfp_concat_asl":    ("fixed_vector",   "concat",     "global_gated_attention", "asl"),
}


class HeldOutDataset(torch.utils.data.Dataset):
    """Minimal dataset for the held-out test CSV.

    Uses the held-out-specific descriptor CSVs but applies the same
    normalization stats computed from the original training split.
    """

    def __init__(
        self,
        csv_path: str,
        api_desc_path: str,
        exc_desc_path: str,
        norm_stats_path: str,
    ):
        self.df = pd.read_csv(csv_path)

        # Load descriptor look-ups
        api_desc_df = pd.read_csv(api_desc_path)
        exc_desc_df = pd.read_csv(exc_desc_path)

        self.api_desc_map = {}
        for _, row in api_desc_df.iterrows():
            self.api_desc_map[row["API_CID"]] = np.array(
                [row[n] for n in DESCRIPTOR_NAMES], dtype=np.float32
            )

        self.exc_desc_map = {}
        for _, row in exc_desc_df.iterrows():
            self.exc_desc_map[row["Excipient_CID"]] = np.array(
                [row[n] for n in DESCRIPTOR_NAMES], dtype=np.float32
            )

        # Load normalization stats (from training split)
        with open(norm_stats_path, "r") as f:
            stats = json.load(f)
        self.api_mean = np.array(stats["api_mean"], dtype=np.float32)
        self.api_std = np.array(stats["api_std"], dtype=np.float32)
        self.exc_mean = np.array(stats["exc_mean"], dtype=np.float32)
        self.exc_std = np.array(stats["exc_std"], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        api_smiles = str(row["API_Smiles"])
        exc_smiles = str(row["Excipient_Smiles"]) if pd.notna(row.get("Excipient_Smiles")) else ""

        # Descriptors (normalized)
        api_desc_raw = self.api_desc_map.get(
            row["API_CID"], np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32)
        )
        exc_desc_raw = self.exc_desc_map.get(
            row["Excipient_CID"], np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32)
        )

        from src.descriptors import normalize_descriptors
        api_desc = normalize_descriptors(api_desc_raw, self.api_mean, self.api_std)
        exc_desc = normalize_descriptors(exc_desc_raw, self.exc_mean, self.exc_std)

        api_desc = np.nan_to_num(api_desc, nan=0.0)
        exc_desc = np.nan_to_num(exc_desc, nan=0.0)

        exc_available = 1.0 if exc_smiles != "" else 0.0
        label = float(row["ground_truth"])

        return {
            "api_smiles": api_smiles,
            "exc_smiles": exc_smiles,
            "api_desc": torch.tensor(api_desc, dtype=torch.float32),
            "exc_desc": torch.tensor(exc_desc, dtype=torch.float32),
            "exc_available": torch.tensor(exc_available, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "sample_weight": torch.tensor(1.0, dtype=torch.float32),
        }


def build_encoder(config, device):
    """Instantiate the selected encoder (same as main.py)."""
    encoder_cls = ENCODER_REGISTRY[config.encoder]

    if config.encoder == "molformer":
        encoder = encoder_cls(config.molformer_model_path, device=str(device))
    elif config.encoder == "pretrained_gin":
        encoder = encoder_cls(config.gin_pretrained_name, device=str(device))
    elif config.encoder == "chemberta":
        encoder = encoder_cls(config.chemberta_model_path, device=str(device))
    elif config.encoder == "fixed_vector":
        encoder = encoder_cls(
            source=config.fixed_vector_source,
            vector_path=config.fixed_vector_path,
            device=str(device),
        )
    else:
        raise ValueError(f"Unknown encoder: {config.encoder}")

    encoder.to(device)
    config.encoder_output_dim = encoder.output_dim
    return encoder


def load_threshold(metrics_dir: str) -> float:
    """Load the val-tuned threshold from saved val_metrics.json."""
    val_path = os.path.join(metrics_dir, "val_metrics.json")
    if os.path.exists(val_path):
        with open(val_path, "r") as f:
            metrics = json.load(f)
        return metrics.get("threshold", 0.5)
    print(f"  Warning: {val_path} not found, using threshold=0.5")
    return 0.5


def run_inference_single_model(
    model_name: str,
    config: Config,
    dataset: HeldOutDataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run inference for a single model checkpoint on the held-out dataset.

    Returns:
        logits:       [N] raw logit values
        probs:        [N] sigmoid probabilities
        predictions:  [N] binary predictions (0/1) using val-tuned threshold
        threshold:    the val-tuned threshold used
    """
    from ensemble._common import build_model_from_checkpoint
    
    # We ignore the passed-in config, as build_model_from_checkpoint reads it directly
    model, config = build_model_from_checkpoint(model_name, device)

    # Load val-tuned threshold
    threshold = load_threshold(config.metrics_dir)

    # Run inference
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),  # All 24 pairs in one batch
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=0,
    )

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch_device = {
                "api_smiles": batch["api_smiles"],
                "exc_smiles": batch["exc_smiles"],
                "api_desc": batch["api_desc"].to(device),
                "exc_desc": batch["exc_desc"].to(device),
                "exc_available": batch["exc_available"].to(device),
            }
            logits = model(batch_device)
            all_logits.append(logits.cpu())

    logits = torch.cat(all_logits).numpy()
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    predictions = (probs >= threshold).astype(int)

    return logits, probs, predictions, threshold


def discover_available_models(checkpoints_dir: str) -> list[str]:
    """Discover all available checkpoint directories with best_model.pt."""
    available = []
    if not os.path.isdir(checkpoints_dir):
        return available
    for name in sorted(os.listdir(checkpoints_dir)):
        if name == "random_split":
            continue
        ckpt_path = os.path.join(checkpoints_dir, name, "best_model.pt")
        if os.path.isfile(ckpt_path):
            available.append(name)
    return available


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on held-out test set across multiple trained models"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific model checkpoint names to run (e.g., molformer_cross_attn_global_gated_asl). "
             "Default: run all available."
    )
    parser.add_argument("--device", default=None, help="Device to use (cpu/cuda)")
    parser.add_argument(
        "--output", default="held_out_testset/held_out_predictions.csv",
        help="Output CSV path (default: held_out_testset/held_out_predictions.csv)"
    )
    parser.add_argument(
        "--held_out_csv", default="held_out_testset/held_out_test_set.csv",
        help="Path to held-out test CSV"
    )
    parser.add_argument(
        "--held_out_api_desc", default="held_out_testset/api_descriptors.csv",
        help="Path to held-out API descriptors"
    )
    parser.add_argument(
        "--held_out_exc_desc", default="held_out_testset/excipient_descriptors.csv",
        help="Path to held-out excipient descriptors"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device or "auto")

    print(f"Device: {device}")
    print(f"Held-out CSV: {args.held_out_csv}")

    # Discover available models
    checkpoints_dir = "checkpoints"
    all_available = discover_available_models(checkpoints_dir)

    if args.models:
        # Validate user-specified models
        models_to_run = []
        for m in args.models:
            ckpt_path = os.path.join(checkpoints_dir, m, "best_model.pt")
            if not os.path.isfile(ckpt_path):
                print(f"Warning: checkpoint not found for '{m}', skipping")
                continue
            models_to_run.append(m)
    else:
        models_to_run = all_available

    if not models_to_run:
        print("ERROR: No valid model checkpoints found to run inference on.")
        sys.exit(1)

    print(f"\nModels to run ({len(models_to_run)}):")
    for m in models_to_run:
        print(f"  - {m}")

    # Build held-out dataset (shared across all models)
    norm_stats_path = "models/descriptor_norm_stats.json"
    dataset = HeldOutDataset(
        csv_path=args.held_out_csv,
        api_desc_path=args.held_out_api_desc,
        exc_desc_path=args.held_out_exc_desc,
        norm_stats_path=norm_stats_path,
    )
    print(f"\nHeld-out samples: {len(dataset)}")

    # Start with the base held-out data columns
    result_df = dataset.df[["Api_name", "Excipient_name", "API_CID", "Excipient_CID",
                             "API_Smiles", "Excipient_Smiles", "ground_truth"]].copy()

    # Run inference for each model
    encoder_cache = {}  # Cache encoders by type to avoid reloading

    from ensemble._common import load_config_for_checkpoint

    for model_name in models_to_run:
        config, _ = load_config_for_checkpoint(model_name, device)

        print(f"\n{'='*60}")
        print(f"Running: {model_name}")
        print(f"  encoder={config.encoder}, fusion={config.fusion}, pooling={config.pooling}, loss={config.loss}")

        # Set fixed_vector_path for fixed_vector encoder
        if config.encoder == "fixed_vector":
            src = getattr(config, "fixed_vector_source", "pubchemfp")
            path_map = {
                "pubchemfp": "data/pubchem_fps.csv",
                "maccs": "data/maccs_keys.csv",
                "mol2vec": "data/mol2vec_embeddings.csv",
                "morgan": "data/morgan_fps.csv",
            }
            config.fixed_vector_path = path_map.get(src, "data/pubchem_fps.csv")

        try:
            logits, probs, predictions, threshold = run_inference_single_model(
                model_name, config, dataset, device
            )

            # Add columns to result DataFrame
            result_df[f"{model_name}_logit"] = logits
            result_df[f"{model_name}_probability"] = probs
            result_df[f"{model_name}_prediction"] = predictions
            result_df[f"{model_name}_threshold"] = threshold

            # Print summary
            correct = (predictions == result_df["ground_truth"].values).sum()
            total = len(predictions)
            print(f"  Threshold: {threshold:.4f}")
            print(f"  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
            print(f"  Predicted positive: {predictions.sum()}/{total}")

        except Exception as e:
            print(f"  ERROR: {e}")
            result_df[f"{model_name}_logit"] = np.nan
            result_df[f"{model_name}_probability"] = np.nan
            result_df[f"{model_name}_prediction"] = np.nan
            result_df[f"{model_name}_threshold"] = np.nan

    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    try:
        result_df.to_csv(args.output, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to: {args.output}")
    except PermissionError:
        fallback_output = args.output.replace(".csv", "_fallback.csv")
        result_df.to_csv(fallback_output, index=False)
        print(f"\n{'='*60}")
        print(f"WARNING: Permission denied when saving to {args.output}.")
        print(f"Results saved to fallback: {fallback_output}")
    print(f"Columns: {list(result_df.columns)}")

    # Print compact summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<50} {'Acc':>6} {'Thresh':>8}")
    print(f"{'-'*50} {'-'*6} {'-'*8}")
    for model_name in models_to_run:
        pred_col = f"{model_name}_prediction"
        thresh_col = f"{model_name}_threshold"
        if pred_col in result_df.columns and not result_df[pred_col].isna().all():
            preds = result_df[pred_col].values
            gt = result_df["ground_truth"].values
            correct = (preds == gt).sum()
            total = len(gt)
            threshold = result_df[thresh_col].iloc[0]
            print(f"{model_name:<50} {correct}/{total:>3} {threshold:>8.4f}")
        else:
            print(f"{model_name:<50} {'FAILED':>6} {'N/A':>8}")


if __name__ == "__main__":
    main()
