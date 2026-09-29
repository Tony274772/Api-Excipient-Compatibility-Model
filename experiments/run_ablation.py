"""Loss & Sampler Ablation Study for MolFormer Cross-Attention (Global Gated).

Runs an 8-experiment 4x2 factorial ablation grid on the group-safe data split:
    - Losses (4):   BCE, Weighted BCE, Focal, ASL
    - Sampler (2):  ON (WeightedRandomSampler), OFF (natural distribution)

Base Model:
    - Encoder:  MoLFormer-XL (frozen embeddings with SMILES cache)
    - Fusion:   Cross-Attention
    - Pooling:  Global Gated Attention
    - Descriptors: 21 physicochemical descriptors included
    - Split:    Group-safe Butina cluster split (data/train.csv, data/val.csv, data/test.csv)

Outputs:
    - Per-experiment checkpoints and metric JSONs in experiments/results/<exp_name>/
    - Consolidated ablation_summary.csv, ablation_summary.json, and markdown table in experiments/results/
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.utils import seed_everything, get_device
from src.encoders import ENCODER_REGISTRY
from src.model import CompatibilityModel
from src.dataset import build_dataloaders
from src.train import train
from src.evaluate import evaluate_model, save_metrics


# ---------------------------------------------------------------------------
# 8-Experiment Ablation Definition (4 Losses x 2 Sampler States)
# ---------------------------------------------------------------------------
ABLATION_EXPERIMENTS = [
    {
        "id": "A",
        "name": "exp_A_bce_sampler_on",
        "description": "BCE loss + Balanced Sampler (ON)",
        "loss": "bce",
        "use_balanced_sampler": True,
    },
    {
        "id": "B",
        "name": "exp_B_bce_sampler_off",
        "description": "BCE loss + No Sampler (OFF)",
        "loss": "bce",
        "use_balanced_sampler": False,
    },
    {
        "id": "C",
        "name": "exp_C_asl_sampler_on",
        "description": "ASL loss + Balanced Sampler (ON)",
        "loss": "asl",
        "use_balanced_sampler": True,
    },
    {
        "id": "D",
        "name": "exp_D_asl_sampler_off",
        "description": "ASL loss + No Sampler (OFF)",
        "loss": "asl",
        "use_balanced_sampler": False,
    },
    {
        "id": "E",
        "name": "exp_E_weighted_bce_sampler_off",
        "description": "Weighted BCE loss + No Sampler (OFF)",
        "loss": "weighted_bce",
        "use_balanced_sampler": False,
    },
    {
        "id": "F",
        "name": "exp_F_weighted_bce_sampler_on",
        "description": "Weighted BCE loss + Balanced Sampler (ON)",
        "loss": "weighted_bce",
        "use_balanced_sampler": True,
    },
    {
        "id": "G",
        "name": "exp_G_focal_sampler_on",
        "description": "Focal loss + Balanced Sampler (ON)",
        "loss": "focal",
        "use_balanced_sampler": True,
    },
    {
        "id": "H",
        "name": "exp_H_focal_sampler_off",
        "description": "Focal loss + No Sampler (OFF)",
        "loss": "focal",
        "use_balanced_sampler": False,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Loss & Sampler Ablation Study (4x2=8 runs)")
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        help="Experiment to run: 'all', specific letter (e.g. 'A', 'E'), name ('exp_A_bce_sampler_on'), or comma-separated list ('A,B,C').",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to run on ('cuda', 'cpu', or auto)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate (default: 1.5e-4)")
    parser.add_argument("--max_epochs", type=int, default=150, help="Max epochs (default: 150)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "experiments", "results"),
                        help="Root output directory for results")
    return parser.parse_args()


def build_base_config(args) -> Config:
    """Instantiate baseline configuration for molformer_cross_attn_global_gated."""
    config = Config()
    config.encoder = "molformer"
    config.fusion = "cross_attn"
    config.pooling = "global_gated_attention"
    config.use_descriptors = True
    config.split_type = "cluster"  # Uses group-safe Butina cluster split from data/
    
    # Training hyperparams
    config.lr = args.lr
    config.batch_size = args.batch_size
    config.max_epochs = args.max_epochs
    config.seed = args.seed
    if args.device is not None:
        config.device = args.device

    # Ensure CSV paths are bound to the group-safe data directory
    config.resolve_csv_paths()
    return config


def run_single_ablation(exp_spec: Dict[str, Any], base_config: Config, shared_encoder, device, output_root: str) -> Dict[str, Any]:
    """Execute a single ablation experiment and return its metrics."""
    exp_id = exp_spec["id"]
    exp_name = exp_spec["name"]
    loss_name = exp_spec["loss"]
    sampler_status = exp_spec["use_balanced_sampler"]

    exp_dir = os.path.join(output_root, exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    metrics_dir = os.path.join(exp_dir, "metrics")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"EXPERIMENT [{exp_id}]: {exp_name.upper()}")
    print(f"Description: {exp_spec['description']}")
    print(f"Loss: {loss_name} | Balanced Sampler: {'ON' if sampler_status else 'OFF'}")
    print(f"Output Directory: {exp_dir}")
    print(f"{'='*75}\n")

    # Clone config for this specific ablation
    exp_config = copy.deepcopy(base_config)
    exp_config.loss = loss_name
    exp_config.use_balanced_sampler = sampler_status
    exp_config.checkpoint_dir = ckpt_dir
    exp_config.metrics_dir = metrics_dir

    # Compute positive prior from training data
    train_df = pd.read_csv(exp_config.train_csv)
    exp_config.positive_prior = float(train_df["Outcome1"].mean())

    # Seed everything for deterministic, fair comparison across runs
    seed_everything(exp_config.seed)

    # Build model using shared frozen encoder (preserves SMILES cache for huge speedup)
    model = CompatibilityModel(exp_config, shared_encoder)
    model.init_bias(exp_config.positive_prior)
    model.to(device)

    # Build dataloaders for group-safe split (data/train.csv, data/val.csv, data/test.csv)
    train_loader, val_loader, test_loader = build_dataloaders(exp_config)
    print(f"Dataset: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    print(f"Training sampler active: {train_loader.sampler is not None and not isinstance(train_loader.sampler, torch.utils.data.RandomSampler)}")

    # Train
    start_time = time.time()
    model, history = train(model, train_loader, val_loader, exp_config, device)
    elapsed_time = time.time() - start_time

    # Evaluate on val (tuning threshold) and evaluate on test (applying val-tuned threshold)
    val_metrics, test_metrics, best_thresh = evaluate_model(
        model, val_loader, test_loader, device, exp_config
    )

    print(f"\n--- Results for Experiment [{exp_id}] ---")
    print(f"Runtime: {elapsed_time:.1f}s | Best Threshold: {best_thresh:.4f}")
    print(f"Val PR-AUC:  {val_metrics['pr_auc']:.4f} | F1: {val_metrics['f1']:.4f} | MCC: {val_metrics['mcc']:.4f} | Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
    print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f} | F1: {test_metrics['f1']:.4f} | MCC: {test_metrics['mcc']:.4f} | Prec: {test_metrics['precision']:.4f} | Rec: {test_metrics['recall']:.4f} | Acc: {test_metrics['accuracy']:.4f}")

    # Save per-experiment files
    save_metrics(val_metrics, os.path.join(metrics_dir, "val_metrics.json"))
    save_metrics(test_metrics, os.path.join(metrics_dir, "test_metrics.json"))
    save_metrics({"history": history}, os.path.join(metrics_dir, "training_history.json"))

    config_dict = {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "loss": loss_name,
        "use_balanced_sampler": sampler_status,
        "positive_prior": exp_config.positive_prior,
        "lr": exp_config.lr,
        "batch_size": exp_config.batch_size,
        "seed": exp_config.seed,
        "split_type": exp_config.split_type,
        "train_csv": exp_config.train_csv,
        "val_csv": exp_config.val_csv,
        "test_csv": exp_config.test_csv,
        "elapsed_seconds": round(elapsed_time, 2),
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    return {
        "id": exp_id,
        "name": exp_name,
        "loss": loss_name,
        "sampler": "ON" if sampler_status else "OFF",
        "best_epoch": int(np.argmax(history["val_pr_auc"])) + 1 if history["val_pr_auc"] else 0,
        "best_threshold": round(best_thresh, 4),
        "runtime_sec": round(elapsed_time, 1),
        # Validation
        "val_pr_auc": round(val_metrics["pr_auc"], 4),
        "val_f1": round(val_metrics["f1"], 4),
        "val_mcc": round(val_metrics["mcc"], 4),
        "val_precision": round(val_metrics["precision"], 4),
        "val_recall": round(val_metrics["recall"], 4),
        "val_accuracy": round(val_metrics["accuracy"], 4),
        # Test
        "test_pr_auc": round(test_metrics["pr_auc"], 4),
        "test_f1": round(test_metrics["f1"], 4),
        "test_mcc": round(test_metrics["mcc"], 4),
        "test_precision": round(test_metrics["precision"], 4),
        "test_recall": round(test_metrics["recall"], 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        # Confusion matrices
        "val_cm": val_metrics["confusion_matrix"],
        "test_cm": test_metrics["confusion_matrix"],
    }


def save_summary_reports(results: List[Dict[str, Any]], output_root: str):
    """Generate consolidated CSV, JSON, and Markdown summary files."""
    os.makedirs(output_root, exist_ok=True)
    summary_json_path = os.path.join(output_root, "ablation_summary.json")
    summary_csv_path = os.path.join(output_root, "ablation_summary.csv")
    summary_md_path = os.path.join(output_root, "ablation_summary.md")

    # If some runs were already saved previously, merge them
    if os.path.exists(summary_json_path):
        try:
            with open(summary_json_path, "r") as f:
                existing_data = json.load(f)
            existing_map = {r["id"]: r for r in existing_data}
            for r in results:
                existing_map[r["id"]] = r
            # Sort by ID
            sorted_keys = sorted(existing_map.keys())
            results = [existing_map[k] for k in sorted_keys]
        except Exception as e:
            print(f"Warning: Could not merge existing summary JSON: {e}")

    # Save JSON
    with open(summary_json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV (flattened)
    csv_rows = []
    for r in results:
        csv_rows.append({
            "Exp_ID": r["id"],
            "Experiment": r["name"],
            "Sampler": r["sampler"],
            "Loss": r["loss"].upper(),
            "Val_PR_AUC": r["val_pr_auc"],
            "Val_F1": r["val_f1"],
            "Val_MCC": r["val_mcc"],
            "Test_PR_AUC": r["test_pr_auc"],
            "Test_F1": r["test_f1"],
            "Test_MCC": r["test_mcc"],
            "Test_Precision": r["test_precision"],
            "Test_Recall": r["test_recall"],
            "Test_Accuracy": r["test_accuracy"],
            "Threshold": r["best_threshold"],
            "Runtime_s": r["runtime_sec"],
        })
    df_summary = pd.DataFrame(csv_rows)
    df_summary.to_csv(summary_csv_path, index=False)

    # Generate Markdown Table
    md_content = [
        "# Loss & Sampler Ablation Study Results",
        "",
        "**Base Model Architecture:** `molformer_cross_attn_global_gated`",
        "**Data Split:** Group-safe API-level Butina cluster split (`data/train.csv`, `data/val.csv`, `data/test.csv`)",
        "",
        "## Summary Comparison Table",
        "",
        "| Exp | Sampler | Loss | Val PR-AUC | Val F1 | Val MCC | Test PR-AUC | Test F1 | Test MCC | Test Prec | Test Rec | Test Acc | Thresh |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        md_content.append(
            f"| **{r['id']}** | {r['sampler']} | {r['loss'].upper()} | "
            f"{r['val_pr_auc']:.4f} | {r['val_f1']:.4f} | {r['val_mcc']:.4f} | "
            f"**{r['test_pr_auc']:.4f}** | {r['test_f1']:.4f} | {r['test_mcc']:.4f} | "
            f"{r['test_precision']:.4f} | {r['test_recall']:.4f} | {r['test_accuracy']:.4f} | "
            f"{r['best_threshold']:.4f} |"
        )

    md_content.extend([
        "",
        "## Key Insights & Discussion",
        "",
        "1. **Sampler ON vs OFF across Losses:**",
        "   - **BCE**: Comparing Exp A (ON) vs Exp B (OFF) tests whether equalizing positive and negative batch representation helps standard cross-entropy.",
        "   - **Weighted BCE**: Comparing Exp E (OFF) vs Exp F (ON) isolates whether `pos_weight` alone is sufficient or whether combining it with balanced sampling produces better discrimination.",
        "   - **Focal Loss**: Comparing Exp G (ON) vs Exp H (OFF) tests how focal modulating factors interact with batch class balance.",
        "   - **ASL (Asymmetric Loss)**: Comparing Exp C (ON) vs Exp D (OFF) tests asymmetric margin clipping and negative suppression under balanced vs natural sampling.",
        "",
    ])

    with open(summary_md_path, "w") as f:
        f.write("\n".join(md_content))

    print(f"\nConsolidated summaries saved to:")
    print(f"  - {summary_csv_path}")
    print(f"  - {summary_json_path}")
    print(f"  - {summary_md_path}")


def main():
    args = parse_args()
    output_root = args.output_dir
    os.makedirs(output_root, exist_ok=True)

    base_config = build_base_config(args)
    seed_everything(base_config.seed)
    device = get_device(base_config.device)

    print("=" * 75)
    print("MOLFORMER LOSS & SAMPLER ABLATION STUDY (4x2 = 8 CONFIGURATIONS)")
    print(f"Device: {device}")
    print(f"Data Dir: {base_config.data_dir} (Group-safe cluster split: train/val/test.csv)")
    print(f"Output Root: {output_root}")
    print("=" * 75)

    # Filter target experiments
    target_exp_arg = args.exp.strip().lower()
    if target_exp_arg == "all":
        selected_exps = ABLATION_EXPERIMENTS
    else:
        target_tokens = [t.strip().upper() for t in target_exp_arg.split(",")]
        selected_exps = [
            e for e in ABLATION_EXPERIMENTS
            if e["id"].upper() in target_tokens or e["name"].lower() in [t.lower() for t in target_tokens]
        ]
        if not selected_exps:
            raise ValueError(f"No matching experiments found for '--exp {args.exp}'. Available IDs: {[e['id'] for e in ABLATION_EXPERIMENTS]}")

    print(f"Experiments scheduled to run ({len(selected_exps)}): {[e['id'] + ': ' + e['name'] for e in selected_exps]}")

    # Build shared MoLFormer encoder once to reuse in-memory SMILES cache across all runs
    print("\nInstantiating shared MoLFormer-XL encoder...")
    encoder_cls = ENCODER_REGISTRY[base_config.encoder]
    shared_encoder = encoder_cls(base_config.molformer_model_path, device=str(device))
    shared_encoder.to(device)
    base_config.encoder_output_dim = shared_encoder.output_dim
    print(f"MoLFormer-XL encoder loaded (output_dim={shared_encoder.output_dim}).")

    all_results = []
    total_start = time.time()

    for idx, exp_spec in enumerate(selected_exps, 1):
        print(f"\n>>> Running Ablation [{idx}/{len(selected_exps)}]: {exp_spec['id']} - {exp_spec['name']} <<<")
        res = run_single_ablation(exp_spec, base_config, shared_encoder, device, output_root)
        all_results.append(res)
        # Update summary after each experiment so partial progress is never lost
        save_summary_reports(all_results, output_root)

    total_time = time.time() - total_start
    print(f"\n{'='*75}")
    print(f"ALL {len(selected_exps)} ABLATIONS COMPLETED IN {total_time/60:.2f} MINUTES.")
    print(f"Results located in: {output_root}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
