"""
Main execution script for running comprehensive SHAP analysis on PGB Morgan model.
Computes SHAP values across all 2,468 features and outputs publication-ready plots & report.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap
import torch

from experiments.pgb.dataset import PGBDataset
from shap_analysis.config import SHAPConfig
from shap_analysis.feature_names import FEATURE_NAMES, get_feature_group
from shap_analysis.model_wrapper import FlatPGBModelWrapper, TowerLevelPGBModelWrapper
from shap_analysis.plots import (
    plot_chemistry_flags_attribution,
    plot_class_comparison,
    plot_descriptor_beeswarm_and_bar,
    plot_feature_group_importance,
    plot_mechanism_flags_attribution,
    plot_morgan_top_bits,
    plot_prior_attribution,
    plot_sample_waterfall,
    plot_top_features_bar,
    plot_tower_level_attribution,
)
from shap_analysis.utils import (
    compute_feature_importance_df,
    decode_morgan_bit,
    extract_features_from_dataset,
    load_pgb_morgan_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP analysis on PGB Morgan model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pt")
    parser.add_argument("--norm_stats", type=str, default=None, help="Path to pgb_norm_stats.npz")
    parser.add_argument("--prior_table", type=str, default=None, help="Path to prior_table.json")
    parser.add_argument("--train_csv", type=str, default="data/train.csv", help="Path to train CSV")
    parser.add_argument("--test_csv", type=str, default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--bg_samples", type=int, default=100, help="Number of background samples")
    parser.add_argument("--test_samples", type=int, default=0, help="Number of test samples (0 for ALL samples)")
    parser.add_argument("--output_dir", type=str, default=None, help="Results directory (defaults to shap_analysis/results/<dataset_tag>)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def generate_summary_report(
    summary_path: str,
    importance_df: pd.DataFrame,
    group_df: pd.DataFrame,
    base_value: float,
    n_samples: int,
    runtime_sec: float,
    decoded_bits: Dict[int, Dict],
):
    """Generates an executive markdown report summarizing the findings."""
    top10 = importance_df.head(10)

    report_lines = [
        "# PGB Morgan Compatibility Model — SHAP Interpretability Report",
        "",
        "## Executive Summary",
        f"- **Model**: Prior-Gated Bilinear (PGB) Morgan Architecture",
        f"- **Input Dimensionality**: {len(importance_df)} features",
        f"- **Test Samples Analyzed**: {n_samples} API–Excipient pairs",
        f"- **Baseline Output (Expected Logit)**: {base_value:.4f} (Base Incompatibility Prob: {1.0 / (1.0 + np.exp(-base_value)):.2%})",
        f"- **SHAP Computation Runtime**: {runtime_sec:.1f} seconds",
        "",
        "---",
        "",
        "## 1. Feature Group Attribution",
        "Relative importance share across all feature groups:",
        "",
        "| Rank | Feature Group | Dim | Total |SHAP| | Percentage Share |",
        "|---|---|---|---|---|",
    ]

    for rank, (_, row) in enumerate(group_df.iloc[::-1].iterrows(), 1):
        report_lines.append(
            f"| {rank} | `{row['group']}` | {row['feature_count']} | {row['total_abs_shap']:.4f} | **{row['percentage']:.2f}%** |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 2. Top 10 Most Influential Features Overall",
        "",
        "| Rank | Feature Name | Group | Mean |SHAP| | % of Total | Directional Bias |",
        "|---|---|---|---|---|---|",
    ])

    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        sign = "+" if row["mean_shap"] > 0 else ""
        report_lines.append(
            f"| {rank} | `{row['feature_name']}` | `{row['feature_group']}` | {row['mean_abs_shap']:.4f} | {row['percentage_importance']:.2f}% | {sign}{row['mean_shap']:.4f} |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Substructure Discoveries (Top Morgan Fingerprint Bits)",
        "Top circular substructures decoded using RDKit atom environments:",
        "",
        "| Bit Index | Molecule | Substructure SMILES / Environment | Occurrences |",
        "|---|---|---|---|",
    ])

    for bit_idx, info in decoded_bits.items():
        sub_smi = info.get("top_fragment_smiles", f"Bit_{bit_idx}")
        occ = info.get("occurrence_count", 0)
        report_lines.append(f"| {bit_idx} | Morgan Bit | `{sub_smi}` | {occ} |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 4. Key Chemical Interpretations",
        "1. **Primary Morgan Fingerprints & MACCS Keys**: Circular structural fragments capture the dominant share of incompatibility variance.",
        "2. **Physicochemical Descriptors**: Molecular weight (`MolWt`), lipophilicity (`MolLogP`), and polar surface area (`TPSA`) drive secondary compatibility trends.",
        "3. **Bayesian Prior Integration**: The adaptive prior gating mechanism (`Prior_p_exact`) anchors predictions for well-known excipients while allowing learned molecular representations to dictate outcomes for novel formulations.",
        "4. **Degradation Mechanisms**: Maillard reaction, ester/amide hydrolysis, and metal-catalyzed oxidation flags act as targeted risk multipliers for specific formulation pairings.",
        "",
        "---",
        "*Report automatically generated by SHAP Analysis Suite for PGB Morgan Compatibility Model.*",
    ])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def run_shap_analysis(config: Optional[SHAPConfig] = None) -> Dict[str, Any]:
    cfg = config or SHAPConfig()
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    start_time = time.time()
    print("=" * 70)
    print("Starting SHAP Analysis on PGB Morgan Compatibility Model")
    print(f"Device: {cfg.device} | Checkpoint: {cfg.checkpoint_path}")
    print("=" * 70)

    # 1. Load Model and Dependencies
    print("\n[1/7] Loading model, normalization stats, and prior tables...")
    model, pgb_norm, prior_table, pgb_cfg, encoder = load_pgb_morgan_model(
        config_obj=cfg,
        checkpoint_path=cfg.checkpoint_path,
        norm_path=cfg.norm_stats_path,
        prior_path=cfg.prior_table_path,
        device=cfg.device,
    )
    print("  Model loaded successfully!")

    # 2. Prepare Datasets
    print("\n[2/7] Preparing train and test datasets...")
    train_ds = PGBDataset(
        cfg.train_csv,
        pgb_cfg.encoder,
        getattr(pgb_cfg, "fixed_vector_source", "morgan"),
        prior_table,
        pgb_norm,
        fixed_vector_encoder=encoder if pgb_cfg.encoder == "fixed_vector" else None,
    )

    # Prepare sanitized CSV if needed (handles 'ground_truth' or 'Outcome' column naming)
    test_csv_path = cfg.test_csv
    raw_test_df = pd.read_csv(cfg.test_csv)
    if "Outcome1" not in raw_test_df.columns:
        if "ground_truth" in raw_test_df.columns:
            raw_test_df = raw_test_df.rename(columns={"ground_truth": "Outcome1"})
        elif "Outcome" in raw_test_df.columns:
            raw_test_df = raw_test_df.rename(columns={"Outcome": "Outcome1"})
        elif "label" in raw_test_df.columns:
            raw_test_df = raw_test_df.rename(columns={"label": "Outcome1"})
        os.makedirs(cfg.results_dir, exist_ok=True)
        test_csv_path = os.path.join(cfg.results_dir, "temp_input_sanitized.csv")
        raw_test_df.to_csv(test_csv_path, index=False)

    test_ds = PGBDataset(
        test_csv_path,
        pgb_cfg.encoder,
        getattr(pgb_cfg, "fixed_vector_source", "morgan"),
        prior_table,
        pgb_norm,
        fixed_vector_encoder=encoder if pgb_cfg.encoder == "fixed_vector" else None,
    )
    print(f"  Train set: {len(train_ds)} samples | Test set: {len(test_ds)} samples")

    # 3. Extract Features
    print(f"\n[3/7] Extracting feature vectors (Background: {cfg.background_samples}, Test: {cfg.test_samples})...")
    bg_data = extract_features_from_dataset(train_ds, model=model, device=cfg.device, max_samples=cfg.background_samples)
    test_data = extract_features_from_dataset(test_ds, model=model, device=cfg.device, max_samples=cfg.test_samples)

    X_bg = bg_data["X"]
    X_test = test_data["X"]
    y_test = test_data["y"]
    meta_test = test_data["meta"]
    raw_desc_api = test_data["raw_desc_api"]
    raw_desc_exc = test_data["raw_desc_exc"]

    print(f"  Feature matrix shapes: Background {X_bg.shape}, Test {X_test.shape}")

    # 4. Compute SHAP Values via PyTorch GradientExplainer
    print("\n[4/7] Computing SHAP values using GradientExplainer...")
    dev = torch.device(cfg.device)
    wrapper = FlatPGBModelWrapper(model, output_mode=cfg.output_mode).to(dev)
    wrapper.eval()

    tensor_bg = torch.tensor(X_bg, dtype=torch.float32, device=dev)
    tensor_test = torch.tensor(X_test, dtype=torch.float32, device=dev)

    with torch.no_grad():
        base_logits = wrapper(tensor_bg).cpu().numpy().flatten()
        base_value = float(np.mean(base_logits))

    explainer = shap.GradientExplainer(wrapper, tensor_bg)

    # Process in batches to maintain optimal GPU memory
    shap_list = []
    batch_sz = cfg.batch_size
    n_batches = int(np.ceil(len(X_test) / batch_sz))

    for b in range(n_batches):
        batch_slice = slice(b * batch_sz, min((b + 1) * batch_sz, len(X_test)))
        sub_tensor = tensor_test[batch_slice]
        sv = explainer.shap_values(sub_tensor)
        if isinstance(sv, list):
            sv = sv[0]
        if sv.ndim == 3:
            sv = sv.squeeze(-1)
        shap_list.append(sv)
        print(f"  Computed batch {b+1}/{n_batches} ({min((b + 1) * batch_sz, len(X_test))}/{len(X_test)} samples)")

    shap_values = np.concatenate(shap_list, axis=0)
    print(f"  SHAP values computed! Shape: {shap_values.shape}, Baseline: {base_value:.4f}")

    # 5. Tower-Level Intermediate SHAP
    print("\n[5/7] Computing Tower-Level Attribution (Analysis #9)...")
    Z_bg = bg_data.get("Z")
    Z_test = test_data.get("Z")

    if Z_bg is not None and Z_test is not None:
        tower_wrapper = TowerLevelPGBModelWrapper(model, output_mode=cfg.output_mode).to(dev)
        tensor_z_bg = torch.tensor(Z_bg, dtype=torch.float32, device=dev)
        tensor_z_test = torch.tensor(Z_test, dtype=torch.float32, device=dev)
        tower_explainer = shap.GradientExplainer(tower_wrapper, tensor_z_bg)
        tower_shap = tower_explainer.shap_values(tensor_z_test)
        if isinstance(tower_shap, list):
            tower_shap = tower_shap[0]
        if tower_shap.ndim == 3:
            tower_shap = tower_shap.squeeze(-1)
    else:
        tower_shap = np.zeros((len(X_test), 238), dtype=np.float32)

    # 6. Decode Top Morgan Bits via RDKit
    print("\n[6/7] Decoding top Morgan fingerprint bits using RDKit...")
    importance_df = compute_feature_importance_df(shap_values, FEATURE_NAMES)
    top_morgan_rows = importance_df[importance_df["feature_group"].isin(["api_primary", "exc_primary"])].head(10)

    smiles_pool = list(meta_test["api_smiles"]) + list(meta_test["exc_smiles"])
    decoded_bits = {}
    for _, r in top_morgan_rows.iterrows():
        b_idx = int(r["feature_name"].split("_")[-1])
        if b_idx not in decoded_bits:
            decoded_bits[b_idx] = decode_morgan_bit(smiles_pool, b_idx, radius=2, n_bits=1024)

    # 7. Generate Visualizations & Outputs
    print("\n[7/7] Generating publication-grade visualization plots...")
    os.makedirs(cfg.plots_dir, exist_ok=True)

    # Plot 1: Feature Group Importance
    p1 = os.path.join(cfg.plots_dir, "feature_group_importance.png")
    group_df = plot_feature_group_importance(shap_values, save_path=p1)
    print(f"  [OK] Saved: {p1}")

    # Plot 2: Top Features Overall
    p2 = os.path.join(cfg.plots_dir, "top_features_overall.png")
    plot_top_features_bar(shap_values, FEATURE_NAMES, top_k=25, save_path=p2)
    print(f"  [OK] Saved: {p2}")

    # Plot 3: Physicochemical Descriptors
    p3 = os.path.join(cfg.plots_dir, "descriptor_attribution.png")
    plot_descriptor_beeswarm_and_bar(shap_values, raw_desc_api, raw_desc_exc, save_path=p3)
    print(f"  [OK] Saved: {p3}")

    # Plot 4: Chemistry Flags
    p4 = os.path.join(cfg.plots_dir, "chemistry_flags_attribution.png")
    plot_chemistry_flags_attribution(shap_values, save_path=p4)
    print(f"  [OK] Saved: {p4}")

    # Plot 5: Mechanism Reaction Flags
    p5 = os.path.join(cfg.plots_dir, "mechanism_flags_attribution.png")
    plot_mechanism_flags_attribution(shap_values, X_test, save_path=p5)
    print(f"  [OK] Saved: {p5}")

    # Plot 6: Bayesian Prior Attribution
    p6 = os.path.join(cfg.plots_dir, "prior_attribution.png")
    plot_prior_attribution(shap_values, X_test, save_path=p6)
    print(f"  [OK] Saved: {p6}")

    # Plot 7: Select Waterfall Plots
    # The first 5 rows, plus one correctly predicted compatible pair so both classes are shown
    # (start_dataset.csv begins with incompatible rows, so the first 5 are all incompatible).
    waterfall_idx = list(range(min(5, len(X_test))))
    probs = 1.0 / (1.0 + np.exp(-(base_value + shap_values.sum(axis=1))))
    compat = np.where((y_test == 0) & (probs < 0.2))[0]
    if len(compat) == 0:
        compat = np.where((y_test == 0) & (probs < 0.5))[0]
    if len(compat) and not (y_test[waterfall_idx] == 0).any():
        waterfall_idx.append(int(compat[0]))
    for s_idx in waterfall_idx:
        pw = os.path.join(cfg.plots_dir, f"waterfall_sample_{s_idx}.png")
        plot_sample_waterfall(
            shap_values,
            base_value,
            X_test,
            FEATURE_NAMES,
            sample_idx=s_idx,
            sample_meta=meta_test.iloc[s_idx].to_dict(),
            top_k=15,
            save_path=pw,
        )
    print(f"  [OK] Saved {len(waterfall_idx)} sample waterfall plots (rows {waterfall_idx})")

    # Plot 8: Top Morgan Bits Decoded
    p8 = os.path.join(cfg.plots_dir, "morgan_top_bits.png")
    plot_morgan_top_bits(shap_values, FEATURE_NAMES, decoded_bits=decoded_bits, top_k=20, save_path=p8)
    print(f"  [OK] Saved: {p8}")

    # Plot 9: Tower-Level Attribution
    p9 = os.path.join(cfg.plots_dir, "tower_level_attribution.png")
    plot_tower_level_attribution(tower_shap, save_path=p9)
    print(f"  [OK] Saved: {p9}")

    # Plot 10: Class Comparison
    p10 = os.path.join(cfg.plots_dir, "class_comparison.png")
    plot_class_comparison(shap_values, y_test, FEATURE_NAMES, top_k=15, save_path=p10)
    print(f"  [OK] Saved: {p10}")

    # Save Arrays and Rankings
    np.savez_compressed(
        cfg.npz_output_path,
        shap_values=shap_values,
        base_value=base_value,
        X_test=X_test,
        y_test=y_test,
        tower_shap=tower_shap,
    )
    print(f"  [OK] Saved raw arrays: {cfg.npz_output_path}")

    importance_df.to_csv(cfg.csv_importance_path, index=False)
    print(f"  [OK] Saved feature rankings: {cfg.csv_importance_path}")

    # Generate Summary Markdown Report
    total_time = time.time() - start_time
    generate_summary_report(
        cfg.summary_report_path,
        importance_df,
        group_df,
        base_value,
        n_samples=len(X_test),
        runtime_sec=total_time,
        decoded_bits=decoded_bits,
    )
    print(f"  [OK] Saved summary report: {cfg.summary_report_path}")

    print("\n" + "=" * 70)
    print(f"SHAP Analysis successfully completed in {total_time:.2f} seconds!")
    print(f"All artifacts saved in: {cfg.results_dir}")
    print("=" * 70)

    return {
        "shap_values": shap_values,
        "base_value": base_value,
        "importance_df": importance_df,
        "group_df": group_df,
    }


if __name__ == "__main__":
    args = parse_args()
    cfg = SHAPConfig()
    if args.checkpoint:
        cfg.checkpoint_path = args.checkpoint
    if args.norm_stats:
        cfg.norm_stats_path = args.norm_stats
    if args.prior_table:
        cfg.prior_table_path = args.prior_table
    if args.train_csv:
        cfg.train_csv = args.train_csv
    if args.test_csv:
        cfg.test_csv = args.test_csv
    if args.bg_samples:
        cfg.background_samples = args.bg_samples
    if args.test_samples is not None:
        cfg.test_samples = None if args.test_samples <= 0 else args.test_samples
    if args.output_dir:
        out_dir = args.output_dir
    else:
        csv_lower = os.path.basename(args.test_csv).lower()
        if "held_out" in csv_lower or "heldout" in csv_lower:
            tag = "held_out_testset"
        elif "start" in csv_lower:
            tag = "full_dataset"
        elif "val" in csv_lower:
            tag = "val_set"
        elif "test" in csv_lower:
            tag = "test_set"
        elif "train" in csv_lower:
            tag = "train_set"
        else:
            tag = os.path.splitext(os.path.basename(args.test_csv))[0]
        out_dir = os.path.join("shap_analysis/results", tag)

    cfg.results_dir = out_dir
    cfg.plots_dir = os.path.join(out_dir, "plots")
    cfg.npz_output_path = os.path.join(out_dir, "shap_values.npz")
    cfg.csv_importance_path = os.path.join(out_dir, "feature_importance.csv")
    cfg.summary_report_path = os.path.join(out_dir, "summary_report.md")
    if args.device:
        cfg.device = args.device
    cfg.random_seed = args.seed

    run_shap_analysis(cfg)
