"""Inference script for global_gated_attention PGB models on the 24-pairs held-out test set."""

import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.pgb.global_gated.config import GATED_FAMILIES, PGBGlobalGatedConfig
from experiments.pgb.dataset import PGBDataset, pgb_collate_fn, PGBNormStats
from experiments.pgb.model import pgb_model_from_config
from experiments.pgb.prior import ExcipientPriorTable
from src.cross_validate import build_encoder
from src.utils import get_device


def get_best_threshold(family: str, metrics_dir: str) -> float:
    candidates = [
        os.path.join(metrics_dir, "val_metrics.json"),
        os.path.join("experiments/results/metrics_pgb/global_gated", family, "val_metrics.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return json.load(f).get("threshold", 0.5)
            except Exception:
                pass
    return 0.5


def find_checkpoint(family: str, config: PGBGlobalGatedConfig) -> str | None:
    candidates = [
        os.path.join(config.checkpoint_dir, "best_model.pt"),
        os.path.join("checkpoints/pgb/global_gated", family, "best_model.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run_inference(
    test_csv: str = "held_out_testset/held_out_test_set.csv",
    output_path: str = "held_out_testset/held_out_predictions_pgb_global_gated.csv",
    families: list[str] = None,
):
    device = get_device("auto")
    test_df = pd.read_csv(test_csv)

    if "ground_truth" in test_df.columns:
        test_df = test_df.rename(columns={"ground_truth": "Outcome1"})

    temp_test_csv = "held_out_testset/temp_held_out_pgb_gated.csv"
    test_df.to_csv(temp_test_csv, index=False)
    results_df = test_df.copy()

    target_families = families if families else list(GATED_FAMILIES.keys())

    for family in target_families:
        if family not in GATED_FAMILIES:
            continue
        print(f"Running PGB (global_gated) inference for {family}...")
        config = PGBGlobalGatedConfig()
        config.apply_family_spec(family)

        ckpt_path = find_checkpoint(family, config)
        if not ckpt_path:
            print(f"  Warning: Checkpoint not found at {config.checkpoint_dir}/best_model.pt. Skipping.")
            continue

        ckpt_dir = os.path.dirname(ckpt_path)

        encoder = build_encoder(config, device)
        fv_enc = encoder if config.encoder == "fixed_vector" else None

        norm_path = os.path.join(ckpt_dir, "pgb_norm_stats.npz")
        prior_path = os.path.join(ckpt_dir, "prior_table.json")

        if not os.path.exists(norm_path) or not os.path.exists(prior_path):
            print(f"  Warning: Stats/Prior missing in {ckpt_dir}. Skipping.")
            continue

        pgb_norm = PGBNormStats.load(norm_path)
        prior_table = ExcipientPriorTable.load(prior_path)

        test_ds = PGBDataset(
            temp_test_csv,
            config.encoder,
            getattr(config, "fixed_vector_source", "maccs"),
            prior_table,
            pgb_norm,
            fixed_vector_encoder=fv_enc,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=config.pgb_batch_size,
            shuffle=False,
            collate_fn=pgb_collate_fn,
            num_workers=0,
        )

        model = pgb_model_from_config(config, encoder)

        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

        model.to(device)
        model.eval()

        all_logits = []
        with torch.no_grad():
            for batch in test_loader:
                batch_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                                for k, v in batch.items()}
                _ = batch_device.pop("label", None)
                _ = batch_device.pop("sample_weight", None)
                logits = model(batch_device)
                all_logits.append(logits.cpu())

        logits = torch.cat(all_logits).numpy()
        probs = 1 / (1 + np.exp(-logits))

        results_df[f"prob_pgb_gated_{family}"] = probs
        thresh = get_best_threshold(family, config.metrics_dir)
        results_df[f"pred_pgb_gated_{family}"] = (probs >= thresh).astype(int)

        del model, encoder, fv_enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df.to_csv(output_path, index=False)
    if os.path.exists(temp_test_csv):
        os.remove(temp_test_csv)

    print(f"\nSaved PGB global_gated inference results to: {output_path}")
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="all", choices=list(GATED_FAMILIES.keys()) + ["all"])
    parser.add_argument("--output", type=str, default=None, help="Custom output CSV path")
    args = parser.parse_args()

    fams = None if args.family == "all" else [args.family]
    out_path = args.output or "held_out_testset/held_out_predictions_pgb_global_gated.csv"
    run_inference(output_path=out_path, families=fams)
