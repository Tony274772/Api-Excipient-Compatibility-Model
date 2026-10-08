"""Inference script for single-run PGB models on the 24-pairs held-out test set."""

import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.pgb.config import PGBConfig, FAMILIES
from experiments.pgb.dataset import PGBDataset, pgb_collate_fn, PGBNormStats
from experiments.pgb.model import pgb_model_from_config
from experiments.pgb.prior import ExcipientPriorTable
from src.cross_validate import build_encoder
from src.utils import get_device


def get_best_threshold(family: str, metrics_dir: str) -> float:
    candidates = [
        os.path.join("experiments/results/metrics_pgb", family, "val_metrics.json"),
        os.path.join(metrics_dir, "val_metrics.json"),
        os.path.join(metrics_dir, "single_run", "val_metrics.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return json.load(f).get("threshold", 0.5)
            except Exception:
                pass
    return 0.5


def find_checkpoint(family: str, config: PGBConfig) -> str | None:
    candidates = [
        os.path.join("checkpoints/pgb", family, "best_model.pt"),
        os.path.join("checkpoints/pgb", family, "single_run", "best_model.pt"),
        os.path.join(config.checkpoint_dir, "best_model.pt"),
        os.path.join(config.checkpoint_dir, "single_run", "best_model.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run_inference(
    test_csv: str = "held_out_testset/held_out_test_set.csv",
    output_path: str = "held_out_testset/held_out_predictions_pgb_single.csv",
    families: list[str] = None,
    random_split: bool = False,
):
    device = get_device("auto")
    test_df = pd.read_csv(test_csv)

    if "ground_truth" in test_df.columns:
        test_df = test_df.rename(columns={"ground_truth": "Outcome1"})

    temp_test_csv = "held_out_testset/temp_held_out_pgb.csv"
    test_df.to_csv(temp_test_csv, index=False)
    results_df = test_df.copy()

    target_families = families if families else list(FAMILIES.keys())

    for family in target_families:
        if family not in FAMILIES:
            continue
        print(f"Running PGB inference for {family} (random_split={random_split})...")
        config = PGBConfig()
        if random_split:
            config.data_dir = "data/random_split"
            config.pgb_metrics_root = "experiments/results/metrics_pgb/random_split"
            config.pgb_checkpoint_root = "checkpoints/pgb/random_split"
        
        config.apply_family_spec(family)

        ckpt_path = find_checkpoint(family, config)
        if not ckpt_path:
            print(f"  Warning: Checkpoint not found for {family}. Skipping.")
            continue

        ckpt_dir = os.path.dirname(ckpt_path)

        # Build encoder
        encoder = build_encoder(config, device)
        if not encoder.is_sequence_capable:
            config.fusion = "concat"

        fv_enc = encoder if config.encoder == "fixed_vector" else None

        # Load norm stats & priors
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

        # Load weights
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

        results_df[f"prob_pgb_{family}"] = probs
        thresh = get_best_threshold(family, config.metrics_dir)
        results_df[f"pred_pgb_{family}"] = (probs >= thresh).astype(int)

        del model, encoder, fv_enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df.to_csv(output_path, index=False)
    if os.path.exists(temp_test_csv):
        os.remove(temp_test_csv)

    print(f"\nSaved PGB inference results to: {output_path}")
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_split", action="store_true", help="Run inference on random_split models")
    args = parser.parse_args()
    
    if args.random_split:
        out_path = "held_out_testset/held_out_predictions_pgb_random_split.csv"
    else:
        out_path = "held_out_testset/held_out_predictions_pgb_single.csv"
        
    run_inference(output_path=out_path, random_split=args.random_split)
