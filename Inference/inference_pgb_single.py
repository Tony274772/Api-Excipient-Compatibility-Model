"""
inference_pgb_single.py
Runs inference using the single-run PGB models on the 24-pairs held-out test set
and saves the probability scores to a CSV file.
"""

import os
import sys

# Allow running from inside the inference/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.config import Config
from src.utils import get_device
from src.dataset import PGBDataset, pgb_collate_fn, PGBNormStats
from src.pgb_prior import ExcipientPriorTable, build_leave_cluster_out_prior_vectors
from src.model import pgb_model_from_config
from src.cross_validate import build_encoder

from run_pgb import FAMILIES

def get_best_threshold(metrics_dir: str) -> float:
    path = os.path.join(metrics_dir, "single_run", "val_metrics.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f).get("threshold", 0.5)
    return 0.5

def run_inference():
    device = get_device("auto")
    test_csv = "held_out_testset/held_out_test_set.csv"
    test_df = pd.read_csv(test_csv)
    
    if "ground_truth" in test_df.columns:
        test_df = test_df.rename(columns={"ground_truth": "Outcome1"})

    # Save to a temporary CSV so PGBDataset can load the Outcome1 column
    temp_test_csv = "held_out_testset/temp_held_out.csv"
    test_df.to_csv(temp_test_csv, index=False)
    results_df = test_df.copy()

    for family, spec in FAMILIES.items():
        print(f"Running inference for {family}...")
        config = Config()
        config.use_pgb_head = True
        for k, v in spec.items():
            setattr(config, k, v)
        config.resolve_paths()

        run_config = copy.deepcopy(config)
        run_config.checkpoint_dir = os.path.join(config.checkpoint_dir, "single_run")
        run_config.metrics_dir = os.path.join(config.metrics_dir, "single_run")

        ckpt_path = os.path.join(run_config.checkpoint_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Warning: Checkpoint not found for {family} at {ckpt_path}. Skipping.")
            continue

        # Build encoder
        encoder = build_encoder(run_config, device)
        if not encoder.is_sequence_capable:
            run_config.fusion = "concat"

        fv_enc = encoder if run_config.encoder == "fixed_vector" else None

        # Load norm stats & priors
        pgb_norm = PGBNormStats.load(os.path.join(run_config.checkpoint_dir, "pgb_norm_stats.npz"))
        prior_table = ExcipientPriorTable.load(os.path.join(run_config.checkpoint_dir, "prior_table.json"))

        test_ds = PGBDataset(
            temp_test_csv, 
            run_config.encoder,
            getattr(run_config, "fixed_vector_source", "maccs"),
            prior_table, 
            pgb_norm, 
            fixed_vector_encoder=fv_enc
        )

        test_loader = DataLoader(
            test_ds, 
            batch_size=run_config.pgb_batch_size,
            shuffle=False, 
            collate_fn=pgb_collate_fn, 
            num_workers=0
        )

        model = pgb_model_from_config(run_config, encoder)
        
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

        # Add to dataframe
        results_df[f"prob_pgb_{family}"] = probs

        # Also apply threshold to get binary prediction
        thresh = get_best_threshold(config.metrics_dir)
        results_df[f"pred_pgb_{family}"] = (probs >= thresh).astype(int)
        
        # Clean up memory
        del model, encoder, fv_enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = "held_out_testset/held_out_predictions_pgb_single.csv"
    results_df.to_csv(output_path, index=False)
    
    if os.path.exists(temp_test_csv):
        os.remove(temp_test_csv)
        
    print(f"\nSaved PGB single-run inference results to: {output_path}")

if __name__ == "__main__":
    run_inference()
