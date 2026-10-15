import os
import copy
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.config import Config
from src.utils import seed_everything, get_device
from src.dataset import PGBDataset, pgb_collate_fn, PGBNormStats
from src.pgb_prior import ExcipientPriorTable, build_leave_cluster_out_prior_vectors
from src.model import pgb_model_from_config
from src.train import pgb_train
from src.evaluate import evaluate_model, save_metrics
from src.cross_validate import build_encoder


def pgb_single_run(config: Config):
    """
    Trains the PGBCompatibilityModel once on a single Train/Val split
    and tests it on the 24-pairs held-out test set.
    """
    seed_everything(config.seed)
    device = get_device(config.device)

    # 1. Load Data
    raw = pd.read_csv(os.path.join(config.data_dir, "start_dataset.csv"))
    
    # We do a single 80/20 train/val split using clusters
    if getattr(config, "split_type", "cluster") == "random":
        train_fold, val_fold = train_test_split(raw, test_size=0.2, random_state=config.seed, stratify=raw["Outcome1"])
    else:
        from rdkit.ML.Cluster import Butina
        from src.cross_validate import butina_cluster
        
        unique_apis = raw["API_Smiles"].unique().tolist()
        smiles_to_cluster = butina_cluster(unique_apis)
        raw["cluster_id"] = raw["API_Smiles"].map(smiles_to_cluster)
        
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed)
        train_idx, val_idx = next(gss.split(raw, y=raw["Outcome1"], groups=raw["cluster_id"]))
        
        train_fold = raw.iloc[train_idx].reset_index(drop=True)
        val_fold   = raw.iloc[val_idx].reset_index(drop=True)

    # The test fold is strictly the 24-pairs held-out test set
    test_fold = pd.read_csv("held_out_testset/held_out_test_set.csv")
    if "ground_truth" in test_fold.columns:
        test_fold = test_fold.rename(columns={"ground_truth": "Outcome1"})

    run_dir = os.path.join(config.metrics_dir, "single_run")
    os.makedirs(run_dir, exist_ok=True)
    
    train_csv = os.path.join(run_dir, "train.csv")
    val_csv   = os.path.join(run_dir, "val.csv")
    test_csv  = os.path.join(run_dir, "test.csv")
    
    train_fold.to_csv(train_csv, index=False)
    val_fold.to_csv(val_csv,   index=False)
    test_fold.to_csv(test_csv,  index=False)

    run_config = copy.deepcopy(config)
    run_config.train_csv = train_csv
    run_config.val_csv   = val_csv
    run_config.test_csv  = test_csv
    run_config.checkpoint_dir = os.path.join(config.checkpoint_dir, "single_run")
    run_config.positive_prior = float(train_fold["Outcome1"].mean())
    run_config.use_balanced_sampler = False
    run_config.loss = "asym_focal"

    os.makedirs(run_config.checkpoint_dir, exist_ok=True)

    # 2. Build encoder
    encoder = build_encoder(run_config, device)
    if not encoder.is_sequence_capable:
        run_config.fusion = "concat"

    # 3. Build PGB norm stats from unique training SMILES only
    all_train_smiles = list(set(
        train_fold["API_Smiles"].dropna().tolist() +
        train_fold["Excipient_Smiles"].dropna().tolist()
    ))
    pgb_norm = PGBNormStats.from_smiles_list(all_train_smiles)
    pgb_norm.save(os.path.join(run_config.checkpoint_dir, "pgb_norm_stats.npz"))

    # 4. Build excipient prior from training rows
    prior_table = ExcipientPriorTable()
    prior_table.fit(train_fold, leave_out_cluster_id=None)
    prior_table.save(os.path.join(run_config.checkpoint_dir, "prior_table.json"))
    
    cluster_col = "cluster_id" if "cluster_id" in train_fold.columns else None
    train_priors = build_leave_cluster_out_prior_vectors(train_fold, cluster_col=cluster_col)

    fv_enc = encoder if run_config.encoder == "fixed_vector" else None

    # 5. Datasets & Loaders
    train_ds = PGBDataset(train_csv, run_config.encoder,
                          getattr(run_config, "fixed_vector_source", "maccs"),
                          prior_table, pgb_norm, fixed_vector_encoder=fv_enc, row_priors=train_priors)
    val_ds   = PGBDataset(val_csv,   run_config.encoder,
                          getattr(run_config, "fixed_vector_source", "maccs"),
                          prior_table, pgb_norm, fixed_vector_encoder=fv_enc)
    test_ds  = PGBDataset(test_csv,  run_config.encoder,
                          getattr(run_config, "fixed_vector_source", "maccs"),
                          prior_table, pgb_norm, fixed_vector_encoder=fv_enc)

    train_loader = DataLoader(train_ds, batch_size=run_config.pgb_batch_size,
                              shuffle=True, collate_fn=pgb_collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=run_config.pgb_batch_size,
                              shuffle=False, collate_fn=pgb_collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=run_config.pgb_batch_size, # (All 24 will fit in 1 batch easily)
                              shuffle=False, collate_fn=pgb_collate_fn, num_workers=0)

    # 6. Build model & Train
    model = pgb_model_from_config(run_config, encoder)
    model.to(device)

    model, history = pgb_train(model, train_loader, val_loader, run_config, device)

    # 7. Evaluate on held-out test set
    val_metrics, test_metrics, threshold = evaluate_model(
        model, val_loader, test_loader, device, run_config
    )
    
    save_metrics(val_metrics, os.path.join(run_dir, "val_metrics.json"))
    save_metrics(test_metrics, os.path.join(run_dir, "test_metrics.json"))

    print(f"\nSingle Run Val:  PR-AUC={val_metrics['pr_auc']:.4f}, F1={val_metrics['f1']:.4f}, MCC={val_metrics['mcc']:.4f}")
    print(f"Single Run Test: PR-AUC={test_metrics['pr_auc']:.4f}, F1={test_metrics['f1']:.4f}, MCC={test_metrics['mcc']:.4f}")
    print(f"  threshold={threshold:.3f}")

    # Clean up GPU memory
    del model, encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_metrics, test_metrics
