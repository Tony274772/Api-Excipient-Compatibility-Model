"""5-fold cross-validation harness – Section 12.

Reuses the same Butina-cluster group assignments from the data split.
"""

import json
import os
import copy

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold

import torch

from src.config import Config
from src.utils import seed_everything, get_device
from src.dataset import CompatibilityDataset, collate_fn, build_dataloaders
from src.evaluate import evaluate_model, compute_metrics, tune_threshold, collect_predictions, save_metrics
from src.train import train
from src.model import CompatibilityModel


def _morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def butina_cluster(unique_smiles, dist_thresh=0.15):
    fps = [_morgan_fp(s) for s in unique_smiles]
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])
    clusters = Butina.ClusterData(dists, n, dist_thresh, isDistData=True)
    smiles_to_cluster = {}
    for cidx, members in enumerate(clusters):
        for midx in members:
            smiles_to_cluster[unique_smiles[midx]] = cidx
    return smiles_to_cluster


def build_encoder(config, device):
    """Build an encoder instance from config."""
    from src.encoders import ENCODER_REGISTRY
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


def cross_validate(config: Config, n_folds: int = 5):
    """Run n-fold cross-validation with Butina-cluster grouping."""
    seed_everything(config.seed)
    device = get_device(config.device)

    # Load full dataset
    raw = pd.read_csv(os.path.join(config.data_dir, "start_dataset.csv"))

    # Butina clustering
    unique_apis = raw["API_Smiles"].unique().tolist()
    smiles_to_cluster = butina_cluster(unique_apis)
    raw["cluster_id"] = raw["API_Smiles"].map(smiles_to_cluster)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)

    all_val_metrics = []
    all_test_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(
        sgkf.split(raw, y=raw["Outcome1"], groups=raw["cluster_id"])
    ):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_i + 1}/{n_folds}")
        print(f"{'='*60}")

        # For CV, we split train_idx further into train/val (80/20 of train)
        train_fold = raw.iloc[train_idx].reset_index(drop=True)
        test_fold = raw.iloc[test_idx].reset_index(drop=True)

        # Split train_fold into actual train + val (stratified, same group constraint)
        inner_sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=config.seed + fold_i)
        inner_train_idx, inner_val_idx = next(
            inner_sgkf.split(train_fold, y=train_fold["Outcome1"], groups=train_fold["cluster_id"])
        )

        actual_train = train_fold.iloc[inner_train_idx].drop(columns=["cluster_id"]).reset_index(drop=True)
        actual_val = train_fold.iloc[inner_val_idx].drop(columns=["cluster_id"]).reset_index(drop=True)
        actual_test = test_fold.drop(columns=["cluster_id"]).reset_index(drop=True)

        # Save fold CSVs to temp locations
        fold_dir = os.path.join(config.metrics_dir, f"fold_{fold_i}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_train_csv = os.path.join(fold_dir, "train.csv")
        fold_val_csv = os.path.join(fold_dir, "val.csv")
        fold_test_csv = os.path.join(fold_dir, "test.csv")
        actual_train.to_csv(fold_train_csv, index=False)
        actual_val.to_csv(fold_val_csv, index=False)
        actual_test.to_csv(fold_test_csv, index=False)

        # Update config for this fold
        fold_config = copy.deepcopy(config)
        fold_config.train_csv = fold_train_csv
        fold_config.val_csv = fold_val_csv
        fold_config.test_csv = fold_test_csv
        fold_config.checkpoint_dir = os.path.join(config.checkpoint_dir, f"fold_{fold_i}")

        # Compute positive prior
        fold_config.positive_prior = actual_train["Outcome1"].mean()

        # Build encoder, model, dataloaders
        encoder = build_encoder(fold_config, device)
        if not encoder.is_sequence_capable:
            fold_config.fusion = "concat"

        model = CompatibilityModel(fold_config, encoder)
        model.init_bias(fold_config.positive_prior)
        model.to(device)

        from src.dataset import CompatibilityDataset, collate_fn
        from torch.utils.data import DataLoader, WeightedRandomSampler

        train_ds = CompatibilityDataset(
            fold_train_csv, config.api_descriptors_path,
            config.excipient_descriptors_path, config.descriptor_norm_stats_path,
        )
        val_ds = CompatibilityDataset(
            fold_val_csv, config.api_descriptors_path,
            config.excipient_descriptors_path, config.descriptor_norm_stats_path,
        )
        test_ds = CompatibilityDataset(
            fold_test_csv, config.api_descriptors_path,
            config.excipient_descriptors_path, config.descriptor_norm_stats_path,
        )

        # Balanced sampler
        train_sampler = None
        train_shuffle = True
        if fold_config.use_balanced_sampler:
            labels = train_ds.df["Outcome1"].values
            class_counts = np.bincount(labels.astype(int))
            weights = 1.0 / class_counts[labels.astype(int)]
            train_sampler = WeightedRandomSampler(weights.tolist(), len(train_ds), replacement=True)
            train_shuffle = False

        train_loader = DataLoader(train_ds, batch_size=fold_config.batch_size,
                                  shuffle=train_shuffle, sampler=train_sampler,
                                  collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=fold_config.batch_size,
                                shuffle=False, collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=fold_config.batch_size,
                                 shuffle=False, collate_fn=collate_fn, num_workers=0)

        # Train
        model, history = train(model, train_loader, val_loader, fold_config, device)

        # Evaluate
        val_metrics, test_metrics, threshold = evaluate_model(
            model, val_loader, test_loader, device, fold_config
        )

        print(f"\nFold {fold_i + 1} Val:  PR-AUC={val_metrics['pr_auc']:.4f}, F1={val_metrics['f1']:.4f}, MCC={val_metrics['mcc']:.4f}")
        print(f"Fold {fold_i + 1} Test: PR-AUC={test_metrics['pr_auc']:.4f}, F1={test_metrics['f1']:.4f}, MCC={test_metrics['mcc']:.4f}")

        all_val_metrics.append(val_metrics)
        all_test_metrics.append(test_metrics)

        # Save fold metrics
        save_metrics(val_metrics, os.path.join(fold_dir, "val_metrics.json"))
        save_metrics(test_metrics, os.path.join(fold_dir, "test_metrics.json"))

        # Clean up GPU memory
        del model, encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate CV results
    cv_summary = {}
    for key in ["pr_auc", "f1", "mcc", "precision", "recall", "accuracy"]:
        vals = [m[key] for m in all_val_metrics]
        test_vals = [m[key] for m in all_test_metrics]
        cv_summary[f"val_{key}_mean"] = float(np.mean(vals))
        cv_summary[f"val_{key}_std"] = float(np.std(vals))
        cv_summary[f"test_{key}_mean"] = float(np.mean(test_vals))
        cv_summary[f"test_{key}_std"] = float(np.std(test_vals))

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    for key in ["pr_auc", "f1", "mcc"]:
        print(f"  Val  {key}: {cv_summary[f'val_{key}_mean']:.4f} ± {cv_summary[f'val_{key}_std']:.4f}")
        print(f"  Test {key}: {cv_summary[f'test_{key}_mean']:.4f} ± {cv_summary[f'test_{key}_std']:.4f}")

    save_metrics(cv_summary, os.path.join(config.metrics_dir, "cv_metrics.json"))

    return cv_summary


if __name__ == "__main__":
    config = Config()
    config.resolve_paths()
    cross_validate(config)
