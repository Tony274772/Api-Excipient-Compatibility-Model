"""Standalone training pipeline for the fixed_vector_maccs_concat_weighted_bce model.
(Updated to exactly replicate the original pipeline, including RDKit descriptors)."""

import os
import time
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "RingCount", "FractionCSP3",
    "HeavyAtomCount", "fr_NH2", "fr_NH1", "fr_ester", "fr_amide", "fr_aldehyde",
    "fr_ketone", "fr_phenol", "fr_ether", "fr_epoxide", "fr_halogen", "fr_COO"
]

# -------------------------------------------------------------------------
# 1. Configuration (Hardcoded for fixed_vector_maccs_concat_weighted_bce)
# -------------------------------------------------------------------------
class Config:
    data_dir = "data"
    train_csv = "data/cluster_split/train.csv"
    val_csv = "data/cluster_split/val.csv"
    test_csv = "data/cluster_split/test.csv"
    maccs_path = "data/maccs_keys.csv"
    
    # RDKit Descriptor paths
    api_descriptors_path = "data/api_descriptors.csv"
    excipient_descriptors_path = "data/excipient_descriptors.csv"
    descriptor_norm_stats_path = "models/descriptor_norm_stats.json"
    num_descriptors = 21
    desc_proj_dim = 24
    desc_dropout = 0.15
    
    # Checkpoint and metrics directories
    checkpoint_dir = "checkpoints/fixed_vector_maccs_concat_weighted_bce"
    metrics_dir = "metrics/fixed_vector_maccs_concat_weighted_bce"
    
    # Model architecture
    proj_dim = 128
    proj_dropout = 0.15
    clf_hidden_dim = 128
    clf_hidden_dim_2 = 64
    clf_dropout_1 = 0.5
    clf_dropout_2 = 0.4
    
    # Training
    lr = 1.5e-4
    weight_decay = 8e-4
    grad_clip_norm = 1.0
    batch_size = 64
    max_epochs = 150
    early_stop_patience = 6
    early_stop_min_delta = 0.002
    lr_patience = 8
    lr_factor = 0.5
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold_step = 0.001

# -------------------------------------------------------------------------
# 2. Utils
# -------------------------------------------------------------------------
def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)

def normalize_descriptors(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / (std + 1e-8)

# -------------------------------------------------------------------------
# 3. Dataset & DataLoader (Dependencies: CSVs, MACCS, Descriptors, Norm Stats)
# -------------------------------------------------------------------------
class StandaloneCompatibilityDataset(Dataset):
    def __init__(self, csv_path: str, maccs_path: str, api_desc_path: str, 
                 exc_desc_path: str, norm_stats_path: str, sample_weight: float = 1.0):
        self.df = pd.read_csv(csv_path)
        self.sample_weight = sample_weight
        
        # --- 1. Load MACCS keys ---
        maccs_df = pd.read_csv(maccs_path)
        cols = list(maccs_df.columns)
        smiles_col = next((c for c in cols if "smiles" in c.lower()), None)
        cid_col = next((c for c in cols if "cid" in c.lower()), None)
        exclude_cols = set([c for c in [smiles_col, cid_col] if c is not None])
        feature_cols = [c for c in cols if c not in exclude_cols]
        if not exclude_cols:
            key_col = cols[0]
            feature_cols = cols[1:]
        else:
            key_col = smiles_col or cid_col
            
        self.maccs_dim = len(feature_cols)
        self.maccs_dict = {}
        for _, row in maccs_df.iterrows():
            vec = row[feature_cols].values.astype(np.float32)
            if smiles_col: self.maccs_dict[str(row[smiles_col])] = vec
            if cid_col: self.maccs_dict[str(row[cid_col])] = vec
            if not smiles_col and not cid_col: self.maccs_dict[str(row[key_col])] = vec
                
        self._smiles_to_cid = {}
        if "API_Smiles" in self.df.columns and "API_CID" in self.df.columns:
            for _, r in self.df[["API_Smiles", "API_CID"]].dropna().iterrows():
                self._smiles_to_cid[str(r["API_Smiles"])] = str(int(r["API_CID"]) if isinstance(r["API_CID"], (int, float)) else r["API_CID"])
        if "Excipient_Smiles" in self.df.columns and "Excipient_CID" in self.df.columns:
            for _, r in self.df[["Excipient_Smiles", "Excipient_CID"]].dropna().iterrows():
                self._smiles_to_cid[str(r["Excipient_Smiles"])] = str(int(r["Excipient_CID"]) if isinstance(r["Excipient_CID"], (int, float)) else r["Excipient_CID"])

        # --- 2. Load Descriptors ---
        api_desc_df = pd.read_csv(api_desc_path)
        exc_desc_df = pd.read_csv(exc_desc_path)

        self.api_desc_map = {}
        for _, row in api_desc_df.iterrows():
            self.api_desc_map[row["API_CID"]] = np.array([row[n] for n in DESCRIPTOR_NAMES], dtype=np.float32)

        self.exc_desc_map = {}
        for _, row in exc_desc_df.iterrows():
            self.exc_desc_map[row["Excipient_CID"]] = np.array([row[n] for n in DESCRIPTOR_NAMES], dtype=np.float32)

        with open(norm_stats_path, "r") as f:
            stats = json.load(f)
        self.api_mean = np.array(stats["api_mean"], dtype=np.float32)
        self.api_std = np.array(stats["api_std"], dtype=np.float32)
        self.exc_mean = np.array(stats["exc_mean"], dtype=np.float32)
        self.exc_std = np.array(stats["exc_std"], dtype=np.float32)

    def __len__(self):
        return len(self.df)
        
    def _get_maccs(self, smiles):
        if smiles in self.maccs_dict:
            return self.maccs_dict[smiles]
        elif smiles in self._smiles_to_cid and self._smiles_to_cid[smiles] in self.maccs_dict:
            return self.maccs_dict[self._smiles_to_cid[smiles]]
        return np.zeros(self.maccs_dim, dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        api_smiles = str(row["API_Smiles"])
        exc_smiles = str(row["Excipient_Smiles"]) if pd.notna(row.get("Excipient_Smiles")) else ""
        
        # MACCS
        api_maccs = self._get_maccs(api_smiles)
        exc_maccs = self._get_maccs(exc_smiles)
        
        # Descriptors
        api_desc_raw = self.api_desc_map.get(row["API_CID"], np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32))
        exc_desc_raw = self.exc_desc_map.get(row["Excipient_CID"], np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32))

        api_desc = normalize_descriptors(api_desc_raw, self.api_mean, self.api_std)
        exc_desc = normalize_descriptors(exc_desc_raw, self.exc_mean, self.exc_std)

        api_desc = np.nan_to_num(api_desc, nan=0.0)
        exc_desc = np.nan_to_num(exc_desc, nan=0.0)
        
        exc_available = 1.0 if exc_smiles != "" else 0.0
        label = float(row["Outcome1"])
        
        return {
            "api_maccs": torch.tensor(api_maccs, dtype=torch.float32),
            "exc_maccs": torch.tensor(exc_maccs, dtype=torch.float32),
            "api_desc": torch.tensor(api_desc, dtype=torch.float32),
            "exc_desc": torch.tensor(exc_desc, dtype=torch.float32),
            "exc_available": torch.tensor(exc_available, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "sample_weight": torch.tensor(self.sample_weight, dtype=torch.float32),
        }

def collate_fn(batch: list[dict]) -> dict:
    return {
        "api_maccs": torch.stack([b["api_maccs"] for b in batch]),
        "exc_maccs": torch.stack([b["exc_maccs"] for b in batch]),
        "api_desc": torch.stack([b["api_desc"] for b in batch]),
        "exc_desc": torch.stack([b["exc_desc"] for b in batch]),
        "exc_available": torch.stack([b["exc_available"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "sample_weight": torch.stack([b["sample_weight"] for b in batch]),
    }

def build_dataloaders(config):
    if not os.path.exists(config.train_csv):
        config.train_csv = "data/train.csv"
        config.val_csv = "data/val.csv"
        config.test_csv = "data/test.csv"
        
    ds_kwargs = {
        "maccs_path": config.maccs_path,
        "api_desc_path": config.api_descriptors_path,
        "exc_desc_path": config.excipient_descriptors_path,
        "norm_stats_path": config.descriptor_norm_stats_path,
    }
    train_ds = StandaloneCompatibilityDataset(config.train_csv, **ds_kwargs)
    val_ds = StandaloneCompatibilityDataset(config.val_csv, **ds_kwargs)
    test_ds = StandaloneCompatibilityDataset(config.test_csv, **ds_kwargs)
    
    labels = train_ds.df["Outcome1"].values
    class_counts = np.bincount(labels.astype(int))
    weights = 1.0 / class_counts[labels.astype(int)]
    train_sampler = WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(train_ds),
        replacement=True,
    )
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

# -------------------------------------------------------------------------
# 4. Model Architecture
# -------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class DescriptorProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 21, desc_proj_dim: int = 24, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, desc_proj_dim),
            nn.LayerNorm(desc_proj_dim),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class StandaloneCompatibilityModel(nn.Module):
    def __init__(self, config, maccs_dim: int):
        super().__init__()
        self.config = config
        
        # Projection heads for MACCS keys
        self.api_proj = ProjectionHead(maccs_dim, config.proj_dim, config.proj_dropout)
        self.exc_proj = ProjectionHead(maccs_dim, config.proj_dim, config.proj_dropout)
        
        # Projection heads for Descriptors
        self.api_desc_proj = DescriptorProjectionHead(config.num_descriptors, config.desc_proj_dim, config.desc_dropout)
        self.exc_desc_proj = DescriptorProjectionHead(config.num_descriptors, config.desc_proj_dim, config.desc_dropout)
        
        # Dimensions
        h_api_dim = config.proj_dim + config.desc_proj_dim        # 128 + 24 = 152
        h_exc_dim = config.proj_dim + config.desc_proj_dim + 1    # 128 + 24 + 1 = 153
        
        pair_dim = h_api_dim + h_exc_dim + h_api_dim + h_api_dim  # 152 + 153 + 152 + 152 = 609
        
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, config.clf_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.clf_dropout_1),
            nn.Linear(config.clf_hidden_dim, config.clf_hidden_dim_2),
            nn.GELU(),
            nn.Dropout(config.clf_dropout_2),
            nn.Linear(config.clf_hidden_dim_2, 1),
        )

    def init_bias(self, positive_prior: float):
        if positive_prior > 0 and positive_prior < 1:
            bias_val = math.log(positive_prior / (1 - positive_prior))
            self.classifier[-1].bias.data.fill_(bias_val)

    def forward(self, batch: dict) -> torch.Tensor:
        api_maccs = batch["api_maccs"]
        exc_maccs = batch["exc_maccs"]
        api_desc = batch["api_desc"]
        exc_desc = batch["exc_desc"]
        exc_available = batch["exc_available"]
        
        # Structural projections
        h_api_struct = self.api_proj(api_maccs)  # [B, 128]
        h_exc_struct = self.exc_proj(exc_maccs)  # [B, 128]
        
        # Descriptor projections
        d_api = self.api_desc_proj(api_desc)     # [B, 24]
        d_exc = self.exc_desc_proj(exc_desc)     # [B, 24]
        
        h_api = torch.cat([h_api_struct, d_api], dim=1)                               # [B, 152]
        h_exc = torch.cat([h_exc_struct, d_exc, exc_available.unsqueeze(1)], dim=1)   # [B, 153]
        
        # Interaction terms
        h_exc_core = h_exc[:, :h_api.shape[1]]  # exclude the exc_available flag for math
        interaction = h_api * h_exc_core
        difference = torch.abs(h_api - h_exc_core)
        
        # Pair vector
        pair_vector = torch.cat([h_api, h_exc, interaction, difference], dim=1)  # [B, 609]
        
        # Classifier
        logits = self.classifier(pair_vector)
        return logits.squeeze(1)

# -------------------------------------------------------------------------
# 5. Loss Function
# -------------------------------------------------------------------------
def weighted_bce_loss(logits, targets, sample_weight=None, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()

def compute_pos_weight(train_loader) -> torch.Tensor:
    labels = train_loader.dataset.df["Outcome1"].values
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    return torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)

# -------------------------------------------------------------------------
# 6. Evaluation Utils
# -------------------------------------------------------------------------
def collect_predictions(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch_device = {
                "api_maccs": batch["api_maccs"].to(device),
                "exc_maccs": batch["exc_maccs"].to(device),
                "api_desc": batch["api_desc"].to(device),
                "exc_desc": batch["exc_desc"].to(device),
                "exc_available": batch["exc_available"].to(device),
            }
            logits = model(batch_device)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"])
            
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = 1 / (1 + np.exp(-all_logits))
    return all_probs, all_labels

def get_val_pr_auc(model, val_loader, device):
    probs, labels = collect_predictions(model, val_loader, device)
    return average_precision_score(labels.astype(int), probs)

def evaluate_model(model, val_loader, test_loader, device, config):
    val_probs, val_labels = collect_predictions(model, val_loader, device)
    
    best_f1 = -1.0
    best_thresh = 0.5
    for t in np.arange(config.threshold_step, 1.0, config.threshold_step):
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    def compute_metrics(probs, labels, threshold):
        preds = (probs >= threshold).astype(int)
        labels_int = labels.astype(int)
        return {
            "pr_auc": float(average_precision_score(labels_int, probs)),
            "f1": float(f1_score(labels_int, preds, zero_division=0)),
            "mcc": float(matthews_corrcoef(labels_int, preds)),
            "precision": float(precision_score(labels_int, preds, zero_division=0)),
            "recall": float(recall_score(labels_int, preds, zero_division=0)),
            "accuracy": float(accuracy_score(labels_int, preds)),
            "threshold": float(threshold),
            "confusion_matrix": confusion_matrix(labels_int, preds).tolist(),
        }
        
    val_metrics = compute_metrics(val_probs, val_labels, best_thresh)
    
    test_probs, test_labels = collect_predictions(model, test_loader, device)
    test_metrics = compute_metrics(test_probs, test_labels, best_thresh)
    
    return val_metrics, test_metrics, best_thresh

# -------------------------------------------------------------------------
# 7. Training Loop
# -------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, config, device):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=config.lr_factor, patience=config.lr_patience)
    pos_weight = compute_pos_weight(train_loader).to(device)
    
    best_pr_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_ckpt_path = os.path.join(config.checkpoint_dir, "best_model.pt")
    history = {"train_loss": [], "val_pr_auc": [], "lr": []}
    
    print(f"\nTraining fixed_vector_maccs_concat_weighted_bce (Standalone)")
    print(f"Device: {device}, Epochs: {config.max_epochs}, Batch size: {config.batch_size}")
    
    for epoch in range(1, config.max_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            batch_device = {
                "api_maccs": batch["api_maccs"].to(device),
                "exc_maccs": batch["exc_maccs"].to(device),
                "api_desc": batch["api_desc"].to(device),
                "exc_desc": batch["exc_desc"].to(device),
                "exc_available": batch["exc_available"].to(device),
            }
            labels = batch["label"].to(device)
            sample_weight = batch["sample_weight"].to(device)
            
            optimizer.zero_grad()
            logits = model(batch_device)
            
            loss = weighted_bce_loss(logits, labels, sample_weight, pos_weight)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        avg_loss = total_loss / max(n_batches, 1)
        val_pr_auc = get_val_pr_auc(model, val_loader, device)
        scheduler.step(val_pr_auc)
        current_lr = optimizer.param_groups[0]["lr"]
        
        history["train_loss"].append(avg_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["lr"].append(current_lr)
        
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{config.max_epochs} | "
            f"Loss: {avg_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | "
            f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
        )
        
        if val_pr_auc > best_pr_auc + config.early_stop_min_delta:
            best_pr_auc = val_pr_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_pr_auc": val_pr_auc,
            }, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={config.early_stop_patience})")
                break
                
    print(f"\nRestoring best model from epoch {best_epoch} (PR-AUC={best_pr_auc:.4f})")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, history

# -------------------------------------------------------------------------
# 8. Main Entry Point
# -------------------------------------------------------------------------
def main():
    config = Config()
    seed_everything(config.seed)
    device = get_device(config.device)
    
    print("Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(config)
    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")
    
    config.positive_prior = float(train_loader.dataset.df["Outcome1"].mean())
    print(f"Positive prior (from train): {config.positive_prior:.4f}")
    
    maccs_dim = train_loader.dataset.maccs_dim
    model = StandaloneCompatibilityModel(config, maccs_dim)
    model.init_bias(config.positive_prior)
    model.to(device)
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_trainable:,} trainable")
    
    model, history = train_model(model, train_loader, val_loader, config, device)
    
    val_metrics, test_metrics, best_thresh = evaluate_model(
        model, val_loader, test_loader, device, config
    )
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best threshold: {best_thresh:.4f}")
    print(f"\nValidation:")
    for k, v in val_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  confusion_matrix: {val_metrics['confusion_matrix']}")

    print(f"\nTest:")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  confusion_matrix: {test_metrics['confusion_matrix']}")

    os.makedirs(config.metrics_dir, exist_ok=True)
    with open(os.path.join(config.metrics_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(config.metrics_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(os.path.join(config.metrics_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"\nMetrics saved to {config.metrics_dir}/")

if __name__ == "__main__":
    main()
