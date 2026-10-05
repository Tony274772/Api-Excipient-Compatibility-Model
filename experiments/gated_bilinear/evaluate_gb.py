"""Evaluation module for the Gated Bilinear architecture.

Implements Section 3.6 of the instructions:
- Screen cut: Highest threshold giving val recall ≥ 0.75
- Main threshold: Max val MCC with True Negative Rate (TNR) ≥ 0.97
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def collect_gb_predictions(model, loader, device):
    """Run model on a DataLoader and collect all predictions/labels."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # The model expects the entire batch dict, and handles device placement internally
            # but we need to move it here as it was doing in main train loop
            logits = model(batch)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid

    return all_probs, all_labels


def _compute_tnr(labels: np.ndarray, preds: np.ndarray) -> float:
    """Compute True Negative Rate (Specificity)."""
    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    elif cm.shape == (1, 1):
        # All one class
        if labels[0] == 0:
            return 1.0  # all TN
        else:
            return 0.0  # no negatives
    return 0.0


def tune_thresholds_gb(probs: np.ndarray, labels: np.ndarray, tnr_floor: float = 0.97, min_recall: float = 0.75):
    """Find main threshold (MCC with TNR constraint) and screen cut threshold.
    
    Args:
        probs: Model probabilities.
        labels: True labels.
        tnr_floor: Minimum TNR required for the main threshold.
        min_recall: Minimum recall required for the screen cut threshold.
        
    Returns:
        main_threshold, screen_cut_threshold
    """
    best_mcc = -1.1
    best_thresh = 0.5
    
    screen_thresh = 0.5
    best_screen_thresh_val = -1.0 # want the highest threshold that meets recall

    thresholds = np.arange(0.001, 1.0, 0.001)
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        
        # Screen cut logic
        rec = recall_score(labels, preds, zero_division=0)
        if rec >= min_recall:
            if t > best_screen_thresh_val:
                best_screen_thresh_val = t
                screen_thresh = t
                
        # Main threshold logic
        tnr = _compute_tnr(labels, preds)
        if tnr >= tnr_floor:
            mcc = matthews_corrcoef(labels, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = t

    # If no threshold met the TNR floor, fallback to max MCC without constraint
    if best_mcc == -1.1:
        for t in thresholds:
            preds = (probs >= t).astype(int)
            mcc = matthews_corrcoef(labels, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = t

    return best_thresh, screen_thresh


def compute_metrics_gb(probs: np.ndarray, labels: np.ndarray, threshold: float):
    """Compute full metric suite at a given threshold."""
    preds = (probs >= threshold).astype(int)
    labels_int = labels.astype(int)

    pr_auc = average_precision_score(labels_int, probs)
    try:
        roc_auc = roc_auc_score(labels_int, probs)
    except ValueError:
        roc_auc = 0.0 # Only one class present

    f1 = f1_score(labels_int, preds, zero_division=0)
    mcc = matthews_corrcoef(labels_int, preds)
    precision = precision_score(labels_int, preds, zero_division=0)
    recall = recall_score(labels_int, preds, zero_division=0)
    accuracy = accuracy_score(labels_int, preds)
    cm = confusion_matrix(labels_int, preds).tolist()

    # Extract TP and FP
    tp = 0
    fp = 0
    if len(cm) == 2 and len(cm[0]) == 2:
        tn_val, fp, fn, tp = confusion_matrix(labels_int, preds).ravel()

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "f1": float(f1),
        "mcc": float(mcc),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "confusion_matrix": cm,
    }


def evaluate_gb_model(model, val_loader, test_loader, device, config):
    """Full evaluation: tune threshold on val, report on test.
    
    Returns:
        val_metrics, test_metrics, best_thresh, screen_thresh
    """
    # Validation
    val_probs, val_labels = collect_gb_predictions(model, val_loader, device)
    best_thresh, screen_thresh = tune_thresholds_gb(
        val_probs, val_labels, 
        tnr_floor=config.gb_tnr_floor,
        min_recall=config.gb_screen_min_recall
    )
    val_metrics = compute_metrics_gb(val_probs, val_labels, best_thresh)

    # Test (using val-tuned threshold)
    test_probs, test_labels = collect_gb_predictions(model, test_loader, device)
    test_metrics = compute_metrics_gb(test_probs, test_labels, best_thresh)

    # We can also compute metrics at the screen cut threshold if desired,
    # but the primary evaluation is at the main threshold.
    
    return val_metrics, test_metrics, best_thresh, screen_thresh


def save_gb_metrics(metrics: dict, filepath: str):
    """Save metrics dict to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)


def get_val_pr_auc_gb(model, val_loader, device):
    """Quick PR-AUC computation for scheduler/early stopping."""
    probs, labels = collect_gb_predictions(model, val_loader, device)
    return average_precision_score(labels.astype(int), probs)


def evaluate_heldout_gb(
    model,
    heldout_loader,
    device,
    threshold: float,
    model_name: str,
    output_csv_path: str = "held_out_testset/held_out_predictions_gatedbilinear.csv",
    heldout_source_csv: str = "held_out_testset/held_out_test_set.csv",
) -> tuple[dict, pd.DataFrame]:
    """Run inference on the 24-pair held-out set and update the accumulator CSV.
    
    Preserves all existing model columns in output_csv_path.
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in heldout_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    # Compute metrics on held-out set
    metrics = compute_metrics_gb(probs, labels, threshold)

    # Load or initialize result dataframe
    os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else ".", exist_ok=True)
    if os.path.isfile(output_csv_path):
        result_df = pd.read_csv(output_csv_path)
    else:
        orig_df = pd.read_csv(heldout_source_csv)
        base_cols = [
            "Api_name", "Excipient_name", "API_CID", "Excipient_CID",
            "API_Smiles", "Excipient_Smiles", "ground_truth"
        ]
        cols_to_keep = [c for c in base_cols if c in orig_df.columns]
        result_df = orig_df[cols_to_keep].copy()
        if "ground_truth" not in result_df.columns and "Outcome1" in orig_df.columns:
            result_df["ground_truth"] = orig_df["Outcome1"]

    keys_to_store = [f"gated_bilinear_{model_name}" if not model_name.startswith("gated_bilinear_") else model_name]

    for k in keys_to_store:
        result_df[f"{k}_logit"] = logits
        result_df[f"{k}_probability"] = probs
        result_df[f"{k}_prediction"] = preds
        result_df[f"{k}_threshold"] = float(threshold)

    try:
        result_df.to_csv(output_csv_path, index=False)
        print(f"Held-out predictions saved/updated in: {output_csv_path}")
    except PermissionError:
        fallback = output_csv_path.replace(".csv", "_fallback.csv")
        result_df.to_csv(fallback, index=False)
        print(f"WARNING: Permission denied for {output_csv_path}. Saved to fallback: {fallback}")

    return metrics, result_df
