"""Evaluation – Section 12.

Metrics: PR-AUC, F1, MCC, Precision, Recall, Accuracy, confusion matrix.
Threshold tuning on validation set.
"""

import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
)


def collect_predictions(model, loader, device):
    """Run model on a dataloader and collect all predictions/labels."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                            for k, v in batch.items()}
            labels = batch_device.pop("label")
            _ = batch_device.pop("sample_weight", None)
            
            logits = model(batch_device)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid

    return all_probs, all_labels


def tune_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    step: float = 0.001,
    tnr_floor: float = 0.0,   # 0.0 means no floor (old behaviour, backward-compat)
):
    """
    Sweep thresholds on a validation set.

    If tnr_floor > 0 (e.g. 0.97):
        - Candidate thresholds are those where TNR >= tnr_floor.
        - Among candidates, pick the one that maximises val MCC.
        - If no threshold satisfies the floor, relax to the threshold
          with the highest TNR (most conservative available).
    If tnr_floor == 0:
        - Pick the threshold that maximises val F1  (old behaviour).
    """
    thresholds = np.arange(step, 1.0, step)
    labels_int = labels.astype(int)

    if tnr_floor == 0.0:
        # ── legacy F1 mode ───────────────────────────────────────────────
        best_f1 = -1.0
        best_thresh = 0.5
        for t in thresholds:
            preds = (probs >= t).astype(int)
            f1 = f1_score(labels_int, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        return best_thresh, best_f1

    # ── MCC + TNR-floor mode ─────────────────────────────────────────────
    negatives = (labels_int == 0).sum()

    best_mcc = -2.0
    best_thresh = 0.5
    best_tnr_fallback = -1.0
    best_thresh_fallback = 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn = int(((preds == 0) & (labels_int == 0)).sum())
        tnr = tn / negatives if negatives > 0 else 1.0

        # Track best TNR seen (for the fallback case)
        if tnr > best_tnr_fallback:
            best_tnr_fallback = tnr
            best_thresh_fallback = t

        if tnr < tnr_floor:
            continue  # does not satisfy the floor

        mcc = matthews_corrcoef(labels_int, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t

    if best_mcc == -2.0:
        # No threshold satisfied the floor — use the most conservative one
        return best_thresh_fallback, 0.0

    return best_thresh, best_mcc


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float):
    """Compute full metric suite at a given threshold."""
    preds = (probs >= threshold).astype(int)
    labels_int = labels.astype(int)

    pr_auc = average_precision_score(labels_int, probs)
    f1 = f1_score(labels_int, preds, zero_division=0)
    mcc = matthews_corrcoef(labels_int, preds)
    precision = precision_score(labels_int, preds, zero_division=0)
    recall = recall_score(labels_int, preds, zero_division=0)
    accuracy = accuracy_score(labels_int, preds)
    cm = confusion_matrix(labels_int, preds).tolist()

    return {
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "mcc": float(mcc),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "threshold": float(threshold),
        "confusion_matrix": cm,
    }


def evaluate_model(model, val_loader, test_loader, device, config):
    """Full evaluation: tune threshold on val, report on test.
    
    Returns:
        val_metrics, test_metrics, best_threshold
    """
    # Validation
    val_probs, val_labels = collect_predictions(model, val_loader, device)
    tnr_floor = getattr(config, "pgb_tnr_floor", 0.0) if getattr(config, "use_pgb_head", False) else 0.0
    best_thresh, _ = tune_threshold(val_probs, val_labels, config.threshold_step, tnr_floor=tnr_floor)
    val_metrics = compute_metrics(val_probs, val_labels, best_thresh)

    # Test (using val-tuned threshold)
    test_probs, test_labels = collect_predictions(model, test_loader, device)
    test_metrics = compute_metrics(test_probs, test_labels, best_thresh)

    return val_metrics, test_metrics, best_thresh


def save_metrics(metrics: dict, filepath: str):
    """Save metrics dict to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)


def get_val_pr_auc(model, val_loader, device):
    """Quick PR-AUC computation for scheduler/early stopping."""
    probs, labels = collect_predictions(model, val_loader, device)
    return average_precision_score(labels.astype(int), probs)
