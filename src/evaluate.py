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
            batch_device = {
                "api_smiles": batch["api_smiles"],
                "exc_smiles": batch["exc_smiles"],
                "api_desc": batch["api_desc"].to(device),
                "exc_desc": batch["exc_desc"].to(device),
                "exc_available": batch["exc_available"].to(device),
            }
            logits = model(batch_device)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"])

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid

    return all_probs, all_labels


def tune_threshold(probs: np.ndarray, labels: np.ndarray, step: float = 0.001):
    """Sweep thresholds on validation set, pick best F1."""
    best_f1 = -1.0
    best_thresh = 0.5

    thresholds = np.arange(step, 1.0, step)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1


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
    best_thresh, _ = tune_threshold(val_probs, val_labels, config.threshold_step)
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
