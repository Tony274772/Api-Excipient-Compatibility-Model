import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CompatibilityDataset, collate_fn
from ensemble._common import build_model_from_checkpoint
from src.evaluate import collect_predictions, tune_threshold

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

COMPATIBLE_MODELS = [
    "dmpnn_chemprop_cross_attn_global_gated_bce",
    "fixed_vector_mol2vec_concat_asl",
    "fixed_vector_mol2vec_concat_bce",
    "fixed_vector_mol2vec_concat_focal",
    "fixed_vector_mol2vec_concat_weighted_bce",
    "molformer_cross_attn_asl"
]

INCOMPATIBLE_MODELS = [
    "fixed_vector_maccs_concat_weighted_bce",
    "fixed_vector_morgan_concat_bce",
    "fixed_vector_maccs_concat_asl",
    "fixed_vector_maccs_concat_bce",
    "fixed_vector_maccs_concat_focal",
    "fixed_vector_morgan_concat_asl",
    "fixed_vector_pubchemfp_concat_bce",
    "fixed_vector_pubchemfp_concat_weighted_bce",
    "fixed_vector_pubchemfp_concat_focal",
    "fixed_vector_morgan_concat_weighted_bce",
    "fixed_vector_pubchemfp_concat_asl"
]

def compute_per_class_metrics(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)

    # Class 1 (Incompatible)
    pr_auc_1 = average_precision_score(labels, probs)
    f1_1 = f1_score(labels, preds, zero_division=0)
    prec_1 = precision_score(labels, preds, zero_division=0)
    rec_1 = recall_score(labels, preds, zero_division=0)
    correct_1 = int(np.sum((preds == 1) & (labels == 1)))
    total_1 = int(np.sum(labels == 1))
    acc_1 = correct_1 / total_1 if total_1 > 0 else 0.0
    mcc_1 = matthews_corrcoef(labels, preds)
    cm_1 = confusion_matrix(labels, preds).tolist()

    # Class 0 (Compatible)
    labels_0 = 1 - labels
    probs_0 = 1 - probs
    preds_0 = 1 - preds
    pr_auc_0 = average_precision_score(labels_0, probs_0)
    f1_0 = f1_score(labels_0, preds_0, zero_division=0)
    prec_0 = precision_score(labels_0, preds_0, zero_division=0)
    rec_0 = recall_score(labels_0, preds_0, zero_division=0)
    correct_0 = int(np.sum((preds_0 == 1) & (labels_0 == 1)))
    total_0 = int(np.sum(labels_0 == 1))
    acc_0 = correct_0 / total_0 if total_0 > 0 else 0.0
    mcc_0 = matthews_corrcoef(labels_0, preds_0)
    cm_0 = confusion_matrix(labels_0, preds_0).tolist()

    overall_acc = (correct_1 + correct_0) / (total_1 + total_0) if (total_1 + total_0) > 0 else 0.0

    return {
        "overall": {
            "correct": correct_1 + correct_0,
            "total": total_1 + total_0,
            "pr_auc": float(pr_auc_1),
            "f1": float(f1_1),
            "mcc": float(mcc_1),
            "precision": float(prec_1),
            "recall": float(rec_1),
            "accuracy": float(overall_acc),
            "threshold": float(threshold),
            "confusion_matrix": cm_1,
        },
        "incompatible_class_1": {
            "correct": correct_1,
            "total": total_1,
            "pr_auc": float(pr_auc_1),
            "f1": float(f1_1),
            "mcc": float(mcc_1),
            "precision": float(prec_1),
            "recall": float(rec_1),
            "accuracy": float(acc_1),
            "threshold": float(threshold),
            "confusion_matrix": cm_1,
        },
        "compatible_class_0": {
            "correct": correct_0,
            "total": total_0,
            "pr_auc": float(pr_auc_0),
            "f1": float(f1_0),
            "mcc": float(mcc_0),
            "precision": float(prec_0),
            "recall": float(rec_0),
            "accuracy": float(acc_0),
            "threshold": float(threshold),
            "confusion_matrix": cm_0,
        }
    }


def save_csv(dataset_df, probs, preds, output_path):
    df = dataset_df[["API_CID", "Excipient_CID", "Outcome1", "API_Smiles", "Excipient_Smiles"]].copy()
    df.rename(columns={"Outcome1": "ground_truth"}, inplace=True)
    df["probability"] = probs
    df["prediction"] = preds
    df.to_csv(output_path, index=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run from root so data paths work properly
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    val_ds = CompatibilityDataset("data/val.csv", "data/api_descriptors.csv", "data/excipient_descriptors.csv", "models/descriptor_norm_stats.json")
    test_ds = CompatibilityDataset("data/test.csv", "data/api_descriptors.csv", "data/excipient_descriptors.csv", "models/descriptor_norm_stats.json")
    
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    val_df = val_ds.df
    test_df = test_ds.df

    tasks = [
        ("compatible", COMPATIBLE_MODELS),
        ("incompatible", INCOMPATIBLE_MODELS)
    ]

    for category, models in tasks:
        for model_name in models:
            print(f"\nProcessing {category} model: {model_name}")
            try:
                model, config = build_model_from_checkpoint(model_name, device, checkpoints_dir="checkpoints")
                model.eval()

                # Get predictions
                val_probs, val_labels = collect_predictions(model, val_loader, device)
                test_probs, test_labels = collect_predictions(model, test_loader, device)

                # Load previously tuned threshold if it exists
                val_metrics_path = os.path.join(config.metrics_dir, "val_metrics.json")
                if os.path.exists(val_metrics_path):
                    with open(val_metrics_path, "r") as f:
                        old_metrics = json.load(f)
                        best_thresh = old_metrics.get("threshold", 0.5)
                else:
                    best_thresh, _ = tune_threshold(val_probs, val_labels, 0.001)

                val_preds = (val_probs >= best_thresh).astype(int)
                test_preds = (test_probs >= best_thresh).astype(int)

                # Compute metrics
                val_metrics = compute_per_class_metrics(val_probs, val_labels, best_thresh)
                test_metrics = compute_per_class_metrics(test_probs, test_labels, best_thresh)

                # Save results
                out_dir = os.path.join("Inference", category, model_name)
                os.makedirs(out_dir, exist_ok=True)

                save_csv(val_df, val_probs, val_preds, os.path.join(out_dir, "val_predictions.csv"))
                save_csv(test_df, test_probs, test_preds, os.path.join(out_dir, "test_predictions.csv"))

                with open(os.path.join(out_dir, "val_metrics.json"), "w") as f:
                    json.dump(val_metrics, f, indent=2)
                
                with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
                    json.dump(test_metrics, f, indent=2)
                    
                print(f"  Saved to {out_dir}")
            except Exception as e:
                print(f"  FAILED to process {model_name}: {e}")

if __name__ == '__main__':
    main()
