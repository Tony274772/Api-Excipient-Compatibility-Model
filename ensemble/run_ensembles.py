import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.dataset import CompatibilityDataset, collate_fn
from inference import HeldOutDataset
from src.evaluate import collect_predictions, tune_threshold, compute_metrics, save_metrics
from src.descriptors import DESCRIPTOR_NAMES, normalize_descriptors
from ensemble._common import load_config_for_checkpoint, build_model_from_checkpoint

# The 3 base models involved
BASE_MODELS = [
    "molformer_concat_asl",
    "molformer_cross_attn_asl",
    "molformer_cross_attn_bce"
]

# The 2 pairs
PAIRS = [
    ("molformer_concat_asl", "molformer_cross_attn_asl"),
    ("molformer_concat_asl", "molformer_cross_attn_bce")
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def step1_cache_predictions():
    os.makedirs("ensemble/_base_predictions", exist_ok=True)
    
    for model_name in BASE_MODELS:
        model_out_dir = f"ensemble/_base_predictions/{model_name}"
        os.makedirs(model_out_dir, exist_ok=True)
        
        # Skip if already cached
        if all(os.path.exists(f"{model_out_dir}/{split}.csv") for split in ["val", "test", "held_out"]):
            print(f"Skipping prediction caching for {model_name}, already exists.")
            continue
            
        print(f"Caching predictions for base model: {model_name}")
        model, config = build_model_from_checkpoint(model_name, device)
        
        splits = {
            "val": CompatibilityDataset(
                config.val_csv, config.api_descriptors_path,
                config.excipient_descriptors_path, config.descriptor_norm_stats_path
            ),
            "test": CompatibilityDataset(
                config.test_csv, config.api_descriptors_path,
                config.excipient_descriptors_path, config.descriptor_norm_stats_path
            ),
            "held_out": HeldOutDataset(
                "held_out_testset/held_out_test_set.csv",
                "held_out_testset/api_descriptors.csv",
                "held_out_testset/excipient_descriptors.csv",
                config.descriptor_norm_stats_path
            )
        }
        
        for split_name, ds in splits.items():
            loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)
            probs, labels = collect_predictions(model, loader, device)
            
            # Explicit key columns
            row_indices = np.arange(len(ds))
            api_cids = ds.df["API_CID"].values
            exc_cids = ds.df["Excipient_CID"].values
            
            df = pd.DataFrame({
                "row_index": row_indices,
                "API_CID": api_cids,
                "Excipient_CID": exc_cids,
                "label": labels.astype(int),
                "prob": probs
            })
            df.to_csv(f"{model_out_dir}/{split_name}.csv", index=False)


def step2_align_pairs():
    aligned_data = {}
    for pair in PAIRS:
        model_a, model_b = pair
        aligned_data[pair] = {}
        for split in ["val", "test", "held_out"]:
            df_a = pd.read_csv(f"ensemble/_base_predictions/{model_a}/{split}.csv")
            df_b = pd.read_csv(f"ensemble/_base_predictions/{model_b}/{split}.csv")
            
            merged = pd.merge(df_a, df_b, on=["row_index", "API_CID", "Excipient_CID"], 
                              suffixes=('_a', '_b'), how='inner')
            
            assert len(merged) == len(df_a), f"Merge row count mismatch for {split} in pair {pair}"
            assert (merged["label_a"] == merged["label_b"]).all(), f"Label mismatch in {split} for pair {pair}"
            
            merged = merged.rename(columns={"label_a": "label", "prob_a": "prob_a", "prob_b": "prob_b"})
            merged = merged.drop(columns=["label_b"])
            
            aligned_data[pair][split] = merged
            
    return aligned_data


def save_held_out_preds(pair_dir, method, held_out_df, final_probs, threshold):
    out_dir = f"ensemble/{pair_dir}/{method}"
    
    # Load original held out CSV for names
    orig_ho = pd.read_csv("held_out_testset/held_out_test_set.csv")
    merged = pd.merge(held_out_df, orig_ho[["API_CID", "Excipient_CID", "Api_name", "Excipient_name"]], 
                      on=["API_CID", "Excipient_CID"], how="left")
    
    preds = (final_probs >= threshold).astype(int)
    out_df = pd.DataFrame({
        "Api_name": merged["Api_name"],
        "Excipient_name": merged["Excipient_name"],
        "API_CID": merged["API_CID"],
        "Excipient_CID": merged["Excipient_CID"],
        "ground_truth": merged["label"],
        "ensemble_probability": final_probs,
        "ensemble_prediction": preds
    })
    out_df.to_csv(f"{out_dir}/held_out_predictions.csv", index=False)


def do_method1_avg(pair, data):
    pair_dir = f"{pair[0]}__{pair[1]}"
    out_dir = f"ensemble/{pair_dir}/avg"
    os.makedirs(f"{out_dir}/checkpoint", exist_ok=True)
    os.makedirs(f"{out_dir}/metrics", exist_ok=True)
    
    val_probs = (data["val"]["prob_a"] + data["val"]["prob_b"]) / 2.0
    val_labels = data["val"]["label"].values
    
    best_thresh, best_f1 = tune_threshold(val_probs.values, val_labels, step=0.001)
    
    config = {
        "method": "avg", 
        "base_model_a": pair[0], 
        "base_model_b": pair[1],
        "weight_a": 0.5, 
        "weight_b": 0.5, 
        "threshold": best_thresh
    }
    with open(f"{out_dir}/checkpoint/ensemble_config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    for split in ["val", "test", "held_out"]:
        probs = (data[split]["prob_a"] + data[split]["prob_b"]) / 2.0
        labels = data[split]["label"].values
        metrics = compute_metrics(probs.values, labels, best_thresh)
        save_metrics(metrics, f"{out_dir}/metrics/{split}_metrics.json")
        
        if split == "held_out":
            save_held_out_preds(pair_dir, "avg", data[split], probs.values, best_thresh)


def do_method2_weighted_avg(pair, data):
    pair_dir = f"{pair[0]}__{pair[1]}"
    out_dir = f"ensemble/{pair_dir}/weighted_avg"
    os.makedirs(f"{out_dir}/checkpoint", exist_ok=True)
    os.makedirs(f"{out_dir}/metrics", exist_ok=True)
    
    val_labels = data["val"]["label"].values
    prob_a = data["val"]["prob_a"].values
    prob_b = data["val"]["prob_b"].values
    
    search_results = []
    best_w = 0.0
    best_pr_auc = -1.0
    best_f1_for_tie = -1.0
    
    for w in np.arange(0.0, 1.01, 0.05):
        blended = w * prob_a + (1 - w) * prob_b
        pr_auc = average_precision_score(val_labels, blended)
        
        # Need F1 for tie breaking
        thresh, f1 = tune_threshold(blended, val_labels, step=0.01)
        
        search_results.append({"weight_a": w, "val_pr_auc": pr_auc})
        
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_w = w
            best_f1_for_tie = f1
        elif np.isclose(pr_auc, best_pr_auc) and f1 > best_f1_for_tie:
            best_pr_auc = pr_auc
            best_w = w
            best_f1_for_tie = f1
            
    pd.DataFrame(search_results).to_csv(f"{out_dir}/checkpoint/weight_search.csv", index=False)
    
    blended_val = best_w * prob_a + (1 - best_w) * prob_b
    best_thresh, _ = tune_threshold(blended_val, val_labels, step=0.001)
    
    config = {
        "method": "weighted_avg", 
        "base_model_a": pair[0], 
        "base_model_b": pair[1],
        "weight_a": best_w, 
        "weight_b": 1.0 - best_w, 
        "selection_metric": "val_pr_auc",
        "threshold": best_thresh
    }
    with open(f"{out_dir}/checkpoint/ensemble_config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    for split in ["val", "test", "held_out"]:
        probs = best_w * data[split]["prob_a"].values + (1 - best_w) * data[split]["prob_b"].values
        labels = data[split]["label"].values
        metrics = compute_metrics(probs, labels, best_thresh)
        save_metrics(metrics, f"{out_dir}/metrics/{split}_metrics.json")
        
        if split == "held_out":
            save_held_out_preds(pair_dir, "weighted_avg", data[split], probs, best_thresh)


def build_stacker_features(df, split):
    with open("models/descriptor_norm_stats.json", "r") as f:
        stats = json.load(f)
    api_mean, api_std = np.array(stats["api_mean"]), np.array(stats["api_std"])
    exc_mean, exc_std = np.array(stats["exc_mean"]), np.array(stats["exc_std"])
    
    prefix = "held_out_testset" if split == "held_out" else "data"
    api_desc_df = pd.read_csv(f"{prefix}/api_descriptors.csv").set_index("API_CID")
    exc_desc_df = pd.read_csv(f"{prefix}/excipient_descriptors.csv").set_index("Excipient_CID")
    
    features = []
    for _, row in df.iterrows():
        api_cid = row["API_CID"]
        exc_cid = row["Excipient_CID"]
        
        api_raw = api_desc_df.loc[api_cid][DESCRIPTOR_NAMES].values.astype(float) if api_cid in api_desc_df.index else np.zeros(21)
        exc_raw = exc_desc_df.loc[exc_cid][DESCRIPTOR_NAMES].values.astype(float) if exc_cid in exc_desc_df.index else np.zeros(21)
        
        api_norm = np.nan_to_num(normalize_descriptors(api_raw, api_mean, api_std), nan=0.0)
        exc_norm = np.nan_to_num(normalize_descriptors(exc_raw, exc_mean, exc_std), nan=0.0)
        
        feat = [row["prob_a"], row["prob_b"]] + api_norm.tolist() + exc_norm.tolist()
        features.append(feat)
        
    return np.array(features)

def do_method3_stacker(pair, data):
    pair_dir = f"{pair[0]}__{pair[1]}"
    out_dir = f"ensemble/{pair_dir}/lr_stacker"
    os.makedirs(f"{out_dir}/checkpoint", exist_ok=True)
    os.makedirs(f"{out_dir}/metrics", exist_ok=True)
    
    X_val = build_stacker_features(data["val"], "val")
    y_val = data["val"]["label"].values
    
    scaler = StandardScaler()
    X_val_scaled = scaler.fit_transform(X_val)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    C_grid = [0.001, 0.01, 0.1, 1, 10, 100]
    
    cv_results = []
    best_C = None
    best_pr_auc = -1
    
    for C in C_grid:
        fold_scores = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_val_scaled, y_val)):
            X_tr, y_tr = X_val_scaled[train_idx], y_val[train_idx]
            X_te, y_te = X_val_scaled[test_idx], y_val[test_idx]
            
            clf = LogisticRegression(penalty="l2", class_weight="balanced", solver="liblinear", C=C)
            clf.fit(X_tr, y_tr)
            preds = clf.predict_proba(X_te)[:, 1]
            score = average_precision_score(y_te, preds)
            fold_scores.append(score)
            cv_results.append({"C": C, "fold": fold, "val_pr_auc": score})
            
        mean_score = np.mean(fold_scores)
        if mean_score > best_pr_auc:
            best_pr_auc = mean_score
            best_C = C
            
    pd.DataFrame(cv_results).to_csv(f"{out_dir}/checkpoint/cv_results.csv", index=False)
    
    # Refit on full val
    clf = LogisticRegression(penalty="l2", class_weight="balanced", solver="liblinear", C=best_C)
    clf.fit(X_val_scaled, y_val)
    
    val_probs = clf.predict_proba(X_val_scaled)[:, 1]
    best_thresh, _ = tune_threshold(val_probs, y_val, step=0.001)
    
    feature_names = ["prob_a", "prob_b"] + [f"api_{n}" for n in DESCRIPTOR_NAMES] + [f"exc_{n}" for n in DESCRIPTOR_NAMES]
    
    joblib.dump(
        {"scaler": scaler, "model": clf,
         "feature_names": feature_names,
         "best_C": best_C, "threshold": best_thresh},
        f"{out_dir}/checkpoint/lr_model.joblib"
    )
    
    # Eval on all splits
    for split in ["val", "test", "held_out"]:
        X = build_stacker_features(data[split], split)
        X_scaled = scaler.transform(X)
        probs = clf.predict_proba(X_scaled)[:, 1]
        labels = data[split]["label"].values
        
        metrics = compute_metrics(probs, labels, best_thresh)
        save_metrics(metrics, f"{out_dir}/metrics/{split}_metrics.json")
        
        if split == "held_out":
            save_held_out_preds(pair_dir, "lr_stacker", data[split], probs, best_thresh)


def build_comparison_summary():
    rows = []
    
    for split in ["val", "test", "held_out"]:
        # Base models
        for bm in BASE_MODELS:
            metrics_path = f"metrics/{bm}/{split}_metrics.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    m = json.load(f)
                m["model_name"] = bm
                m["split"] = split
                rows.append(m)
                
        # Ensembles
        for pair in PAIRS:
            pair_dir = f"{pair[0]}__{pair[1]}"
            for method in ["avg", "weighted_avg", "lr_stacker"]:
                metrics_path = f"ensemble/{pair_dir}/{method}/metrics/{split}_metrics.json"
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        m = json.load(f)
                    m["model_name"] = f"{pair_dir}/{method}"
                    m["split"] = split
                    rows.append(m)
                    
    df = pd.DataFrame(rows)
    # Ensure columns match spec
    cols = ["model_name", "split", "pr_auc", "f1", "mcc", "precision", "recall", "accuracy", "threshold"]
    # confusing matrix not needed in summary, keep only specified
    df = df[cols]
    df.to_csv("ensemble/comparison_summary.csv", index=False)


if __name__ == "__main__":
    print("Step 1: Caching base predictions...")
    step1_cache_predictions()
    
    print("\nStep 2: Aligning pairs...")
    aligned_data = step2_align_pairs()
    
    print("\nStep 3: Building ensembles...")
    for pair in PAIRS:
        print(f"  Pair: {pair}")
        do_method1_avg(pair, aligned_data[pair])
        do_method2_weighted_avg(pair, aligned_data[pair])
        do_method3_stacker(pair, aligned_data[pair])
        
    print("\nStep 4: Building comparison summary...")
    build_comparison_summary()
    print("Done!")
