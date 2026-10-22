"""
SHAP analysis of the final_project Morgan-PGB model (20/24 held-out) over all 2,468 input features.

Same analyses as project/shap_analysis, run on the final_project checkpoint, data and code (final_project/src).
Run from inside final_project/:

    python -m shap_analysis.run_shap --split heldout     # 24 held-out pairs
    python -m shap_analysis.run_shap --split val         # 774 val pairs
    python -m shap_analysis.run_shap --split train       # 2770 train pairs
    python -m shap_analysis.run_shap --split full        # all 3544 pairs (start_dataset.csv)
    python -m shap_analysis.run_shap --csv my_pairs.csv  # any CSV with API_Smiles / Excipient_Smiles / CIDs / label

Outputs: shap_analysis/results/<split>/ (shap_values.npz, feature_importance.csv, predictions.csv,
summary_report.md, plots/).

Method: shap.GradientExplainer (expected gradients) on the logit, background = random train rows with the
leave-own-cluster-out priors they had during training. Before any SHAP value is computed the run checks that
the flat 2,468-feature wrapper gives the same logits as the model's own forward pass.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # final_project/
CALLER_CWD = os.getcwd()
sys.path.insert(0, HERE)
os.chdir(HERE)

import numpy as np
import pandas as pd
import shap
import torch
from rdkit import RDLogger

from shap_analysis.feature_names import FEATURE_NAMES, check_names_match_features
from shap_analysis.model_wrapper import FlatPGBModelWrapper, TowerLevelPGBModelWrapper
from shap_analysis.plots import (plot_chemistry_flags_attribution, plot_class_comparison, plot_descriptor_beeswarm_and_bar,
                                 plot_feature_group_importance, plot_mechanism_flags_attribution, plot_morgan_top_bits,
                                 plot_prior_attribution, plot_sample_waterfall, plot_top_features_bar, plot_tower_level_attribution)
from shap_analysis.utils import compute_feature_importance_df, decode_morgan_bit, extract_features, load_checkpoint, make_dataset
from src.pgb_prior import build_leave_cluster_out_prior_vectors

RDLogger.DisableLog("rdApp.*")

CKPT_DIR = "checkpoints/morgan_pgb_8020_seed42"
TRAIN_CSV = "data/train.csv"
SPLITS = {"heldout": "data/held_out_test_set.csv", "val": "data/val.csv", "train": "data/train.csv", "full": "data/start_dataset.csv"}


def parse_args():
    p = argparse.ArgumentParser(description="SHAP analysis of the final_project Morgan-PGB model.")
    p.add_argument("--split", choices=list(SPLITS), default=None, help="dataset to explain (default: heldout)")
    p.add_argument("--csv", default=None, help="explain this CSV instead of a named split")
    p.add_argument("--checkpoint", default=CKPT_DIR, help="checkpoint folder (best_model.pt, prior_table.json, pgb_norm_stats.npz)")
    p.add_argument("--bg_samples", type=int, default=100, help="background rows sampled from train")
    p.add_argument("--max_samples", type=int, default=0, help="explain at most this many rows (0 = all)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--nsamples", type=int, default=200, help="expected-gradients samples per row (shap default 200)")
    p.add_argument("--out", default=None, help="results folder (default shap_analysis/results/<split>)")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    for attr in ("csv", "checkpoint", "out"):
        v = getattr(args, attr)
        if v and not os.path.isabs(v) and os.path.exists(os.path.join(CALLER_CWD, v)) and not os.path.exists(v):
            setattr(args, attr, os.path.join(CALLER_CWD, v))
    return args


def sanitized_csv(path, out_dir):
    """PGBDataset reads the label from Outcome1; the held-out file calls it ground_truth."""
    df = pd.read_csv(path)
    if "Outcome1" in df.columns:
        return path
    for alt in ["ground_truth", "Outcome", "label"]:
        if alt in df.columns:
            p = os.path.join(out_dir, "_input_with_Outcome1.csv")
            df.rename(columns={alt: "Outcome1"}).to_csv(p, index=False)
            return p
    raise ValueError(f"{path} has no label column (Outcome1 / ground_truth / Outcome / label)")


def explain(explainer, X, batch_size, nsamples, seed):
    out = []
    for s in range(0, len(X), batch_size):
        sv = explainer.shap_values(X[s:s + batch_size], nsamples=nsamples, rseed=seed)
        sv = sv[0] if isinstance(sv, list) else sv
        out.append(sv.squeeze(-1) if sv.ndim == 3 else sv)
        print(f"  {min(s + batch_size, len(X))}/{len(X)} rows")
    return np.concatenate(out)


def write_report(path, tag, data_path, imp, group_df, base_value, n, fidelity, additivity, decoded, threshold, preds, runtime):
    L = ["# Morgan-PGB (final_project) — SHAP report", "",
         f"- **Model**: Prior-Gated Bilinear head on Morgan-1024 + MACCS (`{CKPT_DIR}`), threshold {threshold:.3f}",
         f"- **Data**: `{data_path}` ({tag}), {n} pairs explained",
         f"- **Explainer**: shap.GradientExplainer (expected gradients) on the logit, background = train rows with their training priors",
         f"- **Baseline (mean background logit)**: {base_value:.4f} (probability {1 / (1 + np.exp(-base_value)):.2%})",
         f"- **Fidelity check**: flat-feature wrapper vs model forward, max |logit difference| = {fidelity:.2e}",
         f"- **Additivity** (expected gradients is approximate): mean |base + sum(SHAP) − logit| = {additivity['mean']:.4f}, "
         f"max = {additivity['max']:.4f} (logit std over these rows {additivity['logit_std']:.3f})",
         f"- **Predictions**: {preds['correct']}/{n} correct at the model threshold (TP {preds['tp']}, TN {preds['tn']}, FP {preds['fp']}, FN {preds['fn']})",
         f"- Runtime {runtime:.0f} s", "", "## 1. Feature-group share of mean |SHAP|", "",
         "| Rank | Group | Features | Total mean \\|SHAP\\| | Share |", "|---|---|---|---|---|"]
    for r, (_, row) in enumerate(group_df.sort_values("total_abs_shap", ascending=False).iterrows(), 1):
        L.append(f"| {r} | `{row['group']}` | {row['feature_count']} | {row['total_abs_shap']:.4f} | {row['percentage']:.2f}% |")
    L += ["", "## 2. Top 20 features", "", "| Rank | Feature | Group | Mean \\|SHAP\\| | Share | Mean SHAP (+ pushes towards incompatible) |",
          "|---|---|---|---|---|---|"]
    for r, (_, row) in enumerate(imp.head(20).iterrows(), 1):
        L.append(f"| {r} | `{row['feature_name']}` | `{row['feature_group']}` | {row['mean_abs_shap']:.4f} | {row['percentage_importance']:.2f}% | {row['mean_shap']:+.4f} |")
    L += ["", "## 3. Top Morgan bits, decoded (salt-aware, same generator as the model)", "",
          "| Feature | Most common substructure | Molecules with the bit | Occurrences |", "|---|---|---|---|"]
    for name, info in decoded.items():
        L.append(f"| `{name}` | `{info['top_fragment_smiles']}` | {info['molecules_with_bit']} | {info['occurrence_count']} |")
    L += ["", "SHAP values explain the model's logit (log-odds of incompatible). Positive = pushes towards incompatible.",
          "Expected gradients treats binary fingerprint bits as continuous inputs, so read bit attributions as relative importance."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    args = parse_args()
    check_names_match_features()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.csv:
        data_path, tag = args.csv, os.path.splitext(os.path.basename(args.csv))[0]
    else:
        tag = args.split or "heldout"
        data_path = SPLITS[tag]
    out_dir = args.out or os.path.join("shap_analysis", "results", tag)
    plots = os.path.join(out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    t0 = time.time()
    print(f"SHAP | model {args.checkpoint} | data {data_path} | device {device} | out {out_dir}")

    print("[1/7] loading model")
    model, config, encoder, prior_table, pgb_norm = load_checkpoint(args.checkpoint, device)
    thr_file = os.path.join(args.checkpoint, "threshold.json")
    threshold = json.load(open(thr_file))["threshold"] if os.path.exists(thr_file) else 0.5

    print("[2/7] building features")
    train_df = pd.read_csv(TRAIN_CSV)
    train_priors = build_leave_cluster_out_prior_vectors(train_df, cluster_col="cluster_id")  # as in training
    train_ds = make_dataset(TRAIN_CSV, config, prior_table, pgb_norm, encoder, row_priors=train_priors)
    rng = np.random.default_rng(args.seed)
    bg = extract_features(train_ds, model, device, rng.choice(len(train_ds), size=min(args.bg_samples, len(train_ds)), replace=False))
    data_csv = sanitized_csv(data_path, out_dir)
    data_ds = make_dataset(data_csv, config, prior_table, pgb_norm, encoder)
    idx = None
    if args.max_samples and args.max_samples < len(data_ds):
        idx = rng.choice(len(data_ds), size=args.max_samples, replace=False)
    d = extract_features(data_ds, model, device, idx)
    if data_csv != data_path:
        os.remove(data_csv)
    X, Z, y, meta = d["X"], d["Z"], d["y"], d["meta"]
    print(f"  background {bg['X'].shape} | explained {X.shape}")

    print("[3/7] fidelity check: wrapper vs model forward")
    dev = torch.device(device)
    wrapper = FlatPGBModelWrapper(model, output_mode="logit").to(dev).eval()
    with torch.no_grad():
        wrap_logits = wrapper(torch.tensor(X, device=dev)).squeeze(-1).cpu().numpy()
        base_value = float(wrapper(torch.tensor(bg["X"], device=dev)).mean().cpu())
    fidelity = float(np.abs(wrap_logits - d["model_logits"]).max())
    if fidelity > 1e-4:
        raise RuntimeError(f"the SHAP wrapper does not reproduce the model (max logit difference {fidelity:.2e})")
    print(f"  OK, max |difference| {fidelity:.2e}")

    print("[4/7] SHAP values (GradientExplainer)")
    explainer = shap.GradientExplainer(wrapper, torch.tensor(bg["X"], device=dev))
    shap_values = explain(explainer, torch.tensor(X, device=dev), args.batch_size, args.nsamples, args.seed)
    resid = base_value + shap_values.sum(axis=1) - wrap_logits
    additivity = {"mean": float(np.abs(resid).mean()), "max": float(np.abs(resid).max()), "logit_std": float(wrap_logits.std())}
    print(f"  shape {shap_values.shape} | additivity mean {additivity['mean']:.4f} max {additivity['max']:.4f}")

    print("[5/7] tower-level SHAP")
    tower = TowerLevelPGBModelWrapper(model, output_mode="logit").to(dev).eval()
    tower_shap = explain(shap.GradientExplainer(tower, torch.tensor(bg["Z"], device=dev)), torch.tensor(Z, device=dev),
                         args.batch_size, args.nsamples, args.seed)

    print("[6/7] importance table and Morgan-bit decoding")
    imp = compute_feature_importance_df(shap_values, FEATURE_NAMES)
    smiles = {"api_primary": list(meta.api_smiles), "exc_primary": list(meta.exc_smiles)}
    train_smiles = {"api_primary": list(train_df.API_Smiles), "exc_primary": list(train_df.Excipient_Smiles)}
    decoded, decoded_by_bit = {}, {}
    for _, r in imp[imp.feature_group.isin(smiles)].head(10).iterrows():
        b = int(r.feature_name.split("_")[-1])
        info = decode_morgan_bit(smiles[r.feature_group], b)
        if info["molecules_with_bit"] == 0:
            # the bit is OFF in every explained molecule: its attribution is the absence of a substructure that
            # the (train) background has; show which one
            t = decode_morgan_bit(train_smiles[r.feature_group], b)
            info["top_fragment_smiles"] = f"absent here; in train: {t['top_fragment_smiles']}"
        decoded[r.feature_name] = info
        decoded_by_bit.setdefault(b, info)
    probs = 1 / (1 + np.exp(-wrap_logits))
    pred = (probs >= threshold).astype(int)
    preds_tbl = meta.copy()
    preds_tbl["probability"], preds_tbl["prediction"] = probs, pred
    preds_tbl["correct"] = (pred == y.astype(int)).astype(int)
    for i in range(3):
        o = np.argsort(-np.abs(shap_values), axis=1)[:, i]
        preds_tbl[f"top{i + 1}_feature"] = [FEATURE_NAMES[j] for j in o]
        preds_tbl[f"top{i + 1}_shap"] = shap_values[np.arange(len(o)), o]
    yi = y.astype(int)
    preds = {"correct": int(preds_tbl.correct.sum()), "tp": int(((pred == 1) & (yi == 1)).sum()), "tn": int(((pred == 0) & (yi == 0)).sum()),
             "fp": int(((pred == 1) & (yi == 0)).sum()), "fn": int(((pred == 0) & (yi == 1)).sum())}

    print("[7/7] plots and files")
    group_df = plot_feature_group_importance(shap_values, save_path=os.path.join(plots, "feature_group_importance.png"))
    plot_top_features_bar(shap_values, FEATURE_NAMES, top_k=25, save_path=os.path.join(plots, "top_features_overall.png"))
    plot_descriptor_beeswarm_and_bar(shap_values, d["raw_desc_api"], d["raw_desc_exc"], save_path=os.path.join(plots, "descriptor_attribution.png"))
    plot_chemistry_flags_attribution(shap_values, save_path=os.path.join(plots, "chemistry_flags_attribution.png"))
    plot_mechanism_flags_attribution(shap_values, X, save_path=os.path.join(plots, "mechanism_flags_attribution.png"))
    plot_prior_attribution(shap_values, X, save_path=os.path.join(plots, "prior_attribution.png"))
    plot_morgan_top_bits(shap_values, FEATURE_NAMES, decoded_bits=decoded_by_bit, top_k=20, save_path=os.path.join(plots, "morgan_top_bits.png"))
    plot_tower_level_attribution(tower_shap, save_path=os.path.join(plots, "tower_level_attribution.png"))
    if len(np.unique(yi)) > 1:
        plot_class_comparison(shap_values, y, FEATURE_NAMES, top_k=15, save_path=os.path.join(plots, "class_comparison.png"))
    # every pair when there are few (held-out), otherwise the 5 most confident of each class
    if len(X) <= 30:
        wf = list(range(len(X)))
    else:
        wf = list(np.argsort(-probs * (yi == 1))[:5]) + list(np.argsort(probs + (yi == 1))[:5])
    for i in wf:
        m = meta.iloc[i].to_dict()
        name = f"{m['api_name']}__{m['exc_name']}".replace(" ", "_").replace("/", "-")[:60] if m["api_name"] else f"row{m['row_index']}"
        plot_sample_waterfall(shap_values, base_value, X, FEATURE_NAMES, sample_idx=int(i), sample_meta=m, top_k=15,
                              save_path=os.path.join(plots, f"waterfall_{int(i):03d}_{name}.png"))
    np.savez_compressed(os.path.join(out_dir, "shap_values.npz"), shap_values=shap_values, base_value=base_value, X=X, y=y,
                        logits=wrap_logits, tower_shap=tower_shap, feature_names=np.array(FEATURE_NAMES))
    imp.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
    preds_tbl.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    write_report(os.path.join(out_dir, "summary_report.md"), tag, data_path, imp, group_df, base_value, len(X), fidelity, additivity,
                 decoded, threshold, preds, time.time() - t0)
    print(f"\nDone in {time.time() - t0:.0f} s -> {out_dir}  ({preds['correct']}/{len(X)} correct at threshold {threshold:.3f})")


if __name__ == "__main__":
    main()
