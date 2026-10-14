"""
Summarise the prior ablation across seeds and run paired bootstrap tests.

Reads   experiments/results/ablation_studies/<variant>/<family>/seed_<s>/
Writes  experiments/results/ablation_studies/comparison_<family>/
            summary.csv, summary.md        mean ± std over seeds (val / test / heldout)
            bootstrap.json, bootstrap.md   paired bootstrap of PR-AUC and ROC-AUC differences

Comparisons:
    A   : baseline_4level_5d  vs ablation_3level_5d   (effect of removing the family level)
    B   : ablation_3level_5d  vs ablation_3level_1d   (effect of 5-d -> 1-d prior vector)
    A+B : baseline_4level_5d  vs ablation_3level_1d   (both changes together)

Bootstrap: the evaluation rows are resampled with replacement, stratified by label so every
resample keeps the same number of positives. On each resample both models are scored on the
same rows (paired) and the statistic is
    mean over seeds of metric(model X)  -  mean over seeds of metric(model Y)
so the interval reflects test-set sampling noise with seed noise averaged out.

Usage (from the project root):
    python -m experiments.ablation_studies.compare
    python -m experiments.ablation_studies.compare --n-boot 10000 --splits test val
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.ablation_studies.run_ablation import VARIANTS, RESULTS_ROOT

COMPARISONS = [
    ("A",   "baseline_4level_5d", "ablation_3level_5d", "remove family level (5-d kept)"),
    ("B",   "ablation_3level_5d", "ablation_3level_1d", "5-d -> 1-d prior (3-level kept)"),
    ("A+B", "baseline_4level_5d", "ablation_3level_1d", "both changes"),
]
SUMMARY_METRICS = ["pr_auc", "roc_auc", "f1", "mcc", "precision", "recall", "threshold"]


def fast_average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Same value as sklearn.metrics.average_precision_score (step-wise, ties grouped)."""
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[idx]
    fps = idx + 1 - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def fast_roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney form of ROC-AUC (ties count 1/2), equal to sklearn.metrics.roc_auc_score."""
    ranks = rankdata(s)
    n_pos = y.sum()
    n_neg = y.size - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


METRIC_FNS = {"pr_auc": fast_average_precision, "roc_auc": fast_roc_auc}


def find_seeds(root_dir: str, family: str) -> dict:
    seeds = {}
    for v in VARIANTS:
        root = os.path.join(root_dir, v, family)
        found = set()
        if os.path.isdir(root):
            for d in os.listdir(root):
                if d.startswith("seed_") and os.path.exists(os.path.join(root, d, "test_metrics.json")):
                    found.add(int(d.split("_")[1]))
        seeds[v] = found
    return seeds


def load_metrics(root, variant, family, seed, split):
    path = os.path.join(root, variant, family, f"seed_{seed}", f"{split}_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_predictions(root, variant, family, seed, split):
    return pd.read_csv(os.path.join(root, variant, family, f"seed_{seed}", f"predictions_{split}.csv"))


def summarise(root, family, seeds):
    rows = []
    for v in VARIANTS:
        for split in ["val", "test", "heldout"]:
            ms = [m for m in (load_metrics(root, v, family, s, split) for s in seeds) if m is not None]
            if not ms:
                continue
            row = {"variant": v, "split": split, "n_seeds": len(ms)}
            for k in SUMMARY_METRICS:
                vals = np.array([m[k] for m in ms], dtype=float)
                row[f"{k}_mean"] = vals.mean()
                row[f"{k}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(root, family, seeds, split, n_boot, rng_seed):
    # probs[variant] -> array [n_seeds, n_rows]; all files must share the same rows in the same order
    probs, ref = {}, None
    for v in VARIANTS:
        mats = []
        for s in seeds:
            df = load_predictions(root, v, family, s, split)
            if ref is None:
                ref = df
            elif not (df["label"].equals(ref["label"]) and
                      df["API_Smiles"].astype(str).equals(ref["API_Smiles"].astype(str))):
                raise ValueError(f"row order mismatch in {v} seed {s} {split}")
            mats.append(df["prob"].to_numpy(dtype=float))
        probs[v] = np.stack(mats)
    y = ref["label"].to_numpy(dtype=int)

    # Sanity: fast metrics agree with sklearn on the observed data
    for v in VARIANTS:
        assert np.isclose(fast_average_precision(y, probs[v][0]), average_precision_score(y, probs[v][0]))
        assert np.isclose(fast_roc_auc(y, probs[v][0]), roc_auc_score(y, probs[v][0]))

    pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(rng_seed)
    boot = {m: {v: np.empty(n_boot) for v in VARIANTS} for m in METRIC_FNS}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos_idx, pos_idx.size), rng.choice(neg_idx, neg_idx.size)])
        yb = y[idx]
        for v in VARIANTS:
            for m, fn in METRIC_FNS.items():
                boot[m][v][b] = np.mean([fn(yb, p[idx]) for p in probs[v]])

    results = []
    for name, x, z, desc in COMPARISONS:
        for m, fn in METRIC_FNS.items():
            per_seed_x = np.array([fn(y, p) for p in probs[x]])
            per_seed_z = np.array([fn(y, p) for p in probs[z]])
            per_seed_diff = per_seed_x - per_seed_z
            d = boot[m][x] - boot[m][z]
            p_two_sided = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            results.append({
                "comparison": name,
                "description": desc,
                "split": split,
                "metric": m,
                "model_x": x,
                "model_y": z,
                "x_mean": float(per_seed_x.mean()),
                "y_mean": float(per_seed_z.mean()),
                "diff_x_minus_y": float(per_seed_diff.mean()),
                "per_seed_diff_std": float(per_seed_diff.std(ddof=1)) if len(seeds) > 1 else 0.0,
                "seeds_x_better": int((per_seed_diff > 0).sum()),
                "n_seeds": len(seeds),
                "ci95_low": float(np.percentile(d, 2.5)),
                "ci95_high": float(np.percentile(d, 97.5)),
                "p_two_sided": float(p_two_sided),
                "n_boot": n_boot,
                "n_rows": int(y.size),
                "n_pos": int(y.sum()),
            })
    return results


def write_markdown_summary(df, path):
    lines = ["# Prior ablation — mean ± std over seeds", ""]
    for split in ["test", "val", "heldout"]:
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        lines += [f"## {split}", "",
                  "| variant | seeds | PR-AUC | ROC-AUC | F1 | MCC | precision | recall |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            cells = [f"{r[f'{k}_mean']:.4f} ± {r[f'{k}_std']:.4f}"
                     for k in ["pr_auc", "roc_auc", "f1", "mcc", "precision", "recall"]]
            lines.append(f"| {r['variant']} | {r['n_seeds']} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_markdown_bootstrap(results, path):
    lines = ["# Paired bootstrap (stratified, mean over seeds)", "",
             "diff = model X − model Y. A 95% CI that excludes 0 indicates a difference "
             "beyond test-set sampling noise.", "",
             "| comparison | split | metric | X | Y | X mean | Y mean | diff | 95% CI | p | seeds X better |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['comparison']} | {r['split']} | {r['metric']} | {r['model_x']} | {r['model_y']} | "
            f"{r['x_mean']:.4f} | {r['y_mean']:.4f} | {r['diff_x_minus_y']:+.4f} | "
            f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {r['p_two_sided']:.3f} | "
            f"{r['seeds_x_better']}/{r['n_seeds']} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarise prior ablation runs and run paired bootstrap tests.")
    parser.add_argument("--family", default="morgan")
    parser.add_argument("--splits", nargs="+", default=["test", "val"], help="splits to bootstrap")
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()

    found = find_seeds(args.results_root, args.family)
    seeds = sorted(set.intersection(*found.values()))
    for v, s in found.items():
        print(f"{v}: seeds {sorted(s)}")
    if not seeds:
        raise SystemExit("No seed has finished for all variants yet.")
    if any(found[v] != set(seeds) for v in found):
        print(f"Using only the seeds finished for every variant: {seeds}")

    out_dir = os.path.join(args.results_root, f"comparison_{args.family}")
    os.makedirs(out_dir, exist_ok=True)

    summary = summarise(args.results_root, args.family, seeds)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    write_markdown_summary(summary, os.path.join(out_dir, "summary.md"))

    results = []
    for split in args.splits:
        results += paired_bootstrap(args.results_root, args.family, seeds, split, args.n_boot, args.boot_seed)
    with open(os.path.join(out_dir, "bootstrap.json"), "w") as f:
        json.dump({"seeds": seeds, "results": results}, f, indent=2)
    write_markdown_bootstrap(results, os.path.join(out_dir, "bootstrap.md"))

    print(open(os.path.join(out_dir, "summary.md"), encoding="utf-8").read())
    print(open(os.path.join(out_dir, "bootstrap.md"), encoding="utf-8").read())
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
