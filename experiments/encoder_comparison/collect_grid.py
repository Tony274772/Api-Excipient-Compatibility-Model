"""Collect the grid, run the 11 tests, apply the decision rule.

Reads two result roots:

    experiments/results/encoder_comparison/<family>_seed<N>/   the 45 completed `none`-column runs
    experiments/results/encoder_grid/<cell>_seed<N>/           every new cell

Selection metric is validation PR-AUC. Held-out metrics stay sealed: this script never reads them
unless --reveal-heldout is passed, and that is meant to be run ONCE, after the decision rule has
already picked a model (Part 7, rule 5 of paper/encoder_grid_plan.md).

Two honesty requirements are built in, not optional:

  * Power. From the only existing pair-vs-single comparison on shared seeds (paper/designb_vs_nine.csv)
    the per-seed standard deviation of a paired difference is about 0.0132, so 5 seeds detect about
    0.026 and no less. A null result is reported as "nothing helps by more than X", never as
    "nothing helps".
  * Multiplicity. 72 pair cells tested at p < 0.05 produce about 3.6 winners by luck, so the
    "vs none" family of tests is reported with Benjamini-Hochberg FDR at q = 0.10 and the decision
    rule uses the adjusted p-value.

    python -m experiments.encoder_comparison.collect_grid
    python -m experiments.encoder_comparison.collect_grid --csv-out paper/encoder_grid.csv
    python -m experiments.encoder_comparison.collect_grid --reveal-heldout   # only once, at the end
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.encoder_comparison.families import (ARCHITECTURE, DEEP,  # noqa: E402
                                                     FAMILIES, FINGERPRINTS, ORDER, all_cells,
                                                     other_pooling)
from experiments.encoder_comparison.reps import secondary_params  # noqa: E402

GRID_ROOT = os.path.join(PROJECT_ROOT, "experiments", "results", "encoder_grid")
NONE_ROOT = os.path.join(PROJECT_ROOT, "experiments", "results", "encoder_comparison")
SEEDS = [42, 43, 44, 45, 46]
METRIC = "pr_auc"

# Measured on this project's own shared-seed data; see the module docstring.
SD_PAIRED_DIFF = 0.0132

# final_project full_5seeds mean validation PR-AUC for Morgan+MACCS, for the reproduction check.
FINAL_PROJECT_FULL = 0.710


# --------------------------------------------------------------------------- loading
def _read_run(root: str, run: str, require_sentinel: bool):
    d = os.path.join(root, run)
    if require_sentinel and not os.path.exists(os.path.join(d, "COMPLETE")):
        return None
    vm = os.path.join(d, "val_metrics.json")
    if not os.path.exists(vm):
        return None
    with open(vm) as f:
        metrics = json.load(f)
    info = {}
    ri = os.path.join(d, "run_info.json")
    if os.path.exists(ri):
        with open(ri) as f:
            info = json.load(f)
    return metrics, info


def load_results(seeds=SEEDS) -> pd.DataFrame:
    """One row per completed run, from both roots."""
    rows = []

    # the already-trained `none` column (no sentinel: those runs predate it, and all 5 seeds of
    # all 9 families are verified complete by run_all_seeds.is_done)
    for family in ORDER:
        for seed in seeds:
            got = _read_run(NONE_ROOT, f"{family}_seed{seed}", require_sentinel=False)
            if got is None:
                continue
            metrics, info = got
            rows.append({
                "cell": family, "block": "none", "primary": family, "secondary": None,
                "pooling": FAMILIES[family]["pooling"], "no_cross_attn": False,
                "seed": seed, "val_pr_auc": metrics[METRIC], "val_mcc": metrics.get("mcc"),
                "params_trainable": info.get("parameters_trainable"),
                "description_block_params": 0 if FAMILIES[family]["kind"] == "seq"
                                            else FAMILIES[family]["primary_dim"] * 64 + 64,
                "root": "encoder_comparison",
            })

    # every new cell
    known = {c.name: c for c in all_cells()}
    for name, cell in known.items():
        for seed in seeds:
            got = _read_run(GRID_ROOT, f"{name}_seed{seed}", require_sentinel=True)
            if got is None:
                continue
            metrics, info = got
            rows.append({
                "cell": name, "block": cell.block, "primary": cell.primary,
                "secondary": cell.secondary, "pooling": cell.pooling,
                "no_cross_attn": cell.no_cross_attn,
                "seed": seed, "val_pr_auc": metrics[METRIC], "val_mcc": metrics.get("mcc"),
                "params_trainable": info.get("parameters_trainable"),
                "description_block_params": cell.extra_block_params(),
                "root": "encoder_grid",
            })
    return pd.DataFrame(rows)


def cell_series(df: pd.DataFrame, cell: str) -> dict:
    """{seed: val PR-AUC} for one cell."""
    sub = df[df["cell"] == cell]
    return dict(zip(sub["seed"], sub["val_pr_auc"]))


# --------------------------------------------------------------------------- statistics
def paired_test(a: dict, b: dict) -> dict:
    """Paired comparison of two cells over the seeds they share.

    Paired t-test and Wilcoxon signed-rank, as the plan specifies both. With 5 pairs Wilcoxon has
    very little resolution (its smallest attainable p is 0.0625), so the t-test p-value is what the
    decision rule uses and the Wilcoxon p is reported beside it as a distribution-free check.
    """
    shared = sorted(set(a) & set(b))
    out = {"n_seeds": len(shared), "mean_a": np.nan, "mean_b": np.nan, "delta": np.nan,
           "sd_diff": np.nan, "p_t": np.nan, "p_wilcoxon": np.nan}
    if len(shared) < 2:
        return out
    x = np.array([a[s] for s in shared], dtype=float)
    y = np.array([b[s] for s in shared], dtype=float)
    d = x - y
    out["mean_a"], out["mean_b"] = float(x.mean()), float(y.mean())
    out["delta"] = float(d.mean())
    out["sd_diff"] = float(d.std(ddof=1))
    try:
        from scipy import stats
        if out["sd_diff"] > 0:
            out["p_t"] = float(stats.ttest_rel(x, y).pvalue)
            try:
                out["p_wilcoxon"] = float(stats.wilcoxon(x, y).pvalue)
            except ValueError:
                pass
        else:
            out["p_t"] = 0.0 if out["delta"] != 0 else 1.0
    except ImportError:
        # t statistic by hand, p left as NaN rather than guessed
        pass
    return out


def benjamini_hochberg(pvals, q=0.10):
    """Adjusted p-values (step-up), and the reject/keep decision at level q.

    Applied across the whole family of 72 'does a secondary help?' tests, because at p < 0.05 about
    3.6 of them pass by luck alone.
    """
    p = np.asarray([np.nan if v is None else v for v in pvals], dtype=float)
    ok = ~np.isnan(p)
    adj = np.full(p.shape, np.nan)
    if not ok.any():
        return adj, np.zeros(p.shape, dtype=bool)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = min(prev, p[i] * m / rank)
        adj[i] = val
        prev = val
    return adj, (adj <= q) & ok


def detectable_delta(n_seeds: int, sd: float = SD_PAIRED_DIFF, power=0.80, alpha=0.05) -> float:
    """Smallest paired difference this many seeds can find, at 80% power.

    Solved exactly against the noncentral t distribution, because with 4 degrees of freedom the
    normal approximation is optimistic by a third (0.017 against the true 0.022) and this number
    is the bound a null result is reported with. Falls back to the normal approximation, clearly
    the more generous of the two, only if scipy is unavailable.
    """
    if n_seeds < 2:
        return float("nan")
    try:
        from scipy import optimize, stats

        def achieved(delta):
            df = n_seeds - 1
            ncp = delta / (sd / math.sqrt(n_seeds))
            crit = stats.t.ppf(1 - alpha / 2, df)
            return stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)

        return float(optimize.brentq(lambda x: achieved(x) - power, 1e-6, 1.0))
    except Exception:
        z_a, z_b = 1.959964, 0.841621      # two-sided alpha 0.05, power 0.80
        return (z_a + z_b) * sd / math.sqrt(n_seeds)


# --------------------------------------------------------------------------- the matrix
def matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Mean validation PR-AUC, primaries down the side, secondaries across the top.

    The pooling and no-cross-attention controls are left out: they are not secondary columns, they
    are alternative versions of a primary, and putting them in this table would double-count a row.
    """
    wanted = ["none"] + ORDER + ["shuf_maccs", "shuf_morgan"]
    pairs = df[df["primary"].notna() & ~df["block"].isin(["pooling", "no_cross_attn"])]
    if pairs.empty:
        return pd.DataFrame(index=ORDER)
    g = (pairs.assign(col=lambda d: d["secondary"].fillna("none"))
              .groupby(["primary", "col"])["val_pr_auc"].mean().unstack())
    return g.reindex(index=ORDER, columns=[c for c in wanted if c in g.columns])


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """How many seeds each cell has, so a half-finished sweep cannot be mistaken for a result."""
    g = df.groupby(["block", "cell"])["seed"].nunique().reset_index(name="seeds")
    expected = {c.name: 5 for c in all_cells()}
    expected.update({f: 5 for f in ORDER})
    g["expected"] = g["cell"].map(expected).fillna(5).astype(int)
    g["complete"] = g["seeds"] >= g["expected"]
    return g.sort_values(["block", "cell"])


# --------------------------------------------------------------------------- the 11 tests
def test_floor(df):
    """1. Is any molecular description worth anything at all?"""
    floor = cell_series(df, "floor")
    rows = []
    for f in ORDER:
        t = paired_test(cell_series(df, f), floor)
        rows.append({"method": f, **t})
    return pd.DataFrame(rows)


def test_pooling(df):
    """2. Which pooling for each deep method, decided on 5 seeds instead of 1."""
    rows = []
    for f in DEEP:
        alt = other_pooling(f)
        alt_cell = f"{f}__pool_{'gated' if alt == 'global_gated_attention' else 'cls'}"
        t = paired_test(cell_series(df, alt_cell), cell_series(df, f))
        rows.append({"method": f, "current": FAMILIES[f]["pooling"], "alternative": alt,
                     "alt_cell": alt_cell, **t})
    return pd.DataFrame(rows)


def test_cross_attention(df):
    """3. Is the encoder good, or is cross-attention good?"""
    rows = []
    for f in DEEP:
        t = paired_test(cell_series(df, f), cell_series(df, f"{f}__noca"))
        rows.append({"method": f, "with_ca_minus_without": t["delta"], **t})
    return pd.DataFrame(rows)


def test_fair_comparison(df):
    """4. Which description is best with the fusion held equal. Descriptive ranking, no test."""
    rows = []
    for f in FINGERPRINTS:
        s = cell_series(df, f)
        rows.append({"method": f, "fusion": "concat", "cell": f,
                     "mean": np.mean(list(s.values())) if s else np.nan, "n_seeds": len(s)})
    for f in DEEP:
        s = cell_series(df, f"{f}__noca")
        rows.append({"method": f, "fusion": "concat (no cross-attn)", "cell": f"{f}__noca",
                     "mean": np.mean(list(s.values())) if s else np.nan, "n_seeds": len(s)})
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


def test_secondary_helps(df, q=0.10):
    """5. Does a secondary help? The 72 pair cells, with BH correction across all of them."""
    rows = []
    for p in ORDER:
        base = cell_series(df, p)
        for s in ORDER:
            if s == p:
                continue
            cell = f"{p}__x_{s}"
            t = paired_test(cell_series(df, cell), base)
            rows.append({"cell": cell, "primary": p, "secondary": s,
                         "secondary_kind": "fingerprint" if s in FINGERPRINTS else "encoder",
                         "secondary_params": secondary_params(s), **t})
    out = pd.DataFrame(rows)
    if len(out):
        adj, rej = benjamini_hochberg(out["p_t"].tolist(), q=q)
        out["p_adj_bh"], out["passes_bh"] = adj, rej
    return out


def test_chemistry_or_params(df):
    """6. Is a gain chemistry, or just the extra parameters?"""
    rows = []
    for p in ORDER:
        for s in ORDER:
            if s == p:
                continue
            pair = cell_series(df, f"{p}__x_{s}")
            if not pair:
                continue
            row = {"cell": f"{p}__x_{s}", "primary": p, "secondary": s}
            for ctrl in ("shuf_maccs", "shuf_morgan"):
                t = paired_test(pair, cell_series(df, f"{p}__x_{ctrl}"))
                row[f"delta_vs_{ctrl}"] = t["delta"]
                row[f"p_vs_{ctrl}"] = t["p_t"]
            rows.append(row)
    return pd.DataFrame(rows)


def test_fp_vs_encoder_secondary(df):
    """7. Is a cheap fingerprint as good as a second encoder?

    Reported per primary with the counts, because the diagonal makes the columns uneven: a
    fingerprint row has 3 fingerprint and 5 encoder columns, a deep row has 4 and 4.
    """
    rows = []
    for p in ORDER:
        base = cell_series(df, p)
        base_mean = np.mean(list(base.values())) if base else np.nan
        rec = {"primary": p, "none_mean": base_mean}
        for kind, names in (("fp", FINGERPRINTS), ("emb", DEEP)):
            gains, n = [], 0
            for s in names:
                if s == p:
                    continue
                t = paired_test(cell_series(df, f"{p}__x_{s}"), base)
                if t["n_seeds"]:
                    gains.append(t["delta"])
                    n += 1
            rec[f"{kind}_columns"] = n
            rec[f"{kind}_mean_gain"] = float(np.mean(gains)) if gains else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def test_order(df):
    """8. Does order matter? A+B against B+A, because 64 wide is not 32 wide."""
    rows = []
    for i, a in enumerate(ORDER):
        for b in ORDER[i + 1:]:
            t = paired_test(cell_series(df, f"{a}__x_{b}"), cell_series(df, f"{b}__x_{a}"))
            rows.append({"a_primary": a, "b_primary": b,
                         "cell_ab": f"{a}__x_{b}", "cell_ba": f"{b}__x_{a}", **t})
    return pd.DataFrame(rows)


def test_two_encoder(df):
    """9. The 20 deep+deep cells, split by whether the two encoders share an architecture."""
    rows = []
    for p in DEEP:
        base = cell_series(df, p)
        for s in DEEP:
            if s == p:
                continue
            t = paired_test(cell_series(df, f"{p}__x_{s}"), base)
            rows.append({"cell": f"{p}__x_{s}", "primary": p, "secondary": s,
                         "pairing": ("same_architecture"
                                     if ARCHITECTURE[p] == ARCHITECTURE[s]
                                     else "cross_architecture"), **t})
    out = pd.DataFrame(rows)
    if len(out):
        summary = out.groupby("pairing")["delta"].agg(["count", "mean", "std"])
        return out, summary
    return out, pd.DataFrame()


def test_marginals(df):
    """10. Best primary over all columns, best secondary over all rows, and the interaction.

    A two-way model with seed as a random effect needs statsmodels; where it is not installed the
    marginal means are still reported, which is what the ranking is read from.
    """
    pairs = df[(df["secondary"].notna())
               & (~df["secondary"].isin(["shuf_maccs", "shuf_morgan"]))].copy()
    by_primary = pairs.groupby("primary")["val_pr_auc"].agg(["count", "mean", "std"])
    by_secondary = pairs.groupby("secondary")["val_pr_auc"].agg(["count", "mean", "std"])
    anova = None
    try:
        import statsmodels.formula.api as smf
        m = smf.mixedlm("val_pr_auc ~ C(primary) * C(secondary)", pairs, groups=pairs["seed"]).fit()
        anova = m.summary().as_text()
    except Exception as e:
        anova = f"(two-way model not fitted: {type(e).__name__}: {e})"
    return by_primary, by_secondary, anova


def test_reproduction(df):
    """11. Does this code path agree with final_project on the same model?"""
    s = cell_series(df, "morgan__x_maccs")
    mean = np.mean(list(s.values())) if s else np.nan
    return {"cell": "morgan__x_maccs", "n_seeds": len(s), "grid_mean": mean,
            "final_project_full_5seeds": FINAL_PROJECT_FULL,
            "difference": mean - FINAL_PROJECT_FULL if s else np.nan}


# --------------------------------------------------------------------------- decision rule
def decide(df, helps: pd.DataFrame, chem: pd.DataFrame, ca: pd.DataFrame, alpha=0.05):
    """Part 7's rule, applied exactly as written and before any held-out number is read.

    1. a pair wins only if it beats `none` (BH-adjusted) AND both shuffled controls,
    2. cross-attention is kept for a deep method only if it beats its own no-CA version,
    3. the winner is the best mean among the singles, the no-CA singles and the passing pairs,
       with ties inside one seed-SD broken by fewest trainable parameters,
    4. if no pair passes, keep the single and state the power bound.
    """
    lines = []
    passing = []
    if len(helps) and "passes_bh" in helps:
        ctrl = chem.set_index("cell") if len(chem) else pd.DataFrame()
        for _, r in helps.iterrows():
            if not bool(r.get("passes_bh")) or not (r["delta"] > 0):
                continue
            if r["cell"] not in ctrl.index:
                continue
            c = ctrl.loc[r["cell"]]
            beats_both = all(
                (c.get(f"delta_vs_{k}") is not None)
                and (c.get(f"delta_vs_{k}") > 0)
                and (c.get(f"p_vs_{k}") is not None)
                and (not np.isnan(c.get(f"p_vs_{k}")))
                and (c.get(f"p_vs_{k}") < alpha)
                for k in ("shuf_maccs", "shuf_morgan"))
            if beats_both:
                passing.append(r["cell"])

    lines.append(f"rule 1 - pairs beating `none` (BH q=0.10) AND both shuffled controls: "
                 f"{len(passing)}")
    # Rule 1 is a significance test, as pre-registered, and significance is not size. A cell can
    # clear it on a gain smaller than this study can resolve, so the size is printed beside each
    # winner and flagged when it falls under the power bound. The rule is NOT changed here - that
    # would be moving the goalposts after seeing the data - but a winner carrying this flag should
    # not be described as an improvement without the extra seeds.
    n_seeds = int(df.groupby("cell")["seed"].nunique().max() or 5)
    bound = detectable_delta(max(n_seeds, 2))
    deltas = helps.set_index("cell")["delta"].to_dict() if len(helps) else {}
    for c in passing:
        d = deltas.get(c, float("nan"))
        flag = "  <-- BELOW the power bound; needs seeds 47-51 before it is called a gain" \
               if (not np.isnan(d) and d < bound) else ""
        lines.append(f"           {c:34s} +{d:.4f} vs none{flag}")
    if passing:
        lines.append(f"           (power bound at {n_seeds} seeds: {bound:.4f})")

    keep_ca = []
    for _, r in ca.iterrows():
        if r["n_seeds"] and not np.isnan(r["p_t"]) and r["delta"] > 0 and r["p_t"] < alpha:
            keep_ca.append(r["method"])
    lines.append(f"rule 2 - deep methods where cross-attention earns its place: "
                 f"{', '.join(keep_ca) if keep_ca else 'none'}")

    candidates = {}
    for f in ORDER:
        s = cell_series(df, f)
        if s:
            candidates[f] = np.mean(list(s.values()))
    for f in DEEP:
        s = cell_series(df, f"{f}__noca")
        if s:
            candidates[f"{f}__noca"] = np.mean(list(s.values()))
    for c in passing:
        s = cell_series(df, c)
        if s:
            candidates[c] = np.mean(list(s.values()))

    if candidates:
        best = max(candidates, key=candidates.get)
        best_mean = candidates[best]
        sd = df[df["cell"] == best]["val_pr_auc"].std(ddof=1)
        sd = 0.0 if np.isnan(sd) else sd
        tied = [c for c, m in candidates.items() if m >= best_mean - sd]
        params = {c: df[df["cell"] == c]["params_trainable"].dropna().min() for c in tied}
        winner = min(tied, key=lambda c: (params.get(c) if params.get(c) is not None else 1e18))
        lines.append(f"rule 3 - best mean {best} at {best_mean:.4f}; "
                     f"{len(tied)} cell(s) within one seed-SD ({sd:.4f}); "
                     f"fewest parameters among them: {winner}")
    else:
        lines.append("rule 3 - no completed cells to choose from yet")

    if not passing:
        n = int(df.groupby("cell")["seed"].nunique().max() or 0)
        bound = detectable_delta(max(n, 2))
        lines.append(f"rule 4 - no pair passed. The honest statement is: no second description "
                     f"helps by more than about {bound:.3f} validation PR-AUC at {n} seeds. "
                     "That is NOT 'a second description does not help'.")
    lines.append("rule 5 - unseal the 22 held-out pairs only after the choice above is locked "
                 "(--reveal-heldout).")
    return "\n".join(lines)


# --------------------------------------------------------------------------- held-out
def reveal_heldout(seeds=SEEDS) -> pd.DataFrame:
    """Read the sealed held-out metrics. Run this ONCE, after the decision is locked."""
    rows = []
    for root, names in ((NONE_ROOT, [f"{f}_seed{s}" for f in ORDER for s in seeds]),
                        (GRID_ROOT, [f"{c.name}_seed{s}" for c in all_cells() for s in seeds])):
        for run in names:
            path = os.path.join(root, "_heldout_sealed", run, "heldout_metrics.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                m = json.load(f)
            cell, _, seed = run.rpartition("_seed")
            rows.append({"cell": cell, "seed": int(seed), "heldout_pr_auc": m[METRIC],
                         "heldout_mcc": m.get("mcc"), "heldout_correct": m.get("correct"),
                         "heldout_n": m.get("n")})
    df = pd.DataFrame(rows)
    if len(df) and df["heldout_n"].nunique() > 1:
        print(f"WARNING: held-out row counts differ across runs: "
              f"{sorted(df['heldout_n'].unique())}. Held-out is 22 pairs.")
    return df


# --------------------------------------------------------------------------- report
def main():
    p = argparse.ArgumentParser(description="Collect the encoder grid and run the 11 tests.")
    p.add_argument("--csv-out", default=None, help="write the per-run table here")
    p.add_argument("--matrix-out", default=None, help="write the mean-PR-AUC matrix here")
    p.add_argument("--q", type=float, default=0.10, help="Benjamini-Hochberg level (default 0.10)")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--reveal-heldout", action="store_true",
                   help="unseal the 22 held-out pairs. Only after the decision is locked.")
    args = p.parse_args()

    df = load_results(args.seeds)
    if df.empty:
        raise SystemExit(f"no completed runs found under {GRID_ROOT} or {NONE_ROOT}")

    pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 200)

    cov = coverage(df)
    n_cells = cov["cell"].nunique()
    n_done = int(cov["complete"].sum())
    print("=" * 100)
    print(f"GRID COVERAGE  -  {len(df)} runs over {n_cells} cells; "
          f"{n_done} cells have all 5 seeds, {n_cells - n_done} are partial")
    print("=" * 100)
    print(cov.groupby("block")[["seeds", "complete"]].agg({"seeds": "sum", "complete": "sum"}))
    partial = cov[~cov["complete"]]
    if len(partial):
        print(f"\npartial cells ({len(partial)}) - not usable for a paired test yet:")
        print(partial[["block", "cell", "seeds", "expected"]].to_string(index=False))

    # Cells with no runs at all never reach `coverage`, so they would otherwise be invisible and
    # the matrix would look complete with a hole in it. Name them.
    have = set(cov["cell"]) | set(ORDER)
    absent = [c for c in all_cells() if c.name not in have]
    if absent:
        by_method = {}
        for c in absent:
            for m in (c.primary, c.secondary):
                if m:
                    by_method[m] = by_method.get(m, 0) + 1
        print(f"\nNOT STARTED: {len(absent)} of {len(all_cells())} cells have no runs at all "
              f"({len(absent) * 5} runs). This grid is INCOMPLETE; every table below is over what "
              "exists, not over the design.")
        worst = sorted(by_method.items(), key=lambda kv: -kv[1])[:5]
        print("  methods most affected: "
              + ", ".join(f"{m} ({n} cells)" for m, n in worst))

    print("\n" + "=" * 100)
    print("MATRIX  -  mean validation PR-AUC, rows = primary, columns = secondary")
    print("=" * 100)
    mat = matrix(df)
    print(mat.round(4).to_string())

    print("\n" + "=" * 100)
    print(f"POWER  -  at 5 seeds and sd {SD_PAIRED_DIFF}, the smallest detectable paired "
          f"difference is {detectable_delta(5):.3f}")
    print(f"         at 10 seeds it is {detectable_delta(10):.3f}. Any apparent winner gets "
          "5 more seeds (47-51).")
    print("=" * 100)

    sections = [
        ("1  FLOOR - is any molecular description worth anything?", test_floor(df)),
        ("2  POOLING - which pooling per deep method, on 5 seeds", test_pooling(df)),
        ("3  CROSS-ATTENTION - the encoder, or the fusion?", test_cross_attention(df)),
        ("4  FAIR COMPARISON - all 9 with the fusion held equal", test_fair_comparison(df)),
    ]
    helps = test_secondary_helps(df, q=args.q)
    chem = test_chemistry_or_params(df)
    ca = test_cross_attention(df)
    sections += [
        (f"5  DOES A SECONDARY HELP? - 72 cells, BH FDR q={args.q}", helps),
        ("6  CHEMISTRY OR PARAMETERS? - against both shuffled controls", chem),
        ("7  FINGERPRINT vs ENCODER SECONDARY - per row, with counts",
         test_fp_vs_encoder_secondary(df)),
        ("8  DOES ORDER MATTER? - A+B vs B+A", test_order(df)),
    ]
    two_enc, two_enc_summary = test_two_encoder(df)
    sections += [("9  TWO-ENCODER MODELS - cross- vs same-architecture", two_enc)]

    for title, table in sections:
        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
        if isinstance(table, pd.DataFrame) and table.empty:
            print("(no completed cells in this block yet)")
        else:
            print(table.round(4).to_string(index=False))

    if len(two_enc_summary):
        print("\n  summary by pairing:")
        print(two_enc_summary.round(4).to_string())

    print("\n" + "=" * 100)
    print("10  MARGINALS - best primary overall, best secondary overall")
    print("=" * 100)
    by_p, by_s, anova = test_marginals(df)
    print("by primary (over all non-control secondary columns):")
    print(by_p.round(4).to_string())
    print("\nby secondary (over all primary rows):")
    print(by_s.round(4).to_string())
    print(f"\n{anova}")

    print("\n" + "=" * 100)
    print("11  REPRODUCTION - this code path vs final_project on the same model")
    print("=" * 100)
    rep = test_reproduction(df)
    for k, v in rep.items():
        print(f"  {k:28s} {v}")
    if rep["n_seeds"] and abs(rep["difference"]) > 0.02:
        print("  NOTE: the two code paths disagree by more than 0.02. This grid's numbers are "
              "canonical for this comparison; say so rather than mixing the tables.")

    print("\n" + "=" * 100)
    print("DECISION RULE  (fixed before any run; see Part 7)")
    print("=" * 100)
    print(decide(df, helps, chem, ca))

    if args.csv_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)) or ".", exist_ok=True)
        df.to_csv(args.csv_out, index=False)
        print(f"\nper-run table -> {args.csv_out}")
    if args.matrix_out:
        mat.to_csv(args.matrix_out)
        print(f"matrix        -> {args.matrix_out}")

    if args.reveal_heldout:
        print("\n" + "=" * 100)
        print("HELD-OUT, UNSEALED  -  22 pairs. This should be read once, after the decision.")
        print("=" * 100)
        ho = reveal_heldout(args.seeds)
        if ho.empty:
            print("(nothing sealed yet)")
        else:
            print(ho.groupby("cell")[["heldout_pr_auc", "heldout_mcc", "heldout_correct"]]
                    .mean().round(4).sort_values("heldout_pr_auc", ascending=False).to_string())
    else:
        print("\nheld-out stays sealed. Pass --reveal-heldout once the choice is locked.")


if __name__ == "__main__":
    main()
