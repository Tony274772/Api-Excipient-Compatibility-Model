"""
Comprehensive visualization suite for PGB Morgan SHAP Analysis.
Produces publication-grade figures for all 10 analysis categories.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from shap_analysis.feature_names import (
    CHEMISTRY_FLAG_BASE_NAMES,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    MACRO_GROUPS,
    MECHANISM_BASE_NAMES,
    PGB_DESCRIPTOR_BASE_NAMES,
    PRIOR_BASE_NAMES,
    get_feature_group,
)

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#ebebeb",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

PALETTE_GROUPS = {
    "api_primary": "#1f77b4",
    "exc_primary": "#aec7e8",
    "api_secondary": "#ff7f0e",
    "exc_secondary": "#ffbb78",
    "api_pgb_desc": "#2ca02c",
    "exc_pgb_desc": "#98df8a",
    "api_salt": "#d62728",
    "exc_salt": "#ff9896",
    "api_flags": "#9467bd",
    "exc_flags": "#c5b0d5",
    "prior_vec": "#8c564b",
    "mech_flags": "#e377c2",
}


def plot_feature_group_importance(
    shap_values: np.ndarray,
    save_path: str,
) -> pd.DataFrame:
    """
    Analysis #1: Feature Group Attribution.
    Computes mean |SHAP| for each feature group and macro-group.
    """
    records = []
    total_abs = np.mean(np.abs(shap_values))

    for group_name, (start, end) in FEATURE_GROUPS.items():
        sub_shap = shap_values[:, start:end]
        mean_abs = np.mean(np.abs(sub_shap))
        sum_abs = np.sum(np.mean(np.abs(sub_shap), axis=0))
        n_features = end - start
        records.append({
            "group": group_name,
            "mean_abs_shap": mean_abs,
            "total_abs_shap": sum_abs,
            "feature_count": n_features,
            "color": PALETTE_GROUPS.get(group_name, "#555555"),
        })

    df = pd.DataFrame(records)
    total_sum = df["total_abs_shap"].sum()
    df["percentage"] = (df["total_abs_shap"] / (total_sum + 1e-12)) * 100.0
    df = df.sort_values(by="total_abs_shap", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.barh(
        df["group"],
        df["total_abs_shap"],
        color=df["color"],
        edgecolor="none",
        height=0.65,
    )

    for bar, pct, n_feat in zip(bars, df["percentage"], df["feature_count"]):
        width = bar.get_width()
        ax.text(
            width + 0.005 * df["total_abs_shap"].max(),
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}% ({n_feat} dims)",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )

    ax.set_title("PGB Morgan Feature Group Importance (Total |SHAP| Contribution)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Cumulative Absolute SHAP Value (Impact on Logit)", fontsize=11)
    ax.set_xlim(0, df["total_abs_shap"].max() * 1.25)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return df


def plot_top_features_bar(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int = 25,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Ranks all 2,468 individual features by mean |SHAP| and plots top K.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_k]

    records = []
    for idx in top_indices:
        group = get_feature_group(idx)
        records.append({
            "feature_index": idx,
            "feature_name": feature_names[idx],
            "mean_abs_shap": mean_abs[idx],
            "group": group,
            "color": PALETTE_GROUPS.get(group, "#333333"),
        })

    df = pd.DataFrame(records).iloc[::-1]  # Reverse for barh

    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
    bars = ax.barh(df["feature_name"], df["mean_abs_shap"], color=df["color"], height=0.65)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.005 * df["mean_abs_shap"].max(),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#444444",
        )

    ax.set_title(f"Top {top_k} Most Influential Features (PGB Morgan Model)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Absolute SHAP Value", fontsize=11)
    ax.set_xlim(0, df["mean_abs_shap"].max() * 1.18)

    # Custom legend for groups
    unique_groups = df["group"].unique()
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE_GROUPS.get(g, "#888")) for g in unique_groups]
    ax.legend(handles, unique_groups, title="Feature Group", loc="lower right", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    return df


def plot_descriptor_beeswarm_and_bar(
    shap_values: np.ndarray,
    raw_desc_api: np.ndarray,
    raw_desc_exc: np.ndarray,
    save_path: str,
):
    """
    Analysis #3: Physicochemical Descriptor Attribution.
    Creates side-by-side beeswarm and importance bar charts for API & Excipient descriptors.
    """
    api_start, api_end = FEATURE_GROUPS["api_pgb_desc"]
    exc_start, exc_end = FEATURE_GROUPS["exc_pgb_desc"]

    api_shap = shap_values[:, api_start:api_end]
    exc_shap = shap_values[:, exc_start:exc_end]

    api_mean_abs = np.mean(np.abs(api_shap), axis=0)
    exc_mean_abs = np.mean(np.abs(exc_shap), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=300, sharey=False)

    # Left: API Descriptors
    order_api = np.argsort(api_mean_abs)
    names_api = [PGB_DESCRIPTOR_BASE_NAMES[i] for i in order_api]
    vals_api = api_mean_abs[order_api]

    axes[0].barh(names_api, vals_api, color="#2ca02c", height=0.6, alpha=0.85)
    for i, v in enumerate(vals_api):
        axes[0].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8.5, fontweight="bold")
    axes[0].set_title("API Physicochemical Descriptors", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mean |SHAP| Value", fontsize=10)

    # Right: Excipient Descriptors
    order_exc = np.argsort(exc_mean_abs)
    names_exc = [PGB_DESCRIPTOR_BASE_NAMES[i] for i in order_exc]
    vals_exc = exc_mean_abs[order_exc]

    axes[1].barh(names_exc, vals_exc, color="#98df8a", height=0.6, alpha=0.9, edgecolor="#2ca02c")
    for i, v in enumerate(vals_exc):
        axes[1].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8.5, fontweight="bold")
    axes[1].set_title("Excipient Physicochemical Descriptors", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Mean |SHAP| Value", fontsize=10)

    fig.suptitle("PGB Morgan Physicochemical Descriptor Attribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_chemistry_flags_attribution(
    shap_values: np.ndarray,
    save_path: str,
):
    """
    Analysis #4: Chemistry Flag Attribution.
    Analyzes 18 binary chemistry flags for API and Excipient.
    """
    api_start, api_end = FEATURE_GROUPS["api_flags"]
    exc_start, exc_end = FEATURE_GROUPS["exc_flags"]

    api_shap = shap_values[:, api_start:api_end]
    exc_shap = shap_values[:, exc_start:exc_end]

    api_mean_abs = np.mean(np.abs(api_shap), axis=0)
    exc_mean_abs = np.mean(np.abs(exc_shap), axis=0)

    df = pd.DataFrame({
        "flag": CHEMISTRY_FLAG_BASE_NAMES,
        "api_importance": api_mean_abs,
        "exc_importance": exc_mean_abs,
    })
    df["total"] = df["api_importance"] + df["exc_importance"]
    df = df.sort_values(by="total", ascending=True)

    y = np.arange(len(df))
    height = 0.38

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.barh(y + height / 2, df["api_importance"], height, label="API Chemistry Flag", color="#9467bd", alpha=0.9)
    ax.barh(y - height / 2, df["exc_importance"], height, label="Excipient Chemistry Flag", color="#c5b0d5", alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(df["flag"], fontsize=9.5)
    ax.set_xlabel("Mean Absolute SHAP Value", fontsize=11)
    ax.set_title("Impact of 18 Chemistry Flags on Compatibility Prediction", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_mechanism_flags_attribution(
    shap_values: np.ndarray,
    X: np.ndarray,
    save_path: str,
):
    """
    Analysis #5: Mechanism Flag Attribution.
    Examines the 5 chemical degradation mechanism flags.
    """
    start, end = FEATURE_GROUPS["mech_flags"]
    mech_shap = shap_values[:, start:end]
    mech_x = X[:, start:end]

    mean_abs = np.mean(np.abs(mech_shap), axis=0)
    active_impact = []
    inactive_impact = []

    for j in range(5):
        active_mask = (mech_x[:, j] > 0.5)
        if np.any(active_mask):
            active_impact.append(np.mean(mech_shap[active_mask, j]))
        else:
            active_impact.append(0.0)

        inactive_mask = (mech_x[:, j] <= 0.5)
        if np.any(inactive_mask):
            inactive_impact.append(np.mean(mech_shap[inactive_mask, j]))
        else:
            inactive_impact.append(0.0)

    clean_names = [m.replace("_", " ") for m in MECHANISM_BASE_NAMES]
    df = pd.DataFrame({
        "mechanism": clean_names,
        "mean_abs_shap": mean_abs,
        "impact_when_present": active_impact,
        "impact_when_absent": inactive_impact,
    }).sort_values(by="mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars = ax.barh(df["mechanism"], df["mean_abs_shap"], color="#e377c2", height=0.55, edgecolor="#bc6ca7")

    for bar, active_eff in zip(bars, df["impact_when_present"]):
        width = bar.get_width()
        sign = "+" if active_eff >= 0 else ""
        ax.text(
            width + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"|SHAP|: {width:.4f} (Active effect: {sign}{active_eff:.3f})",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#333333",
        )

    ax.set_title("Attribution of Chemical Degradation Reaction Mechanisms", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Mean Absolute SHAP Value", fontsize=10)
    ax.set_xlim(0, max(df["mean_abs_shap"].max() * 1.5, 0.02))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_prior_attribution(
    shap_values: np.ndarray,
    X: np.ndarray,
    save_path: str,
):
    """
    Analysis #6: Prior Vector Attribution.
    Examines [p_exact, p_delta, p_family, log_n, unseen].
    """
    start, end = FEATURE_GROUPS["prior_vec"]
    p_shap = shap_values[:, start:end]
    p_x = X[:, start:end]

    mean_abs = np.mean(np.abs(p_shap), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Bar chart of 5 components
    axes[0].barh(PRIOR_BASE_NAMES, mean_abs, color="#8c564b", height=0.55, alpha=0.9)
    for i, v in enumerate(mean_abs):
        axes[0].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9, fontweight="bold")
    axes[0].set_title("Excipient Prior Components Importance", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mean |SHAP| Value", fontsize=10)

    # Dependence plot of p_exact vs its SHAP value
    p_exact = p_x[:, 0]
    p_exact_shap = p_shap[:, 0]
    unseen = p_x[:, 4]

    scatter = axes[1].scatter(
        p_exact,
        p_exact_shap,
        c=unseen,
        cmap="coolwarm",
        alpha=0.75,
        edgecolors="none",
        s=35,
    )
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label("Unseen Excipient Flag (1=New, 0=Known)", fontsize=9)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[1].set_title("Prior p_exact vs. SHAP Contribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Historical Excipient Incompatibility Rate (p_exact)", fontsize=10)
    axes[1].set_ylabel("SHAP Value on Logit", fontsize=10)

    fig.suptitle("Bayesian Excipient Prior Influence Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_sample_waterfall(
    shap_values: np.ndarray,
    base_value: float,
    feature_values: np.ndarray,
    feature_names: List[str],
    sample_idx: int,
    sample_meta: Dict[str, Any],
    top_k: int = 15,
    save_path: Optional[str] = None,
):
    """
    Analysis #7: Per-Sample Waterfall Plot.
    Explains a specific API-Excipient pair prediction with individual push effects.
    """
    sample_shap = shap_values[sample_idx]
    sample_x = feature_values[sample_idx]

    order = np.argsort(np.abs(sample_shap))[::-1][:top_k]

    names = [feature_names[i] for i in order]
    vals = [sample_shap[i] for i in order]
    feat_vals = [sample_x[i] for i in order]

    # Cumulative calculation
    total_shap = np.sum(sample_shap)
    final_logit = base_value + total_shap
    final_prob = 1.0 / (1.0 + np.exp(-final_logit))

    # Reverse order for bottom-to-top rendering
    names.reverse()
    vals.reverse()
    feat_vals.reverse()

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]

    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, vals, color=colors, height=0.6, alpha=0.9)

    ax.set_yticks(y_pos)
    labels = [f"{n} = {fv:.2f}" if abs(fv) > 0.001 else f"{n} = 0" for n, fv in zip(names, feat_vals)]
    ax.set_yticklabels(labels, fontsize=9)

    ax.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)

    pred_label = "INCOMPATIBLE" if final_prob >= 0.5 else "COMPATIBLE"
    true_label = "INCOMPATIBLE" if sample_meta.get("label", 0) == 1 else "COMPATIBLE"

    api_disp = sample_meta.get('api_name') if sample_meta.get('api_name') else sample_meta.get('api_smiles', 'N/A')[:28] + '...'
    exc_disp = sample_meta.get('exc_name') if sample_meta.get('exc_name') else sample_meta.get('exc_smiles', 'N/A')[:25] + '...'

    ax.set_title(
        f"SHAP Waterfall: Sample #{sample_idx}\n"
        f"API: {api_disp} | Exc: {exc_disp}\n"
        f"Base logit: {base_value:.2f} → Output: {final_logit:.2f} (P(Incompat) = {final_prob:.1%}) | Truth: {true_label}",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("SHAP Value (Push toward Incompatibility: Red > 0, Blue < 0)", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def plot_morgan_top_bits(
    shap_values: np.ndarray,
    feature_names: List[str],
    decoded_bits: Optional[Dict[int, Dict[str, Any]]] = None,
    top_k: int = 20,
    save_path: Optional[str] = None,
):
    """
    Analysis #2: Individual Morgan Fingerprint Bit Attribution with Substructure Annotations.
    """
    api_start, api_end = FEATURE_GROUPS["api_primary"]
    exc_start, exc_end = FEATURE_GROUPS["exc_primary"]

    morgan_indices = list(range(api_start, api_end)) + list(range(exc_start, exc_end))
    mean_abs = np.mean(np.abs(shap_values[:, morgan_indices]), axis=0)

    top_local = np.argsort(mean_abs)[::-1][:top_k]
    top_global = [morgan_indices[i] for i in top_local]

    labels = []
    vals = []
    colors = []

    for g_idx in top_global:
        name = feature_names[g_idx]
        val = np.mean(np.abs(shap_values[:, g_idx]))
        vals.append(val)
        colors.append("#1f77b4" if "API" in name else "#aec7e8")

        bit_num = int(name.split("_")[-1])
        if decoded_bits and bit_num in decoded_bits:
            sub_smi = decoded_bits[bit_num].get("top_fragment_smiles", "")
            labels.append(f"{name} [{sub_smi[:20]}]")
        else:
            labels.append(name)

    labels.reverse()
    vals.reverse()
    colors.reverse()

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.barh(labels, vals, color=colors, height=0.6)

    ax.set_title(f"Top {top_k} Morgan Fingerprint Bits (Substructural Attribution)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Absolute SHAP Value", fontsize=11)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
        plt.Rectangle((0, 0), 1, 1, color="#aec7e8"),
    ]
    ax.legend(handles, ["API Morgan Bit", "Excipient Morgan Bit"], loc="lower right", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def plot_tower_level_attribution(
    tower_shap: np.ndarray,
    save_path: str,
):
    """
    Analysis #9: Tower-Level Attribution.
    Compares API Tower (96-d) vs Excipient Tower (96-d) vs Prior & Gating.
    """
    h_api_shap = tower_shap[:, 0:96]
    h_exc_shap = tower_shap[:, 96:192]
    prior_shap = tower_shap[:, 192:197]
    api_flags_shap = tower_shap[:, 197:215]
    exc_flags_shap = tower_shap[:, 215:233]
    mech_flags_shap = tower_shap[:, 233:238]

    components = {
        "API Molecule Tower (96-d)": np.sum(np.mean(np.abs(h_api_shap), axis=0)),
        "Excipient Molecule Tower (96-d)": np.sum(np.mean(np.abs(h_exc_shap), axis=0)),
        "Bayesian Prior Vector (5-d)": np.sum(np.mean(np.abs(prior_shap), axis=0)),
        "API Chemistry Flags (18-d)": np.sum(np.mean(np.abs(api_flags_shap), axis=0)),
        "Excipient Chemistry Flags (18-d)": np.sum(np.mean(np.abs(exc_flags_shap), axis=0)),
        "Mechanism Reactions (5-d)": np.sum(np.mean(np.abs(mech_flags_shap), axis=0)),
    }

    df = pd.DataFrame(list(components.items()), columns=["Component", "Cumulative_SHAP"])
    df["Percentage"] = (df["Cumulative_SHAP"] / df["Cumulative_SHAP"].sum()) * 100.0
    df = df.sort_values(by="Cumulative_SHAP", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    bars = ax.barh(df["Component"], df["Cumulative_SHAP"], color="#4c72b0", height=0.6)

    for bar, pct in zip(bars, df["Percentage"]):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontweight="bold")

    ax.set_title("Tower-Level Information Bottleneck Attribution", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Cumulative Absolute SHAP Value at PGBHead Input", fontsize=10)
    ax.set_xlim(0, df["Cumulative_SHAP"].max() * 1.15)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_class_comparison(
    shap_values: np.ndarray,
    y_true: np.ndarray,
    feature_names: List[str],
    top_k: int = 15,
    save_path: Optional[str] = None,
):
    """
    Analysis #10: Incompatible vs. Compatible Class Comparison.
    Highlights key discriminative features separating incompatible from compatible pairs.
    """
    incompat_mask = (y_true == 1.0)
    compat_mask = (y_true == 0.0)

    if not np.any(incompat_mask) or not np.any(compat_mask):
        return

    incompat_shap = np.mean(shap_values[incompat_mask], axis=0)
    compat_shap = np.mean(shap_values[compat_mask], axis=0)

    # Rank by difference in mean attribution
    diff = np.abs(incompat_shap - compat_shap)
    top_idx = np.argsort(diff)[::-1][:top_k]

    df = pd.DataFrame({
        "feature": [feature_names[i] for i in top_idx],
        "incompatible_shap": incompat_shap[top_idx],
        "compatible_shap": compat_shap[top_idx],
    }).iloc[::-1]

    y = np.arange(len(df))
    height = 0.38

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.barh(y + height / 2, df["incompatible_shap"], height, label="Incompatible Pairs (Outcome=1)", color="#d62728", alpha=0.85)
    ax.barh(y - height / 2, df["compatible_shap"], height, label="Compatible Pairs (Outcome=0)", color="#1f77b4", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"], fontsize=9.5)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Directional Mean SHAP Value", fontsize=11)
    ax.set_title(f"Class-Stratified Feature Signatures: Top {top_k} Differential Drivers", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
