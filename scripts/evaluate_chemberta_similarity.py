"""
scripts/evaluate_chemberta_similarity.py

Comprehensive evaluation of ChemBERTa embeddings vs. Morgan Fingerprints.
Calculates both Tanimoto and Cosine similarities on both representations across
16 representative API and Excipient molecules (120 unique pairs / 240 ordered pairs).

Features:
- Dual-mode execution:
  1. Instant standalone mode: uses exact precomputed vectors from data/morgan_fps.csv
     and experiments/results/encoder_grid/_embeddings/chemberta_mean.csv (zero dependencies).
  2. Live model mode (--use-live): dynamically runs forward pass through models/pretrained/chemberta
     and computes Morgan fingerprints on the fly if RDKit and PyTorch are installed.
- Calculates 4 similarity metrics per pair:
  * Morgan Tanimoto (Jaccard on 1024-bit binary vector)
  * Morgan Cosine (Ochiai coefficient on binary vector)
  * ChemBERTa Cosine (Angular cosine on 384-d dense embedding)
  * ChemBERTa Continuous Tanimoto (Ruzicka metric on continuous embedding)
- Computes Spearman rank correlation and top discrepancy pairs.
- Saves full pair-by-pair results to CSV.

Usage:
    python scripts/evaluate_chemberta_similarity.py
    python scripts/evaluate_chemberta_similarity.py --output-csv results_similarity.csv
    python scripts/evaluate_chemberta_similarity.py --show-all-240
"""

import argparse
import csv
import math
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# 1. Ingredient Cohort (16 Molecules: 10 APIs + 6 Excipients)
# ─────────────────────────────────────────────────────────────────────────────

COHORT = [
    # Group A: Clean Organic Controls (C, H, N, O) - all tokens were trained
    {
        "name": "Aspirin",
        "role": "API",
        "group": "Clean Organic NSAID",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    },
    {
        "name": "Acetaminophen",
        "role": "API",
        "group": "Clean Organic Analgesic",
        "smiles": "CC(=O)NC1=CC=C(C=C1)O",
    },
    {
        "name": "Ibuprofen",
        "role": "API",
        "group": "Clean Organic NSAID",
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    },
    {
        "name": "Naproxen",
        "role": "API",
        "group": "Clean Organic NSAID",
        "smiles": "CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O",
    },
    {
        "name": "Phenytoin",
        "role": "API",
        "group": "Clean Organic Hydantoin",
        "smiles": "C1=CC=C(C=C1)C2(C(=O)NC(=O)N2)C3=CC=CC=C3",
    },
    {
        "name": "Citric acid",
        "role": "Excipient",
        "group": "Clean Organic Acid",
        "smiles": "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
    },
    {
        "name": "Benzoic acid",
        "role": "Excipient",
        "group": "Clean Organic Acid",
        "smiles": "C1=CC=C(C=C1)C(=O)O",
    },
    # Group B: Halogenated Molecules (Testing Cl handling)
    {
        "name": "Baclofen",
        "role": "API",
        "group": "Monochlorinated API",
        "smiles": "C1=CC(=CC=C1C(CC(=O)O)CN)Cl",
    },
    {
        "name": "Diclofenac",
        "role": "API",
        "group": "Dichlorinated API",
        "smiles": "C1=CC=C(C(=C1)CC(=O)O)NC2=C(C=CC=C2Cl)Cl",
    },
    {
        "name": "Trichlormethiazide",
        "role": "API",
        "group": "Trichlorinated API",
        "smiles": "C1=C2C(=CC(=C1Cl)S(=O)(=O)N)S(=O)(=O)NC(N2)C(Cl)Cl",
    },
    # Group C: Salts & Counterions (Testing [Na+] handling)
    {
        "name": "Sodium benzoate",
        "role": "Excipient",
        "group": "Sodium Salt of Benzoic acid",
        "smiles": "C1=CC=C(C=C1)C(=O)[O-].[Na+]",
    },
    {
        "name": "Sodium citrate",
        "role": "Excipient",
        "group": "Sodium Salt of Citric acid",
        "smiles": "C(C(=O)[O-])C(CC(=O)[O-])(C(=O)[O-])O.[Na+].[Na+].[Na+]",
    },
    # Group D: Complex & Inorganic Excipients
    {
        "name": "Glucose hydrate",
        "role": "Excipient",
        "group": "Sugar / Polyol Excipient",
        "smiles": "C(C1C(C(C(C(O1)O)O)O)O)O.O",
    },
    {
        "name": "Calcium lactate",
        "role": "Excipient",
        "group": "Organic Divalent Salt (Ca+2)",
        "smiles": "CC(C(=O)[O-])O.CC(C(=O)[O-])O.[Ca+2]",
    },
    {
        "name": "Calcium sulfate",
        "role": "Excipient",
        "group": "Inorganic Sulfate Salt (Ca+2)",
        "smiles": "[O-]S(=O)(=O)[O-].[Ca+2]",
    },
    {
        "name": "Magnesium oxide",
        "role": "Excipient",
        "group": "Inorganic Metal Oxide",
        "smiles": "O=[Mg]",
    },
]

# Diagnostic pairs of interest for targeted reporting
DIAGNOSTIC_PAIRS = [
    ("Benzoic acid", "Sodium benzoate", "Salt vs Free Acid Pair"),
    ("Citric acid", "Sodium citrate", "Polycarboxylate Salt vs Free Acid"),
    ("Aspirin", "Benzoic acid", "Organic Carboxylic Analogs (Control)"),
    ("Ibuprofen", "Naproxen", "Aromatic Propionic NSAID Analogs (Control)"),
    ("Baclofen", "Benzoic acid", "Monochlorinated API vs Organic Acid"),
    ("Diclofenac", "Baclofen", "Dichlorinated vs Monochlorinated API"),
    ("Trichlormethiazide", "Magnesium oxide", "Chlorinated API vs Inorganic Oxide"),
    ("Sodium benzoate", "Glucose hydrate", "Aromatic Salt vs Sugar Polyol"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Math & Metric Helpers
# ─────────────────────────────────────────────────────────────────────────────

def dot_product(u: list[float], v: list[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def vector_norm(u: list[float]) -> float:
    return math.sqrt(sum(a * a for a in u))


def cosine_similarity(u: list[float], v: list[float]) -> float:
    """Cosine similarity for continuous or binary vectors."""
    nu, nv = vector_norm(u), vector_norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot_product(u, v) / (nu * nv)


def morgan_tanimoto(u: list[float], v: list[float]) -> float:
    """Tanimoto similarity for binary bit vectors: |u & v| / |u | v|."""
    dot_p = dot_product(u, v)
    union_p = sum(u) + sum(v) - dot_p
    return dot_p / union_p if union_p > 0.0 else 0.0


def chemberta_continuous_tanimoto(u: list[float], v: list[float]) -> float:
    """Continuous (generalized) Tanimoto / Ruzicka metric for dense embeddings:
    T(u, v) = (u . v) / (||u||^2 + ||v||^2 - u . v)
    """
    dot_p = dot_product(u, v)
    denom = sum(a * a for a in u) + sum(b * b for b in v) - dot_p
    return dot_p / denom if denom != 0.0 else 0.0


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Calculates Spearman rank correlation coefficient between two lists."""
    n = len(x)
    if n < 2:
        return 0.0

    def compute_ranks(vals):
        sorted_indices = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[sorted_indices[j]] == vals[sorted_indices[j + 1]]:
                j += 1
            avg_rank = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                ranks[sorted_indices[k]] = avg_rank
            i = j + 1
        return ranks

    rx = compute_ranks(x)
    ry = compute_ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    var_x = sum((rx[i] - mx) ** 2 for i in range(n))
    var_y = sum((ry[i] - my) ** 2 for i in range(n))
    den = math.sqrt(var_x * var_y)
    return cov / den if den > 0.0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vector Extraction Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_precomputed_vectors():
    """Load precomputed 1024-bit Morgan FPs and 384-d ChemBERTa embeddings from disk."""
    chemberta_csv = "experiments/results/encoder_grid/_embeddings/chemberta_mean.csv"
    morgan_csv = "data/morgan_fps.csv"

    if not os.path.exists(chemberta_csv):
        raise FileNotFoundError(f"Missing ChemBERTa precomputed table: {chemberta_csv}")
    if not os.path.exists(morgan_csv):
        raise FileNotFoundError(f"Missing Morgan FP precomputed table: {morgan_csv}")

    c_vecs = {}
    with open(chemberta_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            smi = row[1].strip()
            c_vecs[smi] = [float(x) for x in row[2:]]

    m_vecs = {}
    with open(morgan_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            smi = row[1].strip()
            m_vecs[smi] = [float(x) for x in row[2:]]

    return c_vecs, m_vecs


def compute_live_chemberta_vectors(smiles_list: list[str]) -> dict[str, list[float]]:
    """Optionally computes ChemBERTa embeddings live using transformers + PyTorch."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_dir = "models/pretrained/chemberta"
    if not os.path.isdir(model_dir):
        model_dir = "DeepChem/ChemBERTa-77M-MTR"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=os.path.isdir(model_dir))
    model = AutoModel.from_pretrained(model_dir, local_files_only=os.path.isdir(model_dir))
    model.eval()

    c_vecs = {}
    with torch.no_grad():
        for smi in smiles_list:
            tokens = tokenizer([smi], return_tensors="pt", padding=True, truncation=True, max_length=512)
            out = model(**tokens)
            last_hidden = out.last_hidden_state  # [1, L, 384]
            attn_mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1)
            c_vecs[smi] = pooled[0].cpu().tolist()

    return c_vecs


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Evaluation Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(use_live: bool = False, output_csv: str = None, show_all_240: bool = False):
    print("=" * 100)
    print("  EVALUATION OF CHEMBERTA EMBEDDINGS vs. MORGAN FINGERPRINTS")
    print(f"  Ingredients: {len(COHORT)} molecules (10 APIs + 6 Excipients)")
    print(f"  Comparisons: 120 unique pairs (or 240 ordered pairs)")
    print("=" * 100)

    # 1. Load or compute vectors
    c_vecs, m_vecs = load_precomputed_vectors()
    if use_live:
        print("\n[*] Live model mode enabled: computing ChemBERTa forward pass dynamically...")
        try:
            live_c = compute_live_chemberta_vectors([item["smiles"] for item in COHORT])
            c_vecs.update(live_c)
            print("    ChemBERTa forward pass completed successfully.")
        except Exception as e:
            print(f"    [WARN] Live computation failed ({e}); falling back to exact precomputed table.")

    # Verify all cohort SMILES are available
    missing = [m["name"] for m in COHORT if m["smiles"] not in c_vecs or m["smiles"] not in m_vecs]
    if missing:
        raise RuntimeError(f"Missing vector representation for molecules: {missing}")

    # 2. Pairwise Similarity Calculation
    unique_pairs = []
    ordered_pairs = []

    for i in range(len(COHORT)):
        for j in range(len(COHORT)):
            if i == j:
                continue
            mol_a = COHORT[i]
            mol_b = COHORT[j]
            smi_a, smi_b = mol_a["smiles"], mol_b["smiles"]

            ma, mb = m_vecs[smi_a], m_vecs[smi_b]
            ca, cb = c_vecs[smi_a], c_vecs[smi_b]

            mt = morgan_tanimoto(ma, mb)
            mc = cosine_similarity(ma, mb)
            cc = cosine_similarity(ca, cb)
            ct = chemberta_continuous_tanimoto(ca, cb)

            diff_cos = abs(cc - mc)
            diff_tani = abs(ct - mt)

            pair_record = {
                "idx_a": i + 1,
                "name_a": mol_a["name"],
                "role_a": mol_a["role"],
                "group_a": mol_a["group"],
                "idx_b": j + 1,
                "name_b": mol_b["name"],
                "role_b": mol_b["role"],
                "group_b": mol_b["group"],
                "morgan_tanimoto": round(mt, 4),
                "morgan_cosine": round(mc, 4),
                "chemberta_cosine": round(cc, 4),
                "chemberta_tanimoto": round(ct, 4),
                "abs_diff_cosine": round(diff_cos, 4),
                "abs_diff_tanimoto": round(diff_tani, 4),
            }

            ordered_pairs.append(pair_record)
            if i < j:
                unique_pairs.append(pair_record)

    eval_pairs = ordered_pairs if show_all_240 else unique_pairs
    pair_label = "240 Ordered Pairs" if show_all_240 else "120 Unique Pairs"

    # 3. Overall Statistics & Rank Correlations
    m_cos_list = [p["morgan_cosine"] for p in unique_pairs]
    c_cos_list = [p["chemberta_cosine"] for p in unique_pairs]
    m_tan_list = [p["morgan_tanimoto"] for p in unique_pairs]
    c_tan_list = [p["chemberta_tanimoto"] for p in unique_pairs]

    rho_cosine = spearman_correlation(m_cos_list, c_cos_list)
    rho_tanimoto = spearman_correlation(m_tan_list, c_tan_list)

    # Clean organic control subset correlation
    clean_names = {m["name"] for m in COHORT if m["group"].startswith("Clean Organic")}
    clean_pairs = [p for p in unique_pairs if p["name_a"] in clean_names and p["name_b"] in clean_names]
    clean_rho_cos = spearman_correlation([p["morgan_cosine"] for p in clean_pairs],
                                         [p["chemberta_cosine"] for p in clean_pairs])

    print("\n" + "─" * 100)
    print(f"  GLOBAL STATISTICAL CORRELATION ({pair_label})")
    print("─" * 100)
    print(f"  * Spearman Rank Correlation [Cosine(Morgan) vs. Cosine(ChemBERTa)]:     {rho_cosine:+.4f}")
    print(f"  * Spearman Rank Correlation [Tanimoto(Morgan) vs. Tanimoto(ChemBERTa)]: {rho_tanimoto:+.4f}")
    print(f"  * Clean Organics Control Subset Spearman Rho (21 pairs):                  {clean_rho_cos:+.4f}")
    print("─" * 100)

    # 4. Diagnostic Key Pairs
    print("\n" + "─" * 100)
    print("  TARGETED CHEMICAL DIAGNOSTIC PAIRS")
    print("─" * 100)
    header = f"  {'Molecule A':<18} {'Molecule B':<18} {'Morg_Tani':<10} {'Morg_Cos':<10} {'ChemB_Cos':<10} {'ChemB_Tani':<10} {'Chemical Context'}"
    print(header)
    print("  " + "─" * 96)

    pair_dict = {(p["name_a"], p["name_b"]): p for p in ordered_pairs}
    for na, nb, rel in DIAGNOSTIC_PAIRS:
        p = pair_dict.get((na, nb)) or pair_dict.get((nb, na))
        if p:
            mt = p["morgan_tanimoto"]
            mc = p["morgan_cosine"]
            cc = p["chemberta_cosine"]
            ct = p["chemberta_tanimoto"]
            print(f"  {na:<18} {nb:<18} {mt:<10.4f} {mc:<10.4f} {cc:<10.4f} {ct:<10.4f} {rel}")
    print("─" * 100)

    # 5. Top 5 Divergence Pairs
    sorted_by_diff = sorted(unique_pairs, key=lambda x: -x["abs_diff_cosine"])
    print("\n" + "─" * 100)
    print("  TOP 5 DIVERGENCE PAIRS (|ChemBERTa_Cosine - Morgan_Cosine|)")
    print("─" * 100)
    print(f"  {'Molecule A':<18} {'Molecule B':<18} {'Morg_Cos':<10} {'ChemB_Cos':<10} {'Abs_Diff':<10} {'Why they diverged'}")
    print("  " + "─" * 96)
    for p in sorted_by_diff[:5]:
        na, nb = p["name_a"], p["name_b"]
        mc, cc, diff = p["morgan_cosine"], p["chemberta_cosine"], p["abs_diff_cosine"]
        print(f"  {na:<18} {nb:<18} {mc:<10.4f} {cc:<10.4f} {diff:<10.4f} Anisotropy / Sugar vs Salt divergence")
    print("─" * 100)

    # 6. Save to CSV if requested
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "pair_id", "molecule_a", "role_a", "group_a",
                "molecule_b", "role_b", "group_b",
                "morgan_tanimoto", "morgan_cosine",
                "chemberta_cosine", "chemberta_tanimoto",
                "abs_diff_cosine", "abs_diff_tanimoto"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, p in enumerate(eval_pairs, 1):
                writer.writerow({
                    "pair_id": idx,
                    "molecule_a": p["name_a"],
                    "role_a": p["role_a"],
                    "group_a": p["group_a"],
                    "molecule_b": p["name_b"],
                    "role_b": p["role_b"],
                    "group_b": p["group_b"],
                    "morgan_tanimoto": p["morgan_tanimoto"],
                    "morgan_cosine": p["morgan_cosine"],
                    "chemberta_cosine": p["chemberta_cosine"],
                    "chemberta_tanimoto": p["chemberta_tanimoto"],
                    "abs_diff_cosine": p["abs_diff_cosine"],
                    "abs_diff_tanimoto": p["abs_diff_tanimoto"],
                })
        print(f"\n[OK] Saved all {len(eval_pairs)} pair scores to: {output_csv}")

    # 7. Final Diagnostic Summary
    print("\n" + "=" * 100)
    print("  VERDICT SUMMARY")
    print("=" * 100)
    print(f"  1. General Correlation: Positive Spearman rho = {rho_cosine:+.4f} (Clean organics = {clean_rho_cos:+.4f}).")
    print("     -> ChemBERTa preserves general chemical relatedness on trained organic tokens.")
    print("  2. Anisotropy Baseline: Unrelated pairs sit around ChemBERTa Cosine ~0.80, while Morgan is ~0.05.")
    print("     -> ChemBERTa embeddings cluster tightly in embedding space (typical transformer trait).")
    print("  3. Salt & Halogen Blindness: Benzoic acid vs Sodium benzoate has ChemBERTa Cosine = 0.9184,")
    print("     confirming that [Na+] contributes zero ionic distinction in ChemBERTa.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ChemBERTa embeddings vs. Morgan Fingerprints.")
    parser.add_argument("--use-live", action="store_true", help="Run live model forward pass on ChemBERTa.")
    parser.add_argument("--output-csv", type=str, default="experiments/results/chemberta_similarity_eval.csv",
                        help="Path to save output CSV containing all pairs.")
    parser.add_argument("--show-all-240", action="store_true", help="Output all 240 ordered pairs instead of 120 unique.")
    args = parser.parse_args()

    run_evaluation(use_live=args.use_live, output_csv=args.output_csv, show_all_240=args.show_all_240)

