"""
Audit and curate the raw API-excipient dataset (data/raw/start_dataset.csv).

The raw file is byte-identical to dataset.csv of the "Towards safer..." repository
(github.com/phuongnvp/drug_exp_compatibility). This script never changes it. It writes:

    data/processed/clean.csv          default curated set  (remove + repair + collapse duplicates)
    data/processed/clean_strict.csv   clean.csv minus rows that carry a "strict" flag (see STRICT_FLAGS)
    data/curation/audit/              everything that was removed, collapsed, repaired or flagged, and why
        audit_log.txt                 the numbers quoted in docs/data_curation.md
        removed_rows.csv              rows dropped, with the category and rationale
        repaired_rows.csv             rows whose SMILES was replaced
        collapsed_duplicates.csv      rows dropped because an identical model input (same structures) already exists
        parent_label_conflicts.csv    same API parent + same excipient parent, different labels (kept, reported)
        molecule_table.csv            every molecule x role: name, rows, incompatible, categories, structure checks

Inputs besides the raw file:
    data/curation/molecule_curation.csv          hand-curated decisions, one line per (cid, role, category)
    data/curation/pubchem_snapshot_2026-11-11.csv PubChem titles / formulas / SMILES for every CID (offline, frozen)

Steps, in order:
    1. strip whitespace from SMILES (4 excipient CIDs have a leading / trailing space on some rows)
    2. check every SMILES parses and matches the PubChem connectivity of its CID
    3. apply molecule_curation.csv: remove rows with a "remove" molecule, replace SMILES of "repair" molecules,
       attach "flag" categories to api_flags / exc_flags
    4. collapse rows whose (API, excipient) structures are identical after canonicalisation (the SMILES are
       stereo-blind, so e.g. mannitol / sorbitol and lactose / maltose / cellobiose are one input); if the
       labels of such a group disagreed the whole group would be dropped (none do)
    5. report, without removing, pairs whose parents (largest organic fragment, neutralised) agree but whose
       labels disagree, and the label / identity confounding of the source data

Run from any directory:
    python scripts/data/curate_dataset.py
"""

import hashlib
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # project root

from apiexc import paths  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RAW_SHA256 = "583f19c4ada9f9c2a71958e930e2591b414460892151eb5332e7898ca86f8cdb"   # upstream dataset.csv
CURATION = paths.DATA / "curation"
CURATION_CSV = CURATION / "molecule_curation.csv"
PUBCHEM_CSV = CURATION / "pubchem_snapshot_2026-11-11.csv"
AUDIT = CURATION / "audit"
PROCESSED = paths.DATA / "processed"

# flags that clean_strict.csv drops: rows that are not a (drug, excipient) pair, or whose structure is wrong
STRICT_FLAGS = {"drug_as_excipient", "non_drug_api", "multi_api", "approximate_structure", "colourant_uncertain"}
METALS = {3, 11, 12, 13, 19, 20, 22, 25, 26, 27, 29, 30, 47, 56, 57, 80, 82}   # Li Na Mg Al K Ca Ti Mn Fe Co Cu Zn Ag Ba La Hg Pb
SIDES = (("API", "API_CID", "API_Smiles"), ("Excipient", "Excipient_CID", "Excipient_Smiles"))

_lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_uncharger = rdMolStandardize.Uncharger()


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def canonical(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def parent(smiles):
    """Largest organic fragment, neutralised, stereo removed. Inorganics are returned whole (their 'parent'
    would be a bare ion that unrelated salts share, e.g. CaCO3 and La2(CO3)3)."""
    mol = Chem.MolFromSmiles(smiles)
    if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        return Chem.MolToSmiles(mol)
    p = _uncharger.uncharge(_lfc.choose(mol))
    Chem.RemoveStereochemistry(p)
    return Chem.MolToSmiles(p)


def structure_checks(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atoms = list(mol.GetAtoms())
    return {
        "n_fragments": len(Chem.GetMolFrags(mol)),
        "net_charge": Chem.GetFormalCharge(mol),
        "has_carbon": any(a.GetAtomicNum() == 6 for a in atoms),
        "has_metal": any(a.GetAtomicNum() in METALS for a in atoms),
        "has_radical": any(a.GetNumRadicalElectrons() > 0 for a in atoms),
        "has_stereo": any(c in smiles for c in "@/\\"),
        "mol_wt": round(Descriptors.MolWt(mol), 1),
    }


def load_curation():
    cur = pd.read_csv(CURATION_CSV, dtype={"repair_from_cid": "Int64"})
    bad = set(cur.action) - {"remove", "repair", "flag"}
    if bad:
        raise SystemExit(f"unknown action(s) in {CURATION_CSV.name}: {bad}")
    per_side = {}
    for side, _, _ in SIDES:
        sub = cur[cur.role.isin([side, "both"])]
        per_side[side] = {cid: g for cid, g in sub.groupby("cid")}
    return cur, per_side


def identity_auc(df, col):
    """ROC-AUC of predicting each row by the leave-one-out incompatible rate of its own API (or excipient)."""
    s = df.groupby(col).Outcome1.transform("sum")
    n = df.groupby(col).Outcome1.transform("size")
    loo = ((s - df.Outcome1) / (n - 1)).where(n > 1, df.Outcome1.mean())
    return roc_auc_score(df.Outcome1, loo)


def confounding(df, tag):
    lines = [f"[{tag}] {len(df)} rows, {int(df.Outcome1.sum())} incompatible ({df.Outcome1.mean():.1%})"]
    for side, col, _ in SIDES:
        g = df.groupby(col).Outcome1.agg(["size", "mean"])
        a0, a1 = g[g["mean"] == 0], g[g["mean"] == 1]
        mixed = g[(g["mean"] > 0) & (g["mean"] < 1)]
        lines.append(f"  {side:9s} {len(g):4d} molecules: {len(a0)} only-compatible ({int(a0['size'].sum())} rows, "
                     f"median {a0['size'].median():.0f} rows each), {len(a1)} only-incompatible "
                     f"({int(a1['size'].sum())} rows, median {a1['size'].median():.0f}), {len(mixed)} mixed "
                     f"({int(mixed['size'].sum())} rows) | leave-one-out identity ROC-AUC {identity_auc(df, col):.3f}")
    mw = df.API_Smiles.map(lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s)))
    n_rows = df.groupby("API_CID").Outcome1.transform("size")
    lines.append(f"  trivial baselines, ROC-AUC: smaller API molecular weight -> incompatible {roc_auc_score(df.Outcome1, -mw):.3f}; "
                 f"fewer rows per API -> incompatible {roc_auc_score(df.Outcome1, -n_rows):.3f}")
    return lines


def main():
    log = [f"curate_dataset.py  {datetime.now():%Y-%m-%d %H:%M}  rdkit {rdBase.rdkitVersion}  pandas {pd.__version__}", ""]
    digest = sha256(paths.RAW_DATASET)
    if digest != RAW_SHA256:
        raise SystemExit(f"{paths.rel(paths.RAW_DATASET)} sha256 {digest} is not the upstream file {RAW_SHA256}")
    raw = pd.read_csv(paths.RAW_DATASET)
    raw.insert(0, "row_id", np.arange(len(raw)))
    pubchem = pd.read_csv(PUBCHEM_CSV).set_index("CID")
    cur, cur_side = load_curation()
    log += [f"raw: {paths.rel(paths.RAW_DATASET)} sha256 {digest} (= upstream dataset.csv)",
            f"     {len(raw)} rows, {raw.API_CID.nunique()} API CIDs, {raw.Excipient_CID.nunique()} excipient CIDs, "
            f"{int(raw.Outcome1.sum())} incompatible ({raw.Outcome1.mean():.1%})",
            f"     missing values: {int(raw.isna().sum().sum())}", ""]

    # ---------------------------------------------------------------- 1. whitespace
    df = raw.copy()
    log.append("1. SMILES whitespace")
    for side, cid, smi in SIDES:
        stripped = df[smi].str.strip()
        changed = df[smi] != stripped
        log.append(f"   {side}: {int(changed.sum())} rows stripped ({df.loc[changed, cid].nunique()} CIDs: "
                   f"{sorted(df.loc[changed, cid].unique().tolist())})")
        df[smi] = stripped

    # ---------------------------------------------------------------- 2. parse + PubChem consistency + raw duplicates
    log += ["", "2. Structure validity and identifier consistency"]
    checks = {}
    for side, cid, smi in SIDES:
        bad = [s for s in df[smi].unique() if Chem.MolFromSmiles(s) is None]
        log.append(f"   {side}: {len(bad)} unparseable SMILES")
        if bad:
            raise SystemExit(f"unparseable SMILES: {bad}")
        multi = df.groupby(cid)[smi].nunique()
        log.append(f"   {side}: CIDs with more than one SMILES after stripping: {int((multi > 1).sum())}")
        mism = []
        for c, s in df[[cid, smi]].drop_duplicates().values:
            ref = Chem.MolFromSmiles(pubchem.loc[c, "ConnectivitySMILES"])
            if ref is None or Chem.MolToSmiles(ref) != canonical(s):
                mism.append(f"{c} {pubchem.loc[c, 'Title']}")
        log.append(f"   {side}: SMILES that differ from the PubChem connectivity SMILES of their CID: {len(mism)} {mism}")
        for s in df[smi].unique():
            checks.setdefault(s, structure_checks(s))
    n_stereo = sum(v["has_stereo"] for v in checks.values())
    log.append(f"   SMILES with any stereo mark: {n_stereo} of {len(checks)} (the data is stereo-blind: enantiomers, "
               f"epimers and anomers share one SMILES)")
    k = ["API_CID", "Excipient_CID"]
    pairs = set(map(tuple, df[k].values))
    reversed_pairs = sorted((a, b) for a, b in pairs if (b, a) in pairs and a < b)
    log += [f"   exact duplicate rows: {int(df.duplicated().sum())}; duplicate (API_CID, Excipient_CID) pairs: "
            f"{int(df.duplicated(k).sum())}; self pairs: {int((df.API_CID == df.Excipient_CID).sum())}",
            f"   pairs present in both orders (A,B) and (B,A): {reversed_pairs} "
            f"{[(pubchem.loc[a, 'Title'], pubchem.loc[b, 'Title']) for a, b in reversed_pairs]}",
            f"   CIDs used both as API and as excipient: {len(set(df.API_CID) & set(df.Excipient_CID))}"]

    # ---------------------------------------------------------------- 3. curation: remove / repair / flag
    log += ["", "3. Curation (data/curation/molecule_curation.csv)"]
    removed, repaired = [], []
    for side, cid, smi in SIDES:
        df[f"{side}_flags"] = ""
        df[f"{side}_remove"] = ""
    for side, cid, smi in SIDES:
        table = cur_side[side]
        for c, entries in table.items():
            rows = df[cid] == c
            if not rows.any():
                continue
            for e in entries.itertuples():
                if e.action == "remove":
                    df.loc[rows, f"{side}_remove"] = e.category
                elif e.action == "repair":
                    src = int(e.repair_from_cid) if pd.notna(e.repair_from_cid) else None
                    new = e.repair_smiles if isinstance(e.repair_smiles, str) else None
                    if new is None:
                        pool = df.loc[df[cid] == src, smi]
                        new = pool.iloc[0] if len(pool) else pubchem.loc[src, "ConnectivitySMILES"]
                    repaired.append(df.loc[rows].assign(side=side, old_smiles=df.loc[rows, smi], new_smiles=new,
                                                        rationale=e.rationale))
                    df.loc[rows, smi] = new
                    df.loc[rows, f"{side}_flags"] += "repaired;"
                else:
                    df.loc[rows, f"{side}_flags"] += e.category + ";"
    for side, _, _ in SIDES:
        df[f"{side}_flags"] = df[f"{side}_flags"].str.rstrip(";")
    drop = (df.API_remove != "") | (df.Excipient_remove != "")
    rm = df[drop].copy()
    rm["reason"] = (rm.API_remove.where(rm.API_remove == "", "API:" + rm.API_remove) + " " +
                    rm.Excipient_remove.where(rm.Excipient_remove == "", "Excipient:" + rm.Excipient_remove)).str.strip()
    rm["api_name"] = rm.API_CID.map(pubchem.Title)
    rm["excipient_name"] = rm.Excipient_CID.map(pubchem.Title)
    removed.append(rm)
    df = df[~drop].drop(columns=["API_remove", "Excipient_remove"])
    cat_rm = pd.concat([rm.API_remove[rm.API_remove != ""], rm.Excipient_remove[rm.Excipient_remove != ""]])
    log.append(f"   removed {len(rm)} rows ({int(rm.Outcome1.sum())} incompatible = "
               f"{rm.Outcome1.sum() / raw.Outcome1.sum():.1%} of all incompatible rows); by category:")
    for c, n in cat_rm.value_counts().items():
        sub = rm[(rm.API_remove == c) | (rm.Excipient_remove == c)]
        log.append(f"      {c:20s} {n:4d} rows, {int(sub.Outcome1.sum())} incompatible")
    rep = pd.concat(repaired, ignore_index=True) if repaired else pd.DataFrame()
    rep = rep[rep.row_id.isin(df.row_id)] if len(rep) else rep
    log.append(f"   repaired SMILES on {len(rep)} rows: " +
               ", ".join(f"{pubchem.loc[c, 'Title']} (CID {c}) {n} rows"
                         for c, n in rep.groupby("Excipient_CID").size().items()) if len(rep) else "   no repairs")
    log.append("   flags kept in the data (rows carrying each flag):")
    for side, _, _ in SIDES:
        f = df[f"{side}_flags"].str.split(";").explode()
        for c, n in f[f != ""].value_counts().items():
            log.append(f"      {side:9s} {c:22s} {n:4d}")

    # ---------------------------------------------------------------- 4. structure-level duplicates
    log += ["", "4. Structure-level duplicates (identical canonical API and excipient SMILES = identical model input)"]
    df["API_can"] = df.API_Smiles.map(canonical)
    df["Excipient_can"] = df.Excipient_Smiles.map(canonical)
    g = df.groupby(["API_can", "Excipient_can"]).Outcome1
    df["dup_n"] = g.transform("size")
    df["dup_conflict"] = g.transform("nunique") > 1
    dup = df[df.dup_n > 1].sort_values(["API_can", "Excipient_can", "row_id"])
    conflict = dup[dup.dup_conflict]
    keep_first = ~df.duplicated(["API_can", "Excipient_can"], keep="first")
    collapsed = df[~keep_first & ~df.dup_conflict].copy()
    kept_ids = df[keep_first].set_index(["API_can", "Excipient_can"]).row_id
    collapsed["kept_row_id"] = [kept_ids.loc[(a, e)] for a, e in zip(collapsed.API_can, collapsed.Excipient_can)]
    log += [f"   {dup.groupby(['API_can', 'Excipient_can']).ngroups} structure pairs occur more than once "
            f"({len(dup)} rows); label conflicts: {conflict.groupby(['API_can', 'Excipient_can']).ngroups}",
            f"   collapsed {len(collapsed)} rows (first row of each group kept); dropped {len(conflict)} conflicting rows"]
    df = df[keep_first & ~df.dup_conflict].copy()
    for side, cid, smi in SIDES:
        grp = df.groupby(f"{side}_can")[cid].unique()
        grp = grp[grp.str.len() > 1]
        log.append(f"   {side} structures shared by more than one CID ({len(grp)}): " +
                   "; ".join(" = ".join(pubchem.loc[list(c), "Title"].str.slice(0, 28)) for c in grp))

    # ---------------------------------------------------------------- 5. parent-level label conflicts
    log += ["", "5. Parent-level label conflicts (salt / hydrate / solvate forms of the same pair; kept, reported)"]
    df["API_parent"] = df.API_Smiles.map(parent)
    df["Excipient_parent"] = df.Excipient_Smiles.map(parent)
    gp = df.groupby(["API_parent", "Excipient_parent"]).Outcome1
    pc = df[(gp.transform("nunique") > 1)].copy()
    pc["api_name"] = pc.API_CID.map(pubchem.Title)
    pc["excipient_name"] = pc.Excipient_CID.map(pubchem.Title)
    log.append(f"   {pc.groupby(['API_parent', 'Excipient_parent']).ngroups} parent pairs with conflicting labels "
               f"({len(pc)} rows):")
    for _, sub in pc.groupby(["API_parent", "Excipient_parent"]):
        log.append("      " + " | ".join(f"{a[:24]} + {e[:26]} = {o}" for a, e, o in
                                         sub[["api_name", "excipient_name", "Outcome1"]].values))
    lact = df[df.Excipient_CID.isin([6134, 104938, 3037558, 84571, 294, 872])]
    log.append("   lactose CIDs (same structure up to hydrate water): " + ", ".join(
        f"{pubchem.loc[c, 'Title']} {len(s)} rows / {int(s.Outcome1.sum())} incompatible"
        for c, s in lact.groupby("Excipient_CID")))

    # ---------------------------------------------------------------- confounding
    log += ["", "6. Label / identity confounding (how much the molecule identity alone predicts the label)"]
    log += ["   " + x for x in confounding(raw, "raw")]
    log += ["   " + x for x in confounding(df, "clean")]

    # ---------------------------------------------------------------- outputs
    out_cols = ["row_id", "API_CID", "Excipient_CID", "Outcome1", "API_Smiles", "Excipient_Smiles", "API_flags",
                "Excipient_flags"]
    strict_mask = df.API_flags.str.split(";").map(lambda f: bool(STRICT_FLAGS & set(f))) | \
        df.Excipient_flags.str.split(";").map(lambda f: bool(STRICT_FLAGS & set(f)))
    strict = df[~strict_mask]
    log += ["", "7. Outputs",
            f"   clean.csv        {len(df)} rows, {int(df.Outcome1.sum())} incompatible ({df.Outcome1.mean():.1%}), "
            f"{df.API_CID.nunique()} APIs, {df.Excipient_CID.nunique()} excipients",
            f"   clean_strict.csv {len(strict)} rows, {int(strict.Outcome1.sum())} incompatible "
            f"({strict.Outcome1.mean():.1%}), {strict.API_CID.nunique()} APIs, {strict.Excipient_CID.nunique()} excipients"
            f"  (drops flags {sorted(STRICT_FLAGS)})"]
    log += ["   " + x for x in confounding(strict, "clean_strict")]

    held = pd.read_csv(paths.HELDOUT_CSV)
    hp = set(zip(held.API_Smiles.map(parent), held.Excipient_Smiles.map(parent)))
    log.append(f"   held-out pairs ({len(held)}) whose parent pair occurs in clean.csv: "
               f"{sum(p in hp for p in zip(df.API_parent, df.Excipient_parent))}")

    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(AUDIT, exist_ok=True)
    df[out_cols].to_csv(PROCESSED / "clean.csv", index=False)
    strict[out_cols].to_csv(PROCESSED / "clean_strict.csv", index=False)
    pd.concat(removed)[["row_id", "API_CID", "api_name", "Excipient_CID", "excipient_name", "Outcome1", "reason",
                        "API_Smiles", "Excipient_Smiles"]].to_csv(AUDIT / "removed_rows.csv", index=False)
    if len(rep):
        rep[["row_id", "side", "API_CID", "Excipient_CID", "Outcome1", "old_smiles", "new_smiles", "rationale"]] \
            .to_csv(AUDIT / "repaired_rows.csv", index=False)
    collapsed.assign(api_name=collapsed.API_CID.map(pubchem.Title), excipient_name=collapsed.Excipient_CID.map(pubchem.Title))[
        ["row_id", "kept_row_id", "API_CID", "api_name", "Excipient_CID", "excipient_name", "Outcome1"]] \
        .to_csv(AUDIT / "collapsed_duplicates.csv", index=False)
    pc[["row_id", "API_CID", "api_name", "Excipient_CID", "excipient_name", "Outcome1", "API_parent",
        "Excipient_parent"]].sort_values(["API_parent", "Excipient_parent"]).to_csv(AUDIT / "parent_label_conflicts.csv", index=False)

    mols = []
    for side, cid, smi in SIDES:
        g = raw.assign(s=raw[smi].str.strip()).groupby(cid).agg(rows=("Outcome1", "size"), incompatible=("Outcome1", "sum"),
                                                                smiles=("s", "first"))
        cats = {c: ";".join(f"{a}:{k}" for a, k in zip(x.action, x.category))
                for c, x in cur[cur.role.isin([side, "both"])].groupby("cid")}
        for c, r in g.iterrows():
            mols.append({"role": side, "cid": c, "name": pubchem.loc[c, "Title"], "formula": pubchem.loc[c, "MolecularFormula"],
                         "rows": r.rows, "incompatible": r.incompatible, "curation": cats.get(c, ""),
                         **checks[r.smiles], "smiles": r.smiles})
    pd.DataFrame(mols).to_csv(AUDIT / "molecule_table.csv", index=False)

    text = "\n".join(log)
    (AUDIT / "audit_log.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nwritten: {paths.rel(PROCESSED)}/clean.csv, clean_strict.csv and {paths.rel(AUDIT)}/")


if __name__ == "__main__":
    main()
