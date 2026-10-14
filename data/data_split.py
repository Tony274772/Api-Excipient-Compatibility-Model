"""
Leakage-safe split for API-Excipient compatibility data (start_dataset.csv).

This is the split strategy that produced data/train.csv, data/val.csv and data/test.csv
(originally mlp+molformer+asl/data/data_split.py). Running --mode 602020 reproduces those
three files byte-for-byte.

Grouping strategy:
  - Primary group key: API_CID (no API's pairs are allowed to cross splits).
  - Diagnostic layer: Butina clustering on API Morgan fingerprints (Tanimoto >= 0.85)
    to detect near-duplicate APIs (different CIDs, near-identical structures, e.g. salt
    forms). If more than MERGE_THRESHOLD_PCT of APIs merge, the Butina cluster is used as
    the group key instead of raw API_CID, so near-duplicates can't cross splits either.
  - Excipients are intentionally NOT grouped (excipient overlap across splits mirrors the
    deployment scenario where the same excipient vocabulary recurs). The split therefore
    measures generalisation to new APIs, not to new excipients.

Modes:
  602020  train / val / test (60/20/20), stratified on Outcome1, grouped as above.
          StratifiedGroupKFold(5) -> test fold, then StratifiedGroupKFold(4) on the rest -> val.
  8020    train / val (80/20) using the same first StratifiedGroupKFold(5) step:
          val = the 602020 test fold, train = the 602020 train + val folds.
          The 24-pair held-out set is the external test set for this mode, so every row
          that matches a held-out pair (same API + same excipient after salt stripping and
          stereo removal) is removed from train and val. Removed rows are written to
          removed_heldout_overlap.csv. Use --keep_heldout_overlap to disable this.
          Descriptor normalisation stats are recomputed from the 80/20 train rows only.

Usage (from the project root):
    python data/data_split.py --mode 602020                      # verify / rebuild data/{train,val,test}.csv
    python data/data_split.py --mode 8020                        # -> data/split_8020/{train,val}.csv
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold

RDLogger.DisableLog("rdApp.*")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DATA_DIR)
INPUT_PATH = os.path.join(DATA_DIR, "start_dataset.csv")
HELDOUT_PATH = os.path.join(PROJECT_ROOT, "held_out_testset", "held_out_test_set.csv")
RANDOM_STATE = 42
BUTINA_CUTOFF = 0.15          # distance cutoff -> Tanimoto similarity >= 0.85
MERGE_THRESHOLD_PCT = 5.0     # only switch to cluster-based grouping if >this% of APIs merge

log = logging.getLogger("data_split")


# ---------------------------------------------------------------------------
# Grouping (Butina near-duplicate check on unique APIs)
# ---------------------------------------------------------------------------
def assign_split_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Add butina_cluster and split_group columns. Returns (df, group key used)."""
    api_df = df[["API_CID", "API_Smiles"]].drop_duplicates().reset_index(drop=True)

    mols, valid_idx = [], []
    for i, smi in enumerate(api_df["API_Smiles"]):
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)
            valid_idx.append(i)

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]

    dists = []
    n = len(fps)
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, BUTINA_CUTOFF, isDistData=True)

    cluster_id_by_pos = {}
    for cid, members in enumerate(clusters):
        for m_idx in members:
            cluster_id_by_pos[m_idx] = cid

    api_df["butina_cluster"] = -1
    for local_pos, orig_idx in enumerate(valid_idx):
        api_df.loc[orig_idx, "butina_cluster"] = cluster_id_by_pos[local_pos]
    # any invalid SMILES fall back to their own singleton cluster (own row index)
    invalid_mask = api_df["butina_cluster"] == -1
    api_df.loc[invalid_mask, "butina_cluster"] = api_df.loc[invalid_mask].index + 10_000_000

    cluster_sizes = api_df.groupby("butina_cluster")["API_CID"].nunique()
    multi_clusters = cluster_sizes[cluster_sizes > 1]
    n_apis_merged = int(multi_clusters.sum())
    pct_merged = 100 * n_apis_merged / len(api_df)

    log.info(f"Unique APIs                     : {len(api_df)}")
    log.info(f"Butina clusters (>=0.85 sim)    : {api_df['butina_cluster'].nunique()}")
    log.info(f"Clusters with >1 API_CID        : {len(multi_clusters)}")
    log.info(f"APIs involved in a merged cluster: {n_apis_merged} ({pct_merged:.1f}% of all APIs)")

    if pct_merged > MERGE_THRESHOLD_PCT:
        group_col_name = "butina_cluster"
        log.info(f"-> {pct_merged:.1f}% > {MERGE_THRESHOLD_PCT}% threshold: using Butina cluster as the grouping key.")
    else:
        group_col_name = "API_CID"
        log.info(f"-> Only {pct_merged:.1f}% of APIs affected (<= {MERGE_THRESHOLD_PCT}%): using plain API_CID as the grouping key.")

    df = df.merge(api_df[["API_CID", "butina_cluster"]], on="API_CID", how="left")
    df["split_group"] = df[group_col_name] if group_col_name == "butina_cluster" else df["API_CID"]
    return df, group_col_name


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def first_fold(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """StratifiedGroupKFold(5) first fold: (80% part, 20% part)."""
    sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    big_idx, small_idx = next(sgkf1.split(df, y=df["Outcome1"], groups=df["split_group"]))
    return df.iloc[big_idx].reset_index(drop=True), df.iloc[small_idx].reset_index(drop=True)


def split_602020(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    trainval_df, test_df = first_fold(df)
    # split trainval (80% of data) into train(60%) / val(20%) -> val is 1/4 of trainval
    sgkf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sgkf2.split(trainval_df, y=trainval_df["Outcome1"], groups=trainval_df["split_group"]))
    return {
        "train": trainval_df.iloc[train_idx].reset_index(drop=True),
        "val": trainval_df.iloc[val_idx].reset_index(drop=True),
        "test": test_df,
    }


def split_8020(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_df, val_df = first_fold(df)
    return {"train": train_df, "val": val_df}


# ---------------------------------------------------------------------------
# Held-out overlap (8020 mode only)
# ---------------------------------------------------------------------------
def _parent_key(smiles) -> str:
    """Largest fragment, stereo removed, canonical SMILES (salt / hydrate / stereo blind)."""
    smiles = "" if pd.isna(smiles) else str(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    frag = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumHeavyAtoms())
    Chem.RemoveStereochemistry(frag)
    return Chem.MolToSmiles(frag)


def remove_heldout_overlap(splits: dict[str, pd.DataFrame], heldout_csv: str) -> tuple[dict, pd.DataFrame]:
    """Drop rows whose (API, excipient) pair matches a held-out pair by CID or by parent molecule."""
    ho = pd.read_csv(heldout_csv)
    ho_cids = set(zip(ho["API_CID"], ho["Excipient_CID"]))
    ho_parents = set(zip(ho["API_Smiles"].map(_parent_key), ho["Excipient_Smiles"].map(_parent_key)))

    removed = []
    for name, d in splits.items():
        parents = list(zip(d["API_Smiles"].map(_parent_key), d["Excipient_Smiles"].map(_parent_key)))
        hit = [((a, e) in ho_cids) or (p in ho_parents) for a, e, p in zip(d["API_CID"], d["Excipient_CID"], parents)]
        hit = pd.Series(hit, index=d.index)
        if hit.any():
            r = d[hit].copy()
            r.insert(0, "removed_from", name)
            removed.append(r)
        splits[name] = d[~hit].reset_index(drop=True)
    removed_df = pd.concat(removed, ignore_index=True) if removed else pd.DataFrame()
    return splits, removed_df


# ---------------------------------------------------------------------------
# Verification + summary
# ---------------------------------------------------------------------------
def verify_and_summarise(splits: dict[str, pd.DataFrame], n_total: int) -> None:
    names = list(splits)
    leak = False
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            g = set(splits[a]["split_group"]) & set(splits[b]["split_group"])
            c = set(splits[a]["API_CID"]) & set(splits[b]["API_CID"])
            log.info(f"Group overlap {a}<->{b}: {len(g)} groups | API_CID overlap: {len(c)}")
            leak = leak or bool(g) or bool(c)
    if leak:
        raise RuntimeError("LEAKAGE DETECTED -- an API group crosses splits. Split not written.")
    log.info("No group leakage: every API (or API cluster) appears in exactly one split.")

    exc = {k: set(v["Excipient_CID"]) for k, v in splits.items()}
    last = names[-1]
    log.info(f"Excipient_CID overlap train<->{last} (expected, NOT leakage): "
             f"{len(exc['train'] & exc[last])} / {len(exc[last])} {last} excipients also seen in train")

    for name, d in splits.items():
        n = len(d)
        pos = int((d["Outcome1"] == 1).sum())
        log.info(f"{name.upper():5s} | rows={n:5d} ({n / n_total:5.1%}) | pos={pos:4d} ({pos / n:5.1%}) | "
                 f"neg={n - pos:4d} | unique APIs={d['API_CID'].nunique():4d} | unique excipients={d['Excipient_CID'].nunique():4d}")


def _md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def write_splits(splits: dict[str, pd.DataFrame], out_dir: str, force: bool) -> None:
    """Write CSVs. Existing files are only replaced when --force is given and they differ."""
    os.makedirs(out_dir, exist_ok=True)
    cols_to_drop = ["butina_cluster", "split_group"]
    with tempfile.TemporaryDirectory() as tmp:
        changed = []
        for name, d in splits.items():
            tmp_path = os.path.join(tmp, f"{name}.csv")
            d.drop(columns=cols_to_drop).to_csv(tmp_path, index=False)
            dst = os.path.join(out_dir, f"{name}.csv")
            if os.path.exists(dst) and _md5(dst) == _md5(tmp_path):
                log.info(f"{dst}: unchanged (identical to existing file)")
                continue
            changed.append((tmp_path, dst))
        existing = [dst for _, dst in changed if os.path.exists(dst)]
        if existing and not force:
            raise SystemExit(
                f"Refusing to overwrite files that differ from this split: {existing}. "
                "Every model trained on them would no longer match. Re-run with --force to replace them."
            )
        for tmp_path, dst in changed:
            with open(tmp_path, "rb") as src, open(dst, "wb") as out:
                out.write(src.read())
            log.info(f"{dst}: written ({sum(1 for _ in open(dst)) - 1} rows)")


def write_norm_stats(train_df: pd.DataFrame, out_path: str) -> None:
    """Descriptor normalisation stats from the train split only (same method as src/descriptors.py)."""
    sys.path.insert(0, PROJECT_ROOT)
    from src.descriptors import compute_norm_stats
    api_desc = pd.read_csv(os.path.join(DATA_DIR, "api_descriptors.csv"))
    exc_desc = pd.read_csv(os.path.join(DATA_DIR, "excipient_descriptors.csv"))
    stats = compute_norm_stats(train_df, api_desc, exc_desc)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"{out_path}: descriptor normalisation stats from {len(train_df)} train rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Grouped, stratified API-level data split.")
    p.add_argument("--mode", choices=["602020", "8020"], default="602020")
    p.add_argument("--input", default=INPUT_PATH)
    p.add_argument("--out_dir", default=None,
                   help="default: data/ for 602020, data/split_8020/ for 8020")
    p.add_argument("--heldout_csv", default=HELDOUT_PATH)
    p.add_argument("--keep_heldout_overlap", action="store_true",
                   help="8020 only: keep rows that match a held-out pair (not recommended)")
    p.add_argument("--force", action="store_true", help="overwrite existing split files that differ")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or (DATA_DIR if args.mode == "602020" else os.path.join(DATA_DIR, "split_8020"))
    os.makedirs(out_dir, exist_ok=True)
    log_name = "splitlog.log" if args.mode == "602020" else "splitlog_8020.log"
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(out_dir, log_name), mode="w")])

    df = pd.read_csv(args.input)
    n_total = len(df)
    n_pos = int((df["Outcome1"] == 1).sum())
    log.info(f"Mode: {args.mode} | input: {args.input} | output: {out_dir}")
    log.info(f"Total pairs: {n_total} | incompatible (1): {n_pos} ({n_pos / n_total:.1%}) | "
             f"unique API_CID: {df['API_CID'].nunique()} | unique Excipient_CID: {df['Excipient_CID'].nunique()}")

    df, group_key = assign_split_groups(df)
    splits = split_602020(df) if args.mode == "602020" else split_8020(df)

    if args.mode == "8020" and not args.keep_heldout_overlap:
        splits, removed = remove_heldout_overlap(splits, args.heldout_csv)
        log.info(f"Removed {len(removed)} rows matching a held-out pair (by CID or salt/stereo-blind parent):")
        for _, r in removed.iterrows():
            log.info(f"   [{r['removed_from']}] API_CID {r['API_CID']} + Excipient_CID {r['Excipient_CID']} label={r['Outcome1']}")
        if len(removed):
            removed.drop(columns=["butina_cluster", "split_group"]).to_csv(
                os.path.join(out_dir, "removed_heldout_overlap.csv"), index=False)

    verify_and_summarise(splits, n_total)
    write_splits(splits, out_dir, force=args.force)
    if args.mode == "8020":
        write_norm_stats(splits["train"], os.path.join(out_dir, "descriptor_norm_stats.json"))
    log.info(f"grouping key used: {group_key}")


if __name__ == "__main__":
    main()
