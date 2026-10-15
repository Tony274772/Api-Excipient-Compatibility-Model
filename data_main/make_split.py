"""
Standalone reproduction of the data split behind the 80/20 PGB Morgan model
(checkpoint checkpoints/pgb_fixed_vector_morgan_concat_asl/single_run_80_20split, trained 2026-10-06).

The split was originally made inside pgb_single_run() in src/single_run.py at git commit f534ef7
(copied unchanged in original_code_f534ef7/). This script contains only that split logic, with the
same functions and parameters, and depends on nothing else in the project:

  1. Read start_dataset.csv (3544 pairs).
  2. Butina-cluster the unique API SMILES (order of first appearance): RDKit Morgan fingerprint
     radius 2, 2048 bits (rdFingerprintGenerator), Tanimoto distance cutoff 0.15 (similarity >= 0.85).
  3. GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) with the cluster as group
     -> train (80%) / val (20%). Not stratified on the label.
  4. test = the 24-pair held-out set (ground_truth renamed to Outcome1).

Usage (from this folder):
    python make_split.py                 # writes into reproduced/ and compares with train/val/test.csv here
    python make_split.py --out_dir DIR   # write somewhere else
The original files in this folder are never overwritten.
"""

import argparse
import hashlib
import os
from datetime import datetime

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
import sklearn
from sklearn.model_selection import GroupShuffleSplit

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42  # config.seed used for the run


# --- identical to src/cross_validate.py at commit f534ef7 ---
def _morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprint(mol)


def butina_cluster(unique_smiles, dist_thresh=0.15):
    fps = [_morgan_fp(s) for s in unique_smiles]
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])
    clusters = Butina.ClusterData(dists, n, dist_thresh, isDistData=True)
    smiles_to_cluster = {}
    for cidx, members in enumerate(clusters):
        for midx in members:
            smiles_to_cluster[unique_smiles[midx]] = cidx
    return smiles_to_cluster


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _stats(name, d):
    return (f"{name:6s} rows={len(d):5d} | incompatible={int(d['Outcome1'].sum()):4d} ({d['Outcome1'].mean():5.1%}) | "
            f"compatible={int((d['Outcome1'] == 0).sum()):5d} | unique APIs={d['API_CID'].nunique():4d} | "
            f"unique excipients={d['Excipient_CID'].nunique():4d}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.path.join(HERE, "reproduced"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    lines = [f"Split reproduction run {datetime.now().isoformat(timespec='seconds')}",
             f"rdkit {rdBase.rdkitVersion} | scikit-learn {sklearn.__version__} | pandas {pd.__version__}", ""]

    # --- identical to pgb_single_run() at commit f534ef7 (cluster split branch) ---
    raw = pd.read_csv(os.path.join(HERE, "start_dataset.csv"))
    unique_apis = raw["API_Smiles"].unique().tolist()
    smiles_to_cluster = butina_cluster(unique_apis)
    raw["cluster_id"] = raw["API_Smiles"].map(smiles_to_cluster)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(raw, y=raw["Outcome1"], groups=raw["cluster_id"]))
    train_fold = raw.iloc[train_idx].reset_index(drop=True)
    val_fold = raw.iloc[val_idx].reset_index(drop=True)

    test_fold = pd.read_csv(os.path.join(HERE, "held_out_test_set.csv"))
    if "ground_truth" in test_fold.columns:
        test_fold = test_fold.rename(columns={"ground_truth": "Outcome1"})

    out = {"train": train_fold, "val": val_fold, "test": test_fold}
    for name, d in out.items():
        d.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)

    # --- log ---
    lines += [f"Input start_dataset.csv: {len(raw)} pairs, {raw['API_CID'].nunique()} APIs, {raw['Excipient_CID'].nunique()} excipients",
              f"Butina (Morgan r=2, 2048 bits, distance cutoff 0.15): {len(unique_apis)} unique API SMILES -> "
              f"{len(set(smiles_to_cluster.values()))} clusters",
              f"GroupShuffleSplit(n_splits=1, test_size=0.2, random_state={SEED}) on cluster_id", ""]
    lines += [_stats(n, d) for n, d in out.items()]
    lines += ["",
              f"API cluster overlap train<->val: {len(set(train_fold.cluster_id) & set(val_fold.cluster_id))}",
              f"API_CID overlap train<->val:     {len(set(train_fold.API_CID) & set(val_fold.API_CID))}",
              f"Excipient_CID overlap train<->val (expected): {len(set(train_fold.Excipient_CID) & set(val_fold.Excipient_CID))} "
              f"of {val_fold.Excipient_CID.nunique()} val excipients", ""]

    ho_pairs = set(zip(test_fold.API_CID, test_fold.Excipient_CID))
    for name, d in [("train", train_fold), ("val", val_fold)]:
        hits = d[[(a, e) in ho_pairs for a, e in zip(d.API_CID, d.Excipient_CID)]]
        for _, r in hits.iterrows():
            lines.append(f"Held-out pair also in {name}: API_CID {r.API_CID} + Excipient_CID {r.Excipient_CID} (label {r.Outcome1})")
    lines.append("")

    for name in out:
        ref = os.path.join(HERE, f"{name}.csv")
        new = os.path.join(args.out_dir, f"{name}.csv")
        if os.path.exists(ref) and os.path.abspath(ref) != os.path.abspath(new):
            same_bytes = _md5(ref) == _md5(new)
            same_rows = pd.read_csv(ref).equals(pd.read_csv(new))
            lines.append(f"{name}.csv vs original: identical rows/order = {same_rows}, identical bytes = {same_bytes} (md5 {_md5(ref)})")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out_dir, "split_log.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
