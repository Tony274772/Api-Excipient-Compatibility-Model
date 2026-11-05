"""How the default train / val split (data/splits/api_cluster/) is made, and a check that it reproduces.

1. Butina-cluster the unique API SMILES on Morgan fingerprints (radius 2, 2048 bits), Tanimoto distance
   cutoff 0.15 (similarity >= 0.85) -> cluster_id
2. GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) over the API clusters -> train / val
   (not stratified on the label)
The 22 held-out pairs (data/heldout/) are the external test set.
"""

import os
from datetime import datetime

import pandas as pd

from apiexc import paths
from apiexc.pipeline import get_split, heldout_overlap, md5


def butina_cluster(unique_smiles, dist_thresh=0.15):
    """Butina clustering on Morgan fingerprints (radius 2, 2048 bits) -> {smiles: cluster id}."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in unique_smiles]
    dists = []
    for i in range(1, len(fps)):
        dists.extend(1.0 - s for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    clusters = Butina.ClusterData(dists, len(fps), dist_thresh, isDistData=True)
    return {unique_smiles[m]: cid for cid, members in enumerate(clusters) for m in members}


def split_stats(name, d):
    return (f"{name:7s} rows={len(d):5d} | incompatible={int(d['Outcome1'].sum()):4d} ({d['Outcome1'].mean():5.1%}) | "
            f"compatible={int((d['Outcome1'] == 0).sum()):5d} | unique APIs={d['API_CID'].nunique():4d} | "
            f"unique excipients={d['Excipient_CID'].nunique():4d}")


def rebuild_split(out_dir=paths.REBUILT_SPLIT) -> bool:
    """Rebuild train / val from data/raw/start_dataset.csv into out_dir, write split_log.txt there, and
    check the result is identical to data/splits/api_cluster/. Returns True when it is."""
    import sklearn
    from rdkit import rdBase
    from sklearn.model_selection import GroupShuffleSplit

    split = get_split(paths.DEFAULT_SPLIT)
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log(f"Split rebuilt {datetime.now().isoformat(timespec='seconds')} by `python run.py split`")
    log(f"rdkit {rdBase.rdkitVersion} | scikit-learn {sklearn.__version__} | pandas {pd.__version__}")
    log()

    raw = pd.read_csv(paths.RAW_DATASET)
    unique_apis = raw["API_Smiles"].unique().tolist()
    smiles_to_cluster = butina_cluster(unique_apis)
    raw["cluster_id"] = raw["API_Smiles"].map(smiles_to_cluster)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(raw, y=raw["Outcome1"], groups=raw["cluster_id"]))
    splits = {"train": raw.iloc[train_idx].reset_index(drop=True), "val": raw.iloc[val_idx].reset_index(drop=True)}
    heldout = split.frame("heldout")

    log(f"Input {paths.rel(paths.RAW_DATASET)}: {len(raw)} pairs, {raw['API_CID'].nunique()} APIs, "
        f"{raw['Excipient_CID'].nunique()} excipients (md5 {md5(paths.RAW_DATASET)})")
    log(f"Butina (Morgan r=2, 2048 bits, distance cutoff 0.15): {len(unique_apis)} unique API SMILES -> "
        f"{len(set(smiles_to_cluster.values()))} clusters")
    log("GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) on cluster_id")
    log()
    log(split_stats("train", splits["train"]))
    log(split_stats("val", splits["val"]))
    log(split_stats("heldout", heldout))
    log()
    tr, va = splits["train"], splits["val"]
    log(f"API cluster overlap train<->val: {len(set(tr.cluster_id) & set(va.cluster_id))}")
    log(f"API_CID overlap train<->val:     {len(set(tr.API_CID) & set(va.API_CID))}")
    log(f"Excipient_CID overlap train<->val (expected): {len(set(tr.Excipient_CID) & set(va.Excipient_CID))} "
        f"of {va.Excipient_CID.nunique()} val excipients")
    overlap = heldout_overlap(split)
    log(f"Held-out pairs also in train: {len(overlap['train'])} | also in val: {len(overlap['val'])}")
    log()

    ok = True
    for name, ref in [("train", split.train_csv), ("val", split.val_csv)]:
        path = os.path.join(out_dir, f"{name}.csv")
        splits[name].to_csv(path, index=False)
        same_rows = pd.read_csv(path).equals(pd.read_csv(ref))
        ok &= same_rows
        log(f"{name}.csv: {len(splits[name])} rows | identical rows/order to {paths.rel(ref)} = {same_rows} | "
            f"identical bytes = {md5(path) == md5(ref)}")

    with open(os.path.join(out_dir, "split_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return ok
