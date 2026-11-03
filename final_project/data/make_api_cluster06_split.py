"""
Build an API group split with looser API clusters: Tanimoto similarity >= 0.6 instead of >= 0.85.

    data/api_cluster06/    train.csv, val.csv, held_out_test_set.csv (the same 24 external pairs), split_log.txt

Same recipe as the API group split in data/ (python run.py split), with one change:
  - Butina on API SMILES, Morgan r=2, 2048 bits, Tanimoto DISTANCE cutoff 0.4 (similarity >= 0.6) -> cluster_id
    (0.85 groups only near-duplicates such as salts and stereoisomers; 0.6 also groups close analogues)
  - GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) on cluster_id -> train / val

cluster_id holds the 0.6 clusters, so the leave-own-cluster-out excipient priors of the train rows use them
too. exc_cluster_id / exc_group_id are added as in data/make_split_variants.py (for the split log only).

Train and evaluate the full model on it:
    python data/make_api_cluster06_split.py
    python run.py train --data-dir data/api_cluster06
"""

import os
from datetime import datetime

import pandas as pd
import sklearn
from rdkit import rdBase

from make_split_variants import (DATA, EXCIPIENT_SYNONYMS, HELDOUT_CSV, START_CSV, butina_cluster,
                                 excipient_groups, md5, val_groups, write_split)

API_SIMILARITY = 0.6                 # Butina distance cutoff = 1 - similarity
FOLDER = "api_cluster06"


def main():
    raw = pd.read_csv(START_CSV)
    heldout = pd.read_csv(HELDOUT_CSV)

    api_smiles = raw["API_Smiles"].unique().tolist()
    exc_smiles = raw["Excipient_Smiles"].unique().tolist()
    raw["cluster_id"] = raw["API_Smiles"].map(butina_cluster(api_smiles, dist_thresh=1 - API_SIMILARITY))
    exc_butina = butina_cluster(exc_smiles)
    cid_smiles = raw.drop_duplicates("Excipient_CID").set_index("Excipient_CID")["Excipient_Smiles"]
    exc_group, _ = excipient_groups(exc_smiles, exc_butina,
                                    [(cid_smiles[a], cid_smiles[b]) for a, b, _ in EXCIPIENT_SYNONYMS])
    raw["exc_cluster_id"] = raw["Excipient_Smiles"].map(exc_butina)
    raw["exc_group_id"] = raw["Excipient_Smiles"].map(exc_group)

    old = butina_cluster(api_smiles)                                   # the 0.85 clusters of data/
    sizes = raw.drop_duplicates("API_Smiles")["cluster_id"].value_counts()
    header = [
        f"Run {datetime.now().isoformat(timespec='seconds')} by data/make_api_cluster06_split.py",
        f"rdkit {rdBase.rdkitVersion} | scikit-learn {sklearn.__version__} | pandas {pd.__version__}",
        f"Input start_dataset.csv: {len(raw)} pairs, {raw['API_CID'].nunique()} APIs, "
        f"{raw['Excipient_CID'].nunique()} excipients (md5 {md5(START_CSV)})",
        f"Butina on API SMILES (Morgan r=2, 2048 bits, distance cutoff {1 - API_SIMILARITY:.2f}, "
        f"similarity >= {API_SIMILARITY}): {len(api_smiles)} API SMILES -> {sizes.size} clusters "
        f"({int((sizes == 1).sum())} single-API, largest {int(sizes.max())}); "
        f"at similarity >= 0.85 (data/): {len(set(old.values()))} clusters",
    ]

    val_api = val_groups(raw, "cluster_id")
    in_val = raw["cluster_id"].isin(val_api)
    tr, va = raw[~in_val], raw[in_val]
    old_val = set(pd.read_csv(os.path.join(DATA, "val.csv"))["API_CID"])
    write_split(FOLDER, f"API GROUP SPLIT AT TANIMOTO >= {API_SIMILARITY} (80/20, val APIs and their close analogues "
                        f"unseen in train)", [
        f"GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) on cluster_id "
        f"({len(val_api)} of {sizes.size} API clusters in val)",
        "Not stratified on the label (same as data/).",
        f"val APIs also in data/val.csv: {va['API_CID'].isin(old_val).sum()} of {len(va)} val rows",
    ], tr, va, heldout, header)


if __name__ == "__main__":
    main()
