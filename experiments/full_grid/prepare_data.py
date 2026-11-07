"""Build the data folder the full grid trains on: the ARES 80/20 split.

Writes data/grid_8020/ with everything every model in the grid needs:

    train.csv                  2770 rows, the ARES api_cluster train split
    val.csv                     774 rows
    test.csv                     22 external held-out pairs, label column renamed to Outcome1
    api_descriptors.csv         21 RDKit descriptors per API    (all molecules)
    excipient_descriptors.csv   21 RDKit descriptors per excipient
    descriptor_norm_stats.json  mean / std computed from the TRAIN split only

The baseline (non-PGB) models read descriptors through CompatibilityDataset, which needs these
files; the PGB models compute their own features from SMILES and ignore them. The normalisation
stats come from train only, so no validation or held-out information reaches the model.

    python experiments/full_grid/prepare_data.py
"""

import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.descriptors import compute_descriptor_table, compute_norm_stats  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "data", "grid_8020")
# the canonical split and held-out set live in the ARES project
ARES = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "..", "ARES", "final_project", "data"))


def main():
    src = {
        "train": os.path.join(ARES, "splits", "api_cluster", "train.csv"),
        "val": os.path.join(ARES, "splits", "api_cluster", "val.csv"),
        "heldout": os.path.join(ARES, "heldout", "held_out_test_set.csv"),
        "raw": os.path.join(ARES, "raw", "start_dataset.csv"),
    }
    missing = [p for p in src.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit("not found:\n  " + "\n  ".join(missing)
                         + "\nPass the ARES data folder by editing ARES at the top of this file.")

    os.makedirs(OUT, exist_ok=True)
    train = pd.read_csv(src["train"])
    val = pd.read_csv(src["val"])
    # the held-out file calls the label ground_truth; every dataset in this repo expects Outcome1
    test = pd.read_csv(src["heldout"]).rename(columns={"ground_truth": "Outcome1"})
    raw = pd.read_csv(src["raw"])

    pairs = set(zip(test["API_CID"], test["Excipient_CID"]))
    for name, df in (("train", train), ("val", val)):
        overlap = pairs & set(zip(df["API_CID"], df["Excipient_CID"]))
        if overlap:
            raise SystemExit(f"{len(overlap)} held-out pair(s) also in {name}: {sorted(overlap)}")

    train.to_csv(os.path.join(OUT, "train.csv"), index=False)
    val.to_csv(os.path.join(OUT, "val.csv"), index=False)
    test.to_csv(os.path.join(OUT, "test.csv"), index=False)

    # descriptors over every molecule the grid can meet (raw + the held-out pairs)
    everything = pd.concat([raw, test], ignore_index=True)
    print("computing API descriptors ...")
    api = compute_descriptor_table(everything, "API_CID", "API_Smiles")
    api.to_csv(os.path.join(OUT, "api_descriptors.csv"), index=False)
    print("computing excipient descriptors ...")
    exc = compute_descriptor_table(everything, "Excipient_CID", "Excipient_Smiles")
    exc.to_csv(os.path.join(OUT, "excipient_descriptors.csv"), index=False)

    print("computing normalisation stats from the train split only ...")
    stats = compute_norm_stats(train, api, exc)
    with open(os.path.join(OUT, "descriptor_norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    need_api = set(train.API_CID) | set(val.API_CID) | set(test.API_CID)
    need_exc = set(train.Excipient_CID) | set(val.Excipient_CID) | set(test.Excipient_CID)
    miss_a = need_api - set(api.iloc[:, 0]); miss_e = need_exc - set(exc.iloc[:, 0])
    print(f"\n{OUT}")
    print(f"  train {len(train)} | val {len(val)} | test (held-out) {len(test)}")
    print(f"  API descriptors {len(api)} rows, missing for {len(miss_a)} needed CIDs")
    print(f"  excipient descriptors {len(exc)} rows, missing for {len(miss_e)} needed CIDs")
    if miss_a or miss_e:
        raise SystemExit("some molecules have no descriptors; the baseline models would silently see zeros")
    print("  no held-out pair occurs in train or val")


if __name__ == "__main__":
    main()
