"""Reagent list and the reagent-free copy of the 80/20 grid split.

Some "excipients" in start_dataset.csv are laboratory reagents (oxidisers, strong acids and bases,
formaldehyde, bare metals, free ions) that no real formulation contains. Most are labelled
incompatible with every API they meet, so they teach "this excipient -> incompatible" rather than an
API-excipient interaction. This file lists them by PubChem CID and builds data/grid_8020_no_reagents/:
data/grid_8020/ with every reagent row removed from train AND val. Nothing else changes: the same
API-cluster assignment, the same 22 held-out pairs (none of them contains a reagent).

    python experiments/reagent_ablation/reagents.py            # build (skips if already built)
    python experiments/reagent_ablation/reagents.py --force    # rebuild

To change what counts as a reagent, edit REAGENTS below and rebuild with --force.
"""

import argparse
import json
import os
import shutil
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

SRC = os.path.join(PROJECT_ROOT, "data", "grid_8020")
DST = os.path.join(PROJECT_ROOT, "data", "grid_8020_no_reagents")

# Excipient_CID -> name. Conservative: only chemicals that are never pharmaceutical excipients.
REAGENTS = {
    516875: "potassium permanganate",
    24823: "ozone",
    1118: "sulfuric acid",
    14798: "sodium hydroxide",
    14797: "potassium hydroxide",
    784: "hydrogen peroxide",
    712: "formaldehyde",
    24434: "potassium nitrate",
    5359268: "aluminium (metal)",
    23994: "zinc (metal)",
    751: "glyceraldehyde",
    1061: "phosphate ion",
    1117: "sulfate ion",
    19660: "carbonate ion",
}


def is_reagent(df):
    return df["Excipient_CID"].isin(REAGENTS)


def build(force=False):
    # split_log.txt is written last, so a build that crashed half-way is redone
    if os.path.exists(os.path.join(DST, "split_log.txt")) and not force:
        print(f"{os.path.relpath(DST, PROJECT_ROOT)} exists (use --force to rebuild)")
        return
    if not os.path.exists(os.path.join(SRC, "train.csv")):
        raise SystemExit("data/grid_8020/ is missing. Run: python experiments/full_grid/prepare_data.py")

    os.makedirs(DST, exist_ok=True)
    log, removed = [], []
    for split in ("train", "val"):
        df = pd.read_csv(os.path.join(SRC, f"{split}.csv"))
        mask = is_reagent(df)
        kept = df[~mask]
        kept.to_csv(os.path.join(DST, f"{split}.csv"), index=False)
        removed.append(df[mask].assign(split=split, reagent=df.loc[mask, "Excipient_CID"].map(REAGENTS)))
        log.append(f"{split:5s} {len(df):5d} -> {len(kept):5d} rows | incompatible {int(df.Outcome1.sum()):4d} -> "
                   f"{int(kept.Outcome1.sum()):4d} | removed {int(mask.sum())} rows "
                   f"({int(df[mask].Outcome1.sum())} incompatible = "
                   f"{df[mask].Outcome1.sum() / df.Outcome1.sum():.1%} of this split's incompatible)")

    test = pd.read_csv(os.path.join(SRC, "test.csv"))
    if is_reagent(test).any():
        raise SystemExit("a held-out pair contains a reagent; the held-out set must stay identical")
    shutil.copy2(os.path.join(SRC, "test.csv"), os.path.join(DST, "test.csv"))
    log.append(f"test  {len(test):5d} held-out pairs, copied unchanged (no reagent in it)")

    # descriptor tables cover every molecule; normalisation stats must come from the new train only
    for f in ("api_descriptors.csv", "excipient_descriptors.csv"):
        shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))
    from src.descriptors import compute_norm_stats
    stats = compute_norm_stats(pd.read_csv(os.path.join(DST, "train.csv")),
                               pd.read_csv(os.path.join(DST, "api_descriptors.csv")),
                               pd.read_csv(os.path.join(DST, "excipient_descriptors.csv")))
    with open(os.path.join(DST, "descriptor_norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    removed = pd.concat(removed, ignore_index=True)
    removed.to_csv(os.path.join(DST, "removed_reagent_rows.csv"), index=False)
    per = removed.groupby("reagent").agg(rows=("Outcome1", "size"), incompatible=("Outcome1", "sum"))
    log += ["", "removed per reagent:", per.to_string(), "",
            "reagent CIDs: " + ", ".join(f"{c} {n}" for c, n in REAGENTS.items())]
    text = "\n".join(log)
    with open(os.path.join(DST, "split_log.txt"), "w") as f:
        f.write(f"data/grid_8020 with reagent rows removed (experiments/reagent_ablation/reagents.py)\n\n{text}\n")
    print(text)
    print(f"\nwritten to {os.path.relpath(DST, PROJECT_ROOT)}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build data/grid_8020_no_reagents/.")
    ap.add_argument("--force", action="store_true")
    build(ap.parse_args().force)
