"""
Re-split the Oct-6 80/20 PGB data into train / val / test (about 60 / 20 / 20) without touching val.

Source (the split the Oct-6 80/20 PGB models were trained on, made by src/single_run.py at
commit f534ef7 with GroupShuffleSplit on Butina API clusters):
    metrics/pgb_fixed_vector_morgan_concat_asl/single_run_80_20split/{train,val}.csv

    val   = the Oct-6 val set, unchanged (774 rows)
    train + test = the Oct-6 train set (2770 rows), split with StratifiedGroupKFold(4) on the same
                   cluster_id groups, stratified on Outcome1; the first fold becomes test.

No API cluster crosses train / val / test. The held-out 24 pairs stay the external test set.

Output: data/oct6_resplit/{train,val,test}.csv, splitlog.log, heldout_overlap.csv
Usage (from the project root):
    python -m experiments.oct6_resplit.make_split
    python -m experiments.oct6_resplit.make_split --drop_heldout_overlap   # also remove rows matching a held-out pair
"""

import argparse
import hashlib
import logging
import os
import sys
import tempfile

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SRC_DIR = "metrics/pgb_fixed_vector_morgan_concat_asl/single_run_80_20split"
OUT_DIR = "data/oct6_resplit"
HELDOUT_CSV = "held_out_testset/held_out_test_set.csv"
RANDOM_STATE = 42

log = logging.getLogger("oct6_resplit")


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def heldout_matches(df: pd.DataFrame, heldout: pd.DataFrame) -> pd.Series:
    """Rows whose (API, excipient) pair matches a held-out pair by CID or by salt/stereo-blind parent."""
    sys.path.insert(0, "data")
    from data_split import _parent_key
    cids = set(zip(heldout["API_CID"], heldout["Excipient_CID"]))
    parents = set(zip(heldout["API_Smiles"].map(_parent_key), heldout["Excipient_Smiles"].map(_parent_key)))
    keys = zip(df["API_CID"], df["Excipient_CID"], df["API_Smiles"].map(_parent_key), df["Excipient_Smiles"].map(_parent_key))
    return pd.Series([((a, e) in cids) or ((pa, pe) in parents) for a, e, pa, pe in keys], index=df.index)


def main():
    p = argparse.ArgumentParser(description="Split the Oct-6 80/20 train set into train/test, keeping the Oct-6 val set.")
    p.add_argument("--drop_heldout_overlap", action="store_true",
                   help="remove rows matching a held-out pair from train/val/test (default: keep the Oct-6 data unchanged)")
    p.add_argument("--force", action="store_true", help="overwrite existing files in data/oct6_resplit that differ")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(OUT_DIR, "splitlog.log"), mode="w")])

    old_train = pd.read_csv(f"{SRC_DIR}/train.csv")
    val = pd.read_csv(f"{SRC_DIR}/val.csv")
    log.info(f"Source: {SRC_DIR}  train {len(old_train)} (md5 {_md5(f'{SRC_DIR}/train.csv')})  val {len(val)} (md5 {_md5(f'{SRC_DIR}/val.csv')})")

    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(old_train, y=old_train["Outcome1"], groups=old_train["cluster_id"]))
    splits = {"train": old_train.iloc[train_idx].reset_index(drop=True),
              "val": val,
              "test": old_train.iloc[test_idx].reset_index(drop=True)}

    # Leakage check: no API cluster or API_CID in two splits
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            g = set(splits[a]["cluster_id"]) & set(splits[b]["cluster_id"])
            c = set(splits[a]["API_CID"]) & set(splits[b]["API_CID"])
            log.info(f"overlap {a}<->{b}: clusters {len(g)}, API_CID {len(c)}")
            if g or c:
                raise RuntimeError(f"LEAKAGE between {a} and {b}; split not written")

    # Rows that match a held-out pair
    heldout = pd.read_csv(HELDOUT_CSV)
    hits = []
    for name, d in splits.items():
        m = heldout_matches(d, heldout)
        for _, r in d[m].iterrows():
            hits.append({"split": name, "API_CID": r["API_CID"], "Excipient_CID": r["Excipient_CID"], "Outcome1": r["Outcome1"]})
        if args.drop_heldout_overlap:
            splits[name] = d[~m].reset_index(drop=True)
    for h in hits:
        log.info(f"held-out overlap in {h['split']}: API_CID {h['API_CID']} + Excipient_CID {h['Excipient_CID']} label {h['Outcome1']}"
                 + ("  -> removed" if args.drop_heldout_overlap else "  -> kept"))
    pd.DataFrame(hits).to_csv(os.path.join(OUT_DIR, "heldout_overlap.csv"), index=False)

    total = sum(len(d) for d in splits.values())
    for name, d in splits.items():
        log.info(f"{name.upper():5s} rows={len(d):5d} ({len(d) / total:5.1%}) | incompatible={int(d.Outcome1.sum()):4d} ({d.Outcome1.mean():5.1%}) | "
                 f"APIs={d.API_CID.nunique():4d} | excipients={d.Excipient_CID.nunique():4d}")

    # Write; never silently replace an existing different split
    with tempfile.TemporaryDirectory() as tmp:
        pending = []
        for name, d in splits.items():
            t = os.path.join(tmp, f"{name}.csv")
            d.to_csv(t, index=False)
            dst = os.path.join(OUT_DIR, f"{name}.csv")
            if os.path.exists(dst) and _md5(dst) == _md5(t):
                log.info(f"{dst}: unchanged")
            else:
                pending.append((t, dst))
        clash = [dst for _, dst in pending if os.path.exists(dst)]
        if clash and not args.force:
            raise SystemExit(f"Refusing to overwrite {clash}; models trained on them would no longer match. Use --force.")
        for t, dst in pending:
            with open(t, "rb") as s, open(dst, "wb") as o:
                o.write(s.read())
            log.info(f"{dst}: written")


if __name__ == "__main__":
    main()
