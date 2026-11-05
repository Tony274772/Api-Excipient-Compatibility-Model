"""
Build the three extra train / val splits next to the default API group split.

    data/splits/api_cluster/          API group split (existing, unchanged): val APIs unseen in train
    data/splits/random/               random 80/20, stratified on Outcome1: APIs and excipients shared
    data/splits/excipient_new/        excipient group split, stratified: val excipients unseen in train
    data/splits/api_excipient_new/    API + excipient group split: val API AND excipient unseen in train

Each folder holds train.csv, val.csv and split_log.txt. All splits are tested on the same 22 held-out
pairs (data/heldout/held_out_test_set.csv). Every train / val file keeps the API `cluster_id` column (used for the leave-own-cluster-out
excipient priors) and adds `exc_cluster_id` and `exc_group_id`.

How the groups are made:
  - API groups (same recipe as `python run.py split`): Butina on API SMILES, Morgan r=2, 2048 bits,
    Tanimoto distance cutoff 0.15 -> cluster_id
  - excipient Butina clusters: the same recipe on excipient SMILES -> exc_cluster_id
    (puts near-identical excipients such as Mg / Zn / Ca stearate in one cluster)
  - excipient groups -> exc_group_id: Butina clusters merged (union-find) whenever two excipients are
    forms of the same parent compound, so a salt / hydrate / acid / lake / mixture and its parent never sit
    on opposite sides of the split (see excipient_groups)

Splits:
  - API: GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) on cluster_id (the existing split)
  - excipient: label-stratified group split on exc_group_id (see stratified_val_groups): best of 1000
    random group shuffles, closest to 20 % of rows and the overall incompatible rate (also scoring the
    incompatible rate of the API + excipient val); groups stay whole.
  - API + excipient: the val API clusters are the ones of the existing API split and the val excipient
    groups are the ones of excipient_new/. val = pairs whose API and excipient are both in val groups,
    train = pairs whose API and excipient are both in train groups. Pairs with one side in each are dropped
    (written to api_excipient_new/dropped_mixed.csv).

Run from any directory:
    python scripts/data/make_split_variants.py
"""

import hashlib
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # project root

from apiexc import paths  # noqa: E402
from apiexc.splitting import butina_cluster  # noqa: E402,F401  (also imported by make_api_cluster06_split.py)

RDLogger.DisableLog("rdApp.*")

START_CSV = str(paths.RAW_DATASET)
HELDOUT_CSV = str(paths.HELDOUT_CSV)
API_TRAIN_CSV = str(paths.split_dir(paths.DEFAULT_SPLIT) / "train.csv")
API_VAL_CSV = str(paths.split_dir(paths.DEFAULT_SPLIT) / "val.csv")

SEED = 42
TEST_SIZE = 0.2


SOLVENTS = {Chem.CanonSmiles(s) for s in ["O", "CO", "CCO", "CC(=O)O"]}   # water, methanol, ethanol, acetic acid
_uncharger = rdMolStandardize.Uncharger()


def _neutral(mol):
    mol = _uncharger.uncharge(Chem.Mol(mol))
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def _is_organic(mol):
    """Has a carbon bonded to H or to another carbon (so carbonates, cyanide and CO2 count as inorganic)."""
    return any(a.GetSymbol() == "C" and (a.GetTotalNumHs() > 0 or any(n.GetSymbol() == "C" for n in a.GetNeighbors()))
               for a in mol.GetAtoms())


def parent_keys(smiles):
    """(full, organic) parent keys of an excipient, both frozensets of neutralised, stereo-blind fragments
    with water / solvent fragments dropped (unless nothing else is left).
      full    : all remaining fragments, counter-ions included (CaCl2 == CaCl2 dihydrate)
      organic : only the organic fragments, counter-ions dropped (sodium saccharin == saccharin,
                Mg stearate == stearic acid); empty for inorganic excipients"""
    mols = [m for m in (Chem.MolFromSmiles(f) for f in smiles.strip().split(".")) if m is not None]
    keep = [m for m in mols if Chem.MolToSmiles(m) not in SOLVENTS] or mols
    return frozenset(_neutral(m) for m in keep), frozenset(_neutral(m) for m in keep if _is_organic(m))


# Same excipient under two PubChem CIDs whose structures no rule above links (polymers / pigments drawn
# differently). Found by reviewing every cross-group pair with the same formula or Tanimoto >= 0.7 against
# the PubChem names.
EXCIPIENT_SYNONYMS = [
    (5281955, 5284448, "polysorbate 80 (two sorbitan ring drawings, both C32H60O10)"),
    (24832095, 57503849, "hypromellose / hydroxypropylmethylcellulose (HPMC)"),
    (3037558, 6134, "lactose, anhydrous / beta-lactose"),
    (23320441, 61560, "ferric oxide yellow (C.I. 77492) / ferric oxide hydrate"),
]


def excipient_groups(unique_smiles, butina, synonym_pairs=()):
    """Merge excipient Butina clusters into parent-compound groups (union-find). Two excipients share a
    group if they are in the same Butina cluster, or have the same full parent key, or one's organic
    parent key is a subset of the other's (a salt / acid / lake form, or a mixture containing the compound),
    or they are listed in synonym_pairs (SMILES pairs).
    Returns {smiles: group id} and the list of merges that Butina alone did not make."""
    up = {s: s for s in unique_smiles}

    def find(x):
        while up[x] != x:
            up[x] = up[up[x]]
            x = up[x]
        return x

    by_cluster = {}
    for s in unique_smiles:
        by_cluster.setdefault(butina[s], []).append(s)
    for members in by_cluster.values():
        for m in members[1:]:
            up[find(m)] = find(members[0])

    keys = {s: parent_keys(s) for s in unique_smiles}
    merges = []
    for i, a in enumerate(unique_smiles):
        (fa, oa) = keys[a]
        for b in unique_smiles[i + 1:]:
            (fb, ob) = keys[b]
            same = fa == fb or (oa and ob and (oa <= ob or ob <= oa))
            if same and butina[a] != butina[b]:
                merges.append((a, b))
                ra, rb = find(a), find(b)
                if ra != rb:
                    up[ra] = rb
    for a, b in synonym_pairs:
        if butina[a] != butina[b]:
            merges.append((a, b))
            up[find(a)] = find(b)
    roots = sorted({find(s) for s in unique_smiles})
    gid = {r: i for i, r in enumerate(roots)}
    return {s: gid[find(s)] for s in unique_smiles}, merges


def val_groups(df, col):
    """Groups GroupShuffleSplit(test_size=0.2, random_state=42) puts in val."""
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    _, val_idx = next(gss.split(df, groups=df[col]))
    return set(df.iloc[val_idx][col])


N_TRIALS = 1000


def stratified_val_groups(df, col, api_val_mask):
    """Label-stratified group split. StratifiedGroupKFold cannot balance this data (its 5 folds range from
    6 % to 13 % incompatible) because a few excipient groups hold up to 11 % of the rows and largely decide
    the label. Instead: N_TRIALS random shuffles of the groups (rng seeded SEED); each fills val in shuffled
    order until it is closest to TEST_SIZE of the rows. Each trial is scored on
      - the excipient split: relative error of the val row share vs TEST_SIZE + of the val incompatible rate
        vs the overall rate, and
      - the API + excipient split (val = rows in these groups AND in api_val_mask): relative error of its val
        incompatible rate vs the overall rate,
    and the lowest total wins. Only row counts and labels are used, never model results.
    Returns (val groups, description)."""
    rng = np.random.default_rng(SEED)
    size = df.groupby(col).size()
    groups = size.index.to_numpy()
    y, g = df["Outcome1"].to_numpy(), df[col].to_numpy()
    n, rate = len(df), y.mean()
    best = None
    for trial in range(N_TRIALS):
        order = rng.permutation(groups)
        cum = size[order].cumsum().to_numpy()
        k = int(np.argmin(np.abs(cum - TEST_SIZE * n))) + 1          # number of groups closest to 20 % of rows
        in_val = np.isin(g, order[:k])
        both = in_val & api_val_mask
        frac, vrate, brate = in_val.mean(), y[in_val].mean(), y[both].mean() if both.any() else 0.0
        score = abs(frac - TEST_SIZE) / TEST_SIZE + abs(vrate - rate) / rate + abs(brate - rate) / rate
        if best is None or score < best[0]:
            best = (score, trial, set(order[:k]), frac, vrate, brate)
    score, trial, val, frac, vrate, brate = best
    return val, (f"best of {N_TRIALS} random group shuffles (numpy default_rng({SEED})): trial {trial}, "
                 f"val = {frac:.1%} of rows at {vrate:.1%} incompatible (overall {rate:.1%}); "
                 f"API + excipient val at {brate:.1%}; score {score:.4f}")


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def stats(name, d):
    s = (f"{name:7s} rows={len(d):5d} | incompatible={int(d['Outcome1'].sum()):4d} ({d['Outcome1'].mean():5.1%}) | "
         f"compatible={int((d['Outcome1'] == 0).sum()):5d} | unique APIs={d['API_CID'].nunique():4d} | "
         f"unique excipients={d['Excipient_CID'].nunique():4d}")
    if "exc_group_id" in d:
        s += (f" | API clusters={d['cluster_id'].nunique():4d} | excipient clusters={d['exc_cluster_id'].nunique():4d}"
              f" | excipient groups={d['exc_group_id'].nunique():4d}")
    return s


def overlap_lines(tr, va, heldout):
    """Entity overlaps train <-> val, and how much of the held-out set train has already seen."""
    lines = [
        f"API cluster overlap train<->val:       {len(set(tr.cluster_id) & set(va.cluster_id))} of {va.cluster_id.nunique()} val API clusters",
        f"API_CID overlap train<->val:           {len(set(tr.API_CID) & set(va.API_CID))} of {va.API_CID.nunique()} val APIs",
        f"excipient cluster overlap train<->val: {len(set(tr.exc_cluster_id) & set(va.exc_cluster_id))} of {va.exc_cluster_id.nunique()} val excipient clusters",
        f"excipient group overlap train<->val:   {len(set(tr.exc_group_id) & set(va.exc_group_id))} of {va.exc_group_id.nunique()} val excipient groups",
        f"Excipient_CID overlap train<->val:     {len(set(tr.Excipient_CID) & set(va.Excipient_CID))} of {va.Excipient_CID.nunique()} val excipients",
        f"pair overlap train<->val:              {len(set(zip(tr.API_CID, tr.Excipient_CID)) & set(zip(va.API_CID, va.Excipient_CID)))}",
        "",
        f"held-out APIs seen in train:           {heldout.API_CID.isin(tr.API_CID).sum()} of {len(heldout)} pairs",
        f"held-out excipients seen in train:     {heldout.Excipient_CID.isin(tr.Excipient_CID).sum()} of {len(heldout)} pairs",
    ]
    train_pairs = set(zip(tr.API_CID, tr.Excipient_CID))
    for _, r in heldout.iterrows():
        if (r.API_CID, r.Excipient_CID) in train_pairs:
            lines.append(f"held-out pair also in train:           API_CID {r.API_CID} + Excipient_CID {r.Excipient_CID}")
    return lines


def write_split(folder, title, method_lines, tr, va, heldout, header, extra=None):
    out = os.path.join(paths.SPLITS, folder)
    os.makedirs(out, exist_ok=True)
    tr = tr.reset_index(drop=True)
    va = va.reset_index(drop=True)
    tr.to_csv(os.path.join(out, "train.csv"), index=False)
    va.to_csv(os.path.join(out, "val.csv"), index=False)
    if extra is not None:
        for name, d in extra.items():
            d.reset_index(drop=True).to_csv(os.path.join(out, name), index=False)

    lines = [title, ""] + header + [""] + method_lines + ["", stats("train", tr), stats("val", va), stats("heldout", heldout.rename(columns={"ground_truth": "Outcome1"})), ""]
    lines += overlap_lines(tr, va, heldout) + [""]
    for name in ["train.csv", "val.csv"] + (list(extra) if extra else []):
        lines.append(f"{name}: md5 {md5(os.path.join(out, name))}")
    with open(os.path.join(out, "split_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines) + "\n" + "-" * 100)


def main():
    raw = pd.read_csv(START_CSV)
    heldout = pd.read_csv(HELDOUT_CSV)

    api_smiles = raw["API_Smiles"].unique().tolist()
    exc_smiles = raw["Excipient_Smiles"].unique().tolist()
    raw["cluster_id"] = raw["API_Smiles"].map(butina_cluster(api_smiles))
    exc_butina = butina_cluster(exc_smiles)
    cid_smiles = raw.drop_duplicates("Excipient_CID").set_index("Excipient_CID")["Excipient_Smiles"]
    exc_group, merges = excipient_groups(exc_smiles, exc_butina,
                                         [(cid_smiles[a], cid_smiles[b]) for a, b, _ in EXCIPIENT_SYNONYMS])
    raw["exc_cluster_id"] = raw["Excipient_Smiles"].map(exc_butina)
    raw["exc_group_id"] = raw["Excipient_Smiles"].map(exc_group)
    group_share = raw["exc_group_id"].value_counts(normalize=True)

    header = [
        f"Run {datetime.now().isoformat(timespec='seconds')} by scripts/data/make_split_variants.py",
        f"rdkit {rdBase.rdkitVersion} | scikit-learn {sklearn.__version__} | pandas {pd.__version__}",
        f"Input start_dataset.csv: {len(raw)} pairs, {raw['API_CID'].nunique()} APIs, "
        f"{raw['Excipient_CID'].nunique()} excipients (md5 {md5(START_CSV)})",
        f"Butina (Morgan r=2, 2048 bits, distance cutoff 0.15): {len(api_smiles)} API SMILES -> "
        f"{raw['cluster_id'].nunique()} clusters | {len(exc_smiles)} excipient SMILES -> "
        f"{raw['exc_cluster_id'].nunique()} clusters",
        f"Excipient parent-compound groups: {raw['exc_cluster_id'].nunique()} Butina clusters -> "
        f"{raw['exc_group_id'].nunique()} groups ({len(merges)} salt / hydrate / acid / mixture links); "
        f"largest groups hold {', '.join(f'{p:.1%}' for p in group_share.head(3))} of rows",
    ]
    merge_lines = ["Excipient pairs merged by the parent-compound rule or EXCIPIENT_SYNONYMS "
                   "(Butina put them in different clusters):"]
    merge_lines += [f"  {a}  <->  {b}" for a, b in merges]
    merge_lines += ["Synonym merges (by CID): " + "; ".join(f"{a} = {b} {why}" for a, b, why in EXCIPIENT_SYNONYMS)]

    # API partition: must reproduce the existing data/splits/api_cluster/train.csv / val.csv exactly
    val_api = val_groups(raw, "cluster_id")
    in_val_api = raw["cluster_id"].isin(val_api)
    cols = ["API_CID", "Excipient_CID", "Outcome1", "API_Smiles", "Excipient_Smiles", "cluster_id"]
    same = (raw[~in_val_api][cols].reset_index(drop=True).equals(pd.read_csv(API_TRAIN_CSV))
            and raw[in_val_api][cols].reset_index(drop=True).equals(pd.read_csv(API_VAL_CSV)))
    if not same:
        raise SystemExit("API partition does not reproduce data/splits/api_cluster/ -- check library versions")

    # 1. random, stratified on the label
    tr, va = train_test_split(raw, test_size=TEST_SIZE, random_state=SEED, stratify=raw["Outcome1"])
    write_split("random", "RANDOM SPLIT (80/20, stratified on Outcome1)", [
        f"train_test_split(test_size={TEST_SIZE}, random_state={SEED}, stratify=Outcome1) on rows",
        "APIs and excipients are shared between train and val by design.",
    ], tr, va, heldout, header)

    # 2. excipient group split, stratified on the label
    val_exc, how = stratified_val_groups(raw, "exc_group_id", in_val_api.to_numpy())
    in_val_exc = raw["exc_group_id"].isin(val_exc)
    write_split("excipient_new", "EXCIPIENT GROUP SPLIT, STRATIFIED (val excipients unseen in train)", [
        "Label-stratified group split on exc_group_id: " + how,
        "APIs are shared between train and val by design.",
        "",
    ] + merge_lines, raw[~in_val_exc], raw[in_val_exc], heldout, header)

    # 3. API + excipient group split: both sides unseen
    write_split("api_excipient_new", "API + EXCIPIENT GROUP SPLIT (val API and val excipient both unseen in train)", [
        f"val API clusters  = GroupShuffleSplit(test_size={TEST_SIZE}, random_state={SEED}) on cluster_id "
        f"(same {len(val_api)} clusters as api_cluster/val.csv)",
        f"val excipient groups = the label-stratified group split of excipient_new/ "
        f"(same {len(val_exc)} groups as excipient_new/val.csv)",
        "train = API in train clusters AND excipient in train groups",
        "val   = API in val clusters AND excipient in val groups",
        f"dropped (one side in each) = {int((in_val_api ^ in_val_exc).sum())} pairs -> dropped_mixed.csv.",
        "The API side is not stratified (it is the existing API split); the excipient side is.",
    ], raw[~in_val_api & ~in_val_exc], raw[in_val_api & in_val_exc], heldout, header,
        extra={"dropped_mixed.csv": raw[in_val_api ^ in_val_exc]})


if __name__ == "__main__":
    main()
