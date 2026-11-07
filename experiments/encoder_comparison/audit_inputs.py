"""Audit every molecule through every representation, looking for silent failures.

Silent failure modes this catches:
  * RDKit cannot parse a SMILES -> the feature functions return zeros, no exception
  * a fingerprint that comes out all-zero (nothing set) for a molecule that did parse
  * NaN / inf anywhere in descriptors, salt context, flags or encoder output
  * an encoder returning an all-zero embedding for a molecule
  * tokenizer truncation for the sequence encoders (molecule silently cut short)
  * descriptor columns with zero variance on train (normalisation divides by ~0)
  * excipients that fall through the prior lookup to the global rate

Read-only: trains nothing, writes nothing.

    python -m experiments.encoder_comparison.audit_inputs
    python -m experiments.encoder_comparison.audit_inputs --device cpu
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.encoder_comparison.data import (SplitPaths, build_vector_lookup,  # noqa: E402
                                                 check_heldout_is_external)
from experiments.encoder_comparison.run_comparison import RESULTS_ROOT, build_encoder, make_config  # noqa: E402
from src.features import chemistry_flags, maccs_bits, morgan_bits, pgb_descriptors, salt_context  # noqa: E402
from src.lookup import get_mechanism_matches  # noqa: E402
from src.pgb_prior import ExcipientPriorTable  # noqa: E402
from src.utils import get_device  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROBLEMS = []


def report(ok: bool, label: str, detail: str = ""):
    print(f"  {'ok  ' if ok else 'WARN'} {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        PROBLEMS.append(f"{label}: {detail}")


def molecules(paths: SplitPaths):
    """Every unique (smiles, cid, where) in train / val / held-out."""
    seen, rows = set(), []
    for which in ("train", "val", "heldout"):
        df = paths.frame(which)
        for side in ("API", "Excipient"):
            for smi, cid in zip(df[f"{side}_Smiles"], df[f"{side}_CID"]):
                if (str(smi), str(cid)) not in seen:
                    seen.add((str(smi), str(cid)))
                    rows.append({"smiles": str(smi), "cid": str(cid), "where": which, "side": side})
    return pd.DataFrame(rows)


def audit_shared_features(mols: pd.DataFrame, paths: SplitPaths):
    print("\n[1] SMILES parsing and the shared per-molecule features")
    unparsed = [s for s in mols.smiles if Chem.MolFromSmiles(s) is None]
    report(not unparsed, f"RDKit parses all {len(mols)} unique molecules",
           f"{len(unparsed)} failed, e.g. {unparsed[:3]}" if unparsed else "")

    zero_morgan, zero_maccs, bad_desc, bad_salt, zero_flags = [], [], [], [], []
    for smi in mols.smiles:
        if not morgan_bits(smi, n_bits=1024).any():
            zero_morgan.append(smi)
        if not maccs_bits(smi).any():
            zero_maccs.append(smi)
        d = pgb_descriptors(smi)
        if not np.isfinite(d).all():
            bad_desc.append(smi)
        if not np.isfinite(salt_context(smi)).all():
            bad_salt.append(smi)
        if not chemistry_flags(smi).any():
            zero_flags.append(smi)
    report(not zero_morgan, "Morgan 1024: no all-zero molecule",
           f"{len(zero_morgan)} all-zero, e.g. {zero_morgan[:3]}" if zero_morgan else "")
    report(not zero_maccs, "MACCS 167: no all-zero molecule",
           f"{len(zero_maccs)} all-zero, e.g. {zero_maccs[:3]}" if zero_maccs else "")
    report(not bad_desc, "12 PGB descriptors: all finite",
           f"{len(bad_desc)} with NaN/inf, e.g. {bad_desc[:3]}" if bad_desc else "")
    report(not bad_salt, "8 salt-context values: all finite",
           f"{len(bad_salt)} with NaN/inf, e.g. {bad_salt[:3]}" if bad_salt else "")
    # an all-zero chemistry-flag vector is legitimate (a plain organic molecule), report as info
    print(f"  info {len(zero_flags)} molecule(s) match none of the 18 chemistry flags (legitimate)")

    print("\n[2] Descriptor normalisation (train-only statistics)")
    train = paths.frame("train")
    train_smiles = sorted(set(train.API_Smiles) | set(train.Excipient_Smiles))
    mat = np.stack([pgb_descriptors(s) for s in train_smiles])
    std = np.nanstd(mat, axis=0)
    flat = [i for i, v in enumerate(std) if v < 1e-6]
    report(not flat, "no descriptor has ~zero variance on train",
           f"columns {flat} would be divided by ~0" if flat else "")

    print("\n[3] Mechanism flags (pair level)")
    fires = 0
    for which in ("train", "val", "heldout"):
        df = paths.frame(which)
        n = sum(any(get_mechanism_matches(a, e)) for a, e in zip(df.API_Smiles, df.Excipient_Smiles))
        fires += n
        print(f"  info {which}: at least one of the 5 flags fires on {n}/{len(df)} pairs")
    report(fires > 0, "mechanism flags fire somewhere", "never fire — the head would see a constant")


def audit_prior(paths: SplitPaths):
    print("\n[4] Excipient prior lookup")
    train = paths.frame("train")
    table = ExcipientPriorTable(global_rate=float(train["Outcome1"].mean()))
    table.fit(train, leave_out_cluster_id=None)
    print(f"  info global rate (train positive rate) = {table.global_rate:.4f}")
    for which in ("val", "heldout"):
        df = paths.frame(which)
        res = [table.lookup(s) for s in df.Excipient_Smiles]
        unseen = sum(r["unseen"] == 1.0 for r in res)
        report(unseen == 0 or which == "heldout",
               f"{which}: excipients found in the prior table",
               f"{unseen}/{len(df)} fall through to the global rate" if unseen else "")
        if unseen:
            print("       (a fallback is by design for a genuinely new excipient, not an error)")


def audit_lookup_families(paths: SplitPaths, mols: pd.DataFrame, device):
    print("\n[5] Lookup families: PubChem and Mol2vec vectors")
    for family in ("pubchemfp", "mol2vec"):
        config = make_config(family, paths, 42, str(device), 40)
        lookup = build_vector_lookup(family, paths, os.path.join(RESULTS_ROOT, "_lookup"))
        enc = build_encoder(config, family, lookup, device)
        zeros, missing = [], []
        for s in mols.smiles:
            v = enc.encode([s]).cpu().numpy()[0]
            if s not in enc._vectors:
                missing.append(s)
            elif not v.any():
                zeros.append(s)
        report(not missing, f"{family}: every molecule has a vector",
               f"{len(missing)} missing, e.g. {missing[:2]}" if missing else "")
        report(not zeros, f"{family}: no all-zero vector",
               f"{len(zeros)} all-zero, e.g. {zeros[:2]}" if zeros else "")


def audit_deep_encoders(paths: SplitPaths, mols: pd.DataFrame, device):
    print("\n[6] Sequence / graph encoders: zeros, NaN, truncation, freezing")
    smiles = list(mols.smiles)
    longest = max(smiles, key=len)
    for family in ("molformer", "chemberta", "gin", "gat", "dmpnn"):
        config = make_config(family, paths, 42, str(device), 40)
        enc = build_encoder(config, family, None, device)

        zero_rows, nan_rows = 0, 0
        for i in range(0, len(smiles), 64):
            chunk = smiles[i:i + 64]
            tok, pooled, mask = enc.encode(chunk)
            pooled = pooled.detach().cpu()
            zero_rows += int((pooled.abs().sum(dim=-1) == 0).sum())
            nan_rows += int((~torch.isfinite(pooled)).any(dim=-1).sum())
        report(zero_rows == 0, f"{family}: no all-zero embedding", f"{zero_rows} molecules" if zero_rows else "")
        report(nan_rows == 0, f"{family}: no NaN/inf embedding", f"{nan_rows} molecules" if nan_rows else "")

        # truncation: a transformer cuts the SMILES at max_length and says nothing about it
        if hasattr(enc, "tokenizer"):
            limit = 512   # molformer_encoder.py / chemberta_encoder.py pass max_length=512
            counts = [len(enc.tokenizer(x)["input_ids"]) for x in smiles]
            cut = [(x, n) for x, n in zip(smiles, counts) if n > limit]
            report(not cut, f"{family}: no molecule truncated at {limit} tokens",
                   f"{len(cut)} truncated, longest {max(n for _, n in cut)} tokens" if cut else "")
            for x, n in sorted(cut, key=lambda t: -t[1])[:3]:
                where = sorted(set(mols[mols.smiles == x]["where"]))
                print(f"       {n} tokens -> cut to {limit}, appears in {where}: {x[:55]}...")
        else:
            tok, _, mask = enc.encode([longest])
            n_tok = int((~mask[0]).sum()) if mask is not None else tok.shape[1]
            heavy = Chem.MolFromSmiles(longest).GetNumHeavyAtoms()
            report(n_tok >= heavy, f"{family}: graph keeps every atom of the largest molecule",
                   f"only {n_tok} nodes for {heavy} heavy atoms" if n_tok < heavy else "")

        n_param = sum(p.numel() for p in enc.parameters())
        n_grad = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        if n_param == 0:
            report(True, f"{family}: encoder exposes no nn.Parameters",
                   "weights held outside the module; frozen by @torch.no_grad, nothing to train")
        else:
            report(n_grad == 0, f"{family}: encoder frozen", f"{n_grad:,} params still trainable" if n_grad else "")


def main():
    p = argparse.ArgumentParser(description="Audit every representation for silent failures.")
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="api_cluster")
    p.add_argument("--device", default=None)
    p.add_argument("--skip-deep", action="store_true", help="skip the 5 pretrained encoders")
    args = p.parse_args()

    paths = SplitPaths(args.data_root, args.split)
    check_heldout_is_external(paths)
    device = get_device(args.device or "auto")
    mols = molecules(paths)
    print(f"data root {paths.root} | device {device}")
    print(f"{len(mols)} unique molecules across train / val / held-out")

    audit_shared_features(mols, paths)
    audit_prior(paths)
    audit_lookup_families(paths, mols, device)
    if not args.skip_deep:
        audit_deep_encoders(paths, mols, device)

    print("\n" + "=" * 90)
    if PROBLEMS:
        print(f"{len(PROBLEMS)} thing(s) to look at:")
        for x in PROBLEMS:
            print(f"  - {x}")
    else:
        print("No silent zeros, NaNs or unfrozen weights found.")
    print("=" * 90)


if __name__ == "__main__":
    main()
