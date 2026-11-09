"""Compute each frozen encoder's mean-pooled vector for every molecule, once.

Two blocks of the grid need a deep encoder reduced to ONE vector per molecule:

  * the embedding-secondary columns - a second encoder's averaged output in the 32-wide block,
  * the no-cross-attention control  - the encoder's averaged output in the 64-wide primary block.

Both are cheap only if the encoder runs once per molecule instead of once per epoch per run. There
are 505 runs and roughly 1,100 distinct molecules, so this script saves the encoder forward pass
about half a million times.

What is written, per encoder, to
experiments/results/encoder_grid/_embeddings/<name>_mean.csv:

    cid, smiles, f0 ... f<d-1>        one row per molecule in train + val + held-out

The vector is the encoder's OWN pooled output - the masked mean over its tokens or atoms, the
second element of encode(). We do not re-pool it ourselves, so the number here is exactly what the
encoder reports for that molecule.

RAW, not scaled. Z-scoring is per run and uses TRAIN molecules only (reps.Representation.fit_scaler),
so scaling must not be baked into a table that the held-out rows also read from.

The encoders are frozen and in eval mode, so this is deterministic: running it twice gives the same
numbers. It writes nothing outside _embeddings/ and trains nothing.

    python -m experiments.encoder_comparison.precompute_embeddings
    python -m experiments.encoder_comparison.precompute_embeddings --encoders chemberta gat
    python -m experiments.encoder_comparison.precompute_embeddings --device cpu --overwrite
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.encoder_comparison.data import (EMBEDDING_DIR, SplitPaths,  # noqa: E402
                                                 check_heldout_is_external, embedding_table_path)
from experiments.encoder_comparison.families import FAMILIES, DEEP  # noqa: E402
from experiments.encoder_comparison.reps import REP_SPECS  # noqa: E402
from src.utils import get_device  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def molecule_table(paths: SplitPaths) -> pd.DataFrame:
    """Every distinct molecule in train + val + held-out, with a CID for traceability."""
    rows = {}
    for which in ("train", "val", "heldout"):
        df = paths.frame(which)
        pairs = (list(zip(df["API_CID"], df["API_Smiles"]))
                 + list(zip(df["Excipient_CID"], df["Excipient_Smiles"])))
        for cid, smi in pairs:
            if pd.notna(smi) and str(smi):
                rows.setdefault(str(smi), str(cid))
    return pd.DataFrame({"cid": list(rows.values()), "smiles": list(rows.keys())}) \
             .sort_values("smiles").reset_index(drop=True)


def build_encoder(name: str, device):
    """The same frozen encoder the `none`-column runs used, in eval mode with gradients off."""
    import inspect

    from src.config import Config
    from src.encoders import ENCODER_REGISTRY

    spec = FAMILIES[name]
    config = Config()
    config.encoder = spec["encoder"]
    config.device = str(device)
    config.resolve_paths()

    cls = ENCODER_REGISTRY[config.encoder]
    params = list(inspect.signature(cls.__init__).parameters)[1:]
    kwargs = {"device": str(device)}
    if "config" in params:
        kwargs["config"] = config
    if "model_path" in params:
        kwargs["model_path"] = {"molformer": config.molformer_model_path,
                                "chemberta": config.chemberta_model_path}[config.encoder]
    if "pretrained_name" in params:
        kwargs["pretrained_name"] = config.gin_pretrained_name
    enc = cls(**kwargs)
    enc.to(device)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    if not getattr(enc, "is_sequence_capable", False):
        raise SystemExit(f"{name}: encoder is not sequence-capable, nothing to mean-pool")
    return enc


@torch.no_grad()
def encode_all(enc, smiles: list, batch_size: int, device) -> np.ndarray:
    """Pooled vector per molecule, in the order given."""
    out = []
    for i in range(0, len(smiles), batch_size):
        chunk = smiles[i:i + batch_size]
        _, pooled, _ = enc.encode(chunk)
        out.append(pooled.detach().to("cpu").float().numpy())
    return np.concatenate(out, axis=0)


def audit(name: str, mat: np.ndarray, smiles: list) -> list:
    """Checks that must pass before 505 runs read this table. Returns a list of problems."""
    problems = []
    expected = REP_SPECS[name]["dim"]
    if mat.shape[1] != expected:
        problems.append(f"width {mat.shape[1]}, expected {expected} - the registry in reps.py is wrong")

    bad = ~np.isfinite(mat)
    if bad.any():
        rows = sorted(set(np.where(bad)[0].tolist()))[:3]
        problems.append(f"{int(bad.sum())} non-finite value(s) in {len(set(np.where(bad)[0]))} "
                        f"molecule(s), e.g. {[smiles[r][:40] for r in rows]}")

    zero_rows = np.where(~mat.any(axis=1))[0]
    if len(zero_rows):
        problems.append(f"{len(zero_rows)} all-zero vector(s) - the encoder produced nothing for "
                        f"these molecules, e.g. {[smiles[r][:40] for r in zero_rows[:3]]}")

    # Two different molecules sharing a vector means the encoder is not distinguishing them, which
    # would silently make the secondary column less informative than it looks.
    uniq = len({r.tobytes() for r in np.ascontiguousarray(mat)})
    if uniq < len(smiles):
        problems.append(f"only {uniq} distinct vectors for {len(smiles)} molecules "
                        f"({len(smiles) - uniq} collision(s))")

    dead = int((mat.std(axis=0) < 1e-6).sum())
    if dead:
        problems.append(f"{dead} of {mat.shape[1]} dimension(s) are constant across all molecules "
                        "(they survive z-scoring as exact zeros, so they add parameters and no "
                        "information)")
    return problems


def run_one(name: str, mols: pd.DataFrame, device, batch_size: int, overwrite: bool) -> dict:
    path = embedding_table_path(name)
    if os.path.exists(path) and not overwrite:
        df = pd.read_csv(path)
        print(f"[{name}] already exists ({len(df)} molecules, {df.shape[1] - 2} dims) - skipping. "
              "Pass --overwrite to recompute.")
        return {"encoder": name, "status": "skipped", "path": path}

    t0 = time.time()
    smiles = mols["smiles"].tolist()
    print(f"[{name}] encoding {len(smiles)} molecules on {device} ...", flush=True)
    enc = build_encoder(name, device)
    mat = encode_all(enc, smiles, batch_size, device)

    problems = audit(name, mat, smiles)
    for p in problems:
        print(f"[{name}] WARN {p}")

    hard = [p for p in problems if "non-finite" in p or "all-zero" in p or "expected" in p]
    if hard:
        raise SystemExit(f"[{name}] refusing to write a table with: " + "; ".join(hard))

    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    out = pd.concat(
        [mols.reset_index(drop=True),
         pd.DataFrame(mat, columns=[f"f{i}" for i in range(mat.shape[1])])],
        axis=1)
    tmp = f"{path}.{os.getpid()}.tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)       # atomic, so a half-written table can never be read as a real one

    del enc
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    secs = time.time() - t0
    print(f"[{name}] wrote {len(out)} x {mat.shape[1]} to {path}  "
          f"(range {mat.min():+.3f}..{mat.max():+.3f}, mean |v| {np.abs(mat).mean():.4f}, "
          f"{secs:.0f}s)")
    return {"encoder": name, "status": "written", "path": path, "molecules": len(out),
            "dim": int(mat.shape[1]), "problems": problems, "seconds": round(secs, 1)}


def main():
    p = argparse.ArgumentParser(
        description="Precompute the 5 frozen encoders' mean-pooled vectors for the grid.")
    p.add_argument("--encoders", nargs="+", default=DEEP, help=f"default: all ({', '.join(DEEP)})")
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="api_cluster")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    for name in args.encoders:
        if name not in DEEP:
            raise SystemExit(f"'{name}' is not a deep encoder. Choices: {', '.join(DEEP)}")

    paths = SplitPaths(args.data_root, args.split)
    check_heldout_is_external(paths)
    device = get_device(args.device or "auto")
    mols = molecule_table(paths)

    print(f"data root : {paths.root}")
    print(f"molecules : {len(mols)} distinct, across train + val + the 22 held-out pairs")
    print(f"device    : {device}")
    print(f"output    : {EMBEDDING_DIR}\n")

    results, failed = [], []
    for name in args.encoders:
        try:
            results.append(run_one(name, mols, device, args.batch_size, args.overwrite))
        except SystemExit:
            raise
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"[{name}] FAILED - {type(e).__name__}: {e}\n")

    print()
    warned = [r for r in results if r.get("problems")]
    print(f"written/skipped: {len(results)} | failed: {len(failed)}")
    for name, err in failed:
        print(f"  failed : {name} - {err}")
    for r in warned:
        print(f"  warning: {r['encoder']} - {len(r['problems'])} issue(s), see above")
    if failed:
        raise SystemExit(1)
    print("\nNext: python -m experiments.encoder_comparison.run_grid --dry-run")


if __name__ == "__main__":
    main()
