"""Check a fresh clone can run the grid, and say exactly what is missing.

A git clone does not carry models/pretrained/ (it is gitignored, ~300 MB), so the encoders that
need weights cannot run until those are in place. This reports, per encoder family, whether its
weights are present, can be downloaded, or must be copied from the machine that has them.

    python experiments/full_grid/check_setup.py
    python experiments/full_grid/check_setup.py --download   # fetch what can be fetched
"""

import argparse
import importlib
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from experiments.full_grid.grid import GRID  # noqa: E402

# family -> (how many configs need it, local path, how it is obtained if absent)
WEIGHTS = {
    "fixed_vector": (None, "computed from SMILES; the lookup tables are in the repository"),
    "chemberta": ("models/pretrained/chemberta",
                  "downloads itself from Hugging Face (DeepChem/ChemBERTa-77M-MTR) on first use"),
    "molformer": ("models/pretrained/molformer",
                  "downloads itself from Hugging Face (ibm/MoLFormer-XL-both-10pct) on first use"),
    "pretrained_gin": ("models/pretrained/gin/gin_supervised_contextpred_pre_trained.pth",
                       "falls back to dgllife's 'gin_supervised_contextpred', which needs dgl + dgllife"),
    "pretrained_gat": ("models/pretrained/stanford_gat/gat_contextpred.pth",
                       "python scripts/download_gnn_resources.py --resource stanford_gat"),
    "dmpnn_chemprop": ("models/pretrained/chemeleon_mp.pt",
                       "python scripts/download_gnn_resources.py --resource chemeleon"),
}
PACKAGES = ["torch", "rdkit", "sklearn", "pandas", "numpy", "transformers", "chemprop", "torch_geometric"]
DATA = ["data/grid_8020/train.csv", "data/grid_8020/val.csv", "data/grid_8020/test.csv",
        "data/grid_8020/api_descriptors.csv", "data/grid_8020/excipient_descriptors.csv",
        "data/grid_8020/descriptor_norm_stats.json",
        "data/pubchem_fps.csv", "data/pubchem_fps_heldout.csv",
        "data/mol2vec_embeddings.csv", "data/mol2vec_embeddings_heldout.csv"]


def counts():
    n = {}
    for g in GRID.values():
        n[g["encoder"]] = n.get(g["encoder"], 0) + 1
    return n


def main():
    p = argparse.ArgumentParser(description="Check this machine can run the grid.")
    p.add_argument("--download", action="store_true", help="fetch the two checkpoints that have a downloader")
    args = p.parse_args()

    print("packages")
    missing_pkg = []
    for m in PACKAGES:
        try:
            mod = importlib.import_module(m)
            print(f"  ok      {m:18s} {getattr(mod, '__version__', '?')}")
        except Exception:
            missing_pkg.append(m)
            print(f"  MISSING {m}")

    print("\ndata (comes with the clone)")
    missing_data = [f for f in DATA if not os.path.exists(f)]
    for f in DATA:
        print(f"  {'ok     ' if os.path.exists(f) else 'MISSING'} {f}")
    if "data/grid_8020/train.csv" in missing_data:
        print("     -> python experiments/full_grid/prepare_data.py   (needs the ARES data folder)")

    print("\nencoder weights (NOT in the clone: models/pretrained/ is gitignored)")
    n = counts()
    runnable = 0
    blocked = []
    for fam, (path, how) in WEIGHTS.items():
        k = n.get(fam, 0)
        if path is None:
            print(f"  ok      {fam:16s} {k:2d} configs   {how}")
            runnable += k
            continue
        if os.path.exists(path):
            print(f"  ok      {fam:16s} {k:2d} configs   {path}")
            runnable += k
        else:
            print(f"  ABSENT  {fam:16s} {k:2d} configs   {how}")
            blocked.append((fam, k, path, how))

    if args.download:
        print("\ndownloading what has a downloader ...")
        for res in ("chemeleon", "stanford_gat"):
            print(f"  {res} ...", flush=True)
            subprocess.call([sys.executable, "scripts/download_gnn_resources.py", "--resource", res])

    print(f"\n{runnable} of {len(GRID)} configurations can run right now.")
    if missing_pkg:
        print(f"\ninstall: pip install {' '.join('scikit-learn' if m == 'sklearn' else m for m in missing_pkg)}")
    if blocked and not args.download:
        auto = [b for b in blocked if "download" in b[3] or "Hugging Face" in b[3]]
        if auto:
            print("\nfetchable on this machine:")
            print("  python experiments/full_grid/check_setup.py --download")
            print("  (chemberta and molformer need no action: they download themselves on first use)")
        print("\nor copy the whole folder from the machine that has it, which needs no internet:")
        print("  scp -r <you>@<that machine>:<path>/project/models/pretrained models/")
    print("\nWhen ready: python experiments/full_grid/run_grid.py --all --device cuda")


if __name__ == "__main__":
    main()
