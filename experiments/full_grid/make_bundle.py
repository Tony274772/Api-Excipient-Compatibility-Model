"""Zip the parts of the project Colab needs, so the whole repository is not uploaded.

    python experiments/full_grid/make_bundle.py

Writes grid_bundle.zip next to the project folder. Upload that to the top level of Google Drive,
then run experiments/full_grid/run_grid_colab.ipynb.

Included: the library, the entry points, the grid scripts, the 80/20 data and the pretrained
encoder weights. Excluded: results, checkpoints, caches and everything the grid does not read.
"""

import os
import sys
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(PROJECT_ROOT), "grid_bundle.zip")

INCLUDE_DIRS = [
    "src",                      # the library
    "experiments/full_grid",    # the grid scripts
    "data/grid_8020",           # the 80/20 split, descriptors and train-only norm stats
    "models/pretrained",        # encoder weights (~300 MB)
]
INCLUDE_FILES = ["main.py", "requirements.txt"]
# vector tables the mol2vec and pubchemfp families look up
INCLUDE_FILES += ["data/mol2vec_embeddings.csv", "data/mol2vec_embeddings_heldout.csv",
                  "data/pubchem_fps.csv", "data/pubchem_fps_heldout.csv"]
SKIP = ("__pycache__", ".pyc", ".ipynb_checkpoints")


def add(z, rel):
    full = os.path.join(PROJECT_ROOT, rel)
    if os.path.isfile(full):
        z.write(full, os.path.join("project", rel))
        return 1, os.path.getsize(full)
    n, size = 0, 0
    for root, dirs, files in os.walk(full):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            if any(s in f for s in SKIP):
                continue
            p = os.path.join(root, f)
            z.write(p, os.path.join("project", os.path.relpath(p, PROJECT_ROOT)))
            n += 1
            size += os.path.getsize(p)
    return n, size


def main():
    missing = [p for p in INCLUDE_DIRS + INCLUDE_FILES if not os.path.exists(os.path.join(PROJECT_ROOT, p))]
    if "data/grid_8020" in missing:
        sys.exit("data/grid_8020 is missing. Run: python experiments/full_grid/prepare_data.py")
    for m in missing:
        print(f"  warning: {m} not found, skipped")

    total_files = total_bytes = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in INCLUDE_DIRS + INCLUDE_FILES:
            if rel in missing:
                continue
            n, size = add(z, rel)
            total_files += n
            total_bytes += size
            print(f"  {rel:34s} {n:5d} files  {size / 1e6:8.1f} MB")
    print(f"\n{OUT}")
    print(f"  {total_files} files, {total_bytes / 1e6:.0f} MB uncompressed, "
          f"{os.path.getsize(OUT) / 1e6:.0f} MB zipped")
    print("\nUpload it to the top level of your Google Drive, then open "
          "experiments/full_grid/run_grid_colab.ipynb in Colab.")


if __name__ == "__main__":
    main()
