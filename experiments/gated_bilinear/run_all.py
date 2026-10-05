"""Script to run all or selected Gated Bilinear models sequentially.

Usage:
    python -m experiments.gated_bilinear.run_all
    python -m experiments.gated_bilinear.run_all --skip gat
    python -m experiments.gated_bilinear.run_all --families maccs morgan pubchem
"""

import argparse
import subprocess
import sys
import time

ALL_FAMILIES = [
    "maccs",
    "morgan",
    "pubchem",
    "mol2vec",
    "molformer",
    "chemberta",
    "gin",
    "gat",
    "dmpnn",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple Gated Bilinear models sequentially")
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="List of families to run. If not set, runs all available families.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=["gat"],
        help="List of families to skip (default: gat, since it was already trained).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.families:
        to_run = [f for f in args.families if f not in (args.skip or [])]
    else:
        to_run = [f for f in ALL_FAMILIES if f not in (args.skip or [])]

    print("=" * 60)
    print(f"GATED BILINEAR SEQUENTIAL RUNNER")
    print(f"Models to train ({len(to_run)}): {', '.join(to_run)}")
    print("=" * 60)

    start_total = time.time()
    results = {}

    for idx, family in enumerate(to_run, 1):
        print(f"\n[{idx}/{len(to_run)}] Starting training for family: {family.upper()}")
        print("-" * 60)
        t0 = time.time()
        
        cmd = [sys.executable, "-m", "experiments.gated_bilinear.train_gb", "--family", family]
        ret = subprocess.run(cmd)
        
        elapsed = time.time() - t0
        success = (ret.returncode == 0)
        results[family] = "SUCCESS" if success else f"FAILED (code {ret.returncode})"
        
        print("-" * 60)
        print(f"[{idx}/{len(to_run)}] Finished {family.upper()} in {elapsed:.1f}s — Status: {results[family]}")

    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    print("RUN COMPLETE SUMMARY")
    print(f"Total time elapsed: {total_time/60:.2f} minutes")
    print("=" * 60)
    for family, status in results.items():
        print(f"  {family:12s} : {status}")


if __name__ == "__main__":
    main()
