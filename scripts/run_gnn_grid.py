"""Run the full 5 encoder x 2 fusion x 4 loss = 40 experiment grid.

Usage:
    python scripts/run_gnn_grid.py                    # run all 40
    python scripts/run_gnn_grid.py --dry_run           # print commands only
    python scripts/run_gnn_grid.py --encoder gine      # run only GINE (8 combos)
    python scripts/run_gnn_grid.py --smoke             # 2-epoch smoke test

This script runs each combination as a subprocess, logging successes and failures.
"""

import argparse
import os
import subprocess
import sys
from itertools import product


ENCODERS = [
    "dmpnn_chemprop",
    "pretrained_gin",
    "pretrained_gat",
]

FUSIONS = ["concat", "cross_attn"]

LOSSES = ["bce", "weighted_bce", "focal", "asl"]


def build_command(encoder, fusion, loss, smoke=False, extra_args=None):
    """Build the main.py command for one experiment."""
    cmd = [
        sys.executable, "main.py",
        "--encoder", encoder,
        "--fusion", fusion,
        "--loss", loss,
    ]

    if fusion == "cross_attn":
        cmd += ["--pooling", "global_gated_attention"]

    if smoke:
        cmd += ["--max_epochs", "2", "--batch_size", "4"]

    if extra_args:
        cmd += extra_args

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run GNN experiment grid")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    parser.add_argument("--smoke", action="store_true", help="2-epoch smoke test")
    parser.add_argument("--encoder", nargs="+", default=None,
                        help="Limit to specific encoders")
    parser.add_argument("--fusion", nargs="+", default=None,
                        help="Limit to specific fusions")
    parser.add_argument("--loss", nargs="+", default=None,
                        help="Limit to specific losses")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue to next combination on error")
    args = parser.parse_args()

    encoders = args.encoder or ENCODERS
    fusions = args.fusion or FUSIONS
    losses = args.loss or LOSSES

    total = len(encoders) * len(fusions) * len(losses)
    print(f"GNN Grid: {len(encoders)} encoders x {len(fusions)} fusions x {len(losses)} losses = {total} combinations")

    if args.smoke:
        print("MODE: Smoke test (2 epochs, batch_size=4)")

    results = {"passed": [], "failed": [], "skipped": []}

    for i, (encoder, fusion, loss) in enumerate(product(encoders, fusions, losses)):
        combo_name = f"{encoder}_{fusion}_{loss}"
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {combo_name}")
        print(f"{'='*60}")

        cmd = build_command(encoder, fusion, loss, smoke=args.smoke)

        if args.dry_run:
            print(f"  CMD: {' '.join(cmd)}")
            results["skipped"].append(combo_name)
            continue

        try:
            result = subprocess.run(
                cmd,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout per run
            )

            if result.returncode == 0:
                print(f"  PASSED")
                results["passed"].append(combo_name)
            else:
                print(f"  FAILED (exit code {result.returncode})")
                print(f"  STDERR: {result.stderr[-500:]}" if result.stderr else "")
                results["failed"].append((combo_name, result.stderr[-500:] if result.stderr else ""))
                if not args.continue_on_error:
                    print("Stopping. Use --continue_on_error to skip failures.")
                    break

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (2h)")
            results["failed"].append((combo_name, "timeout"))
            if not args.continue_on_error:
                break
        except Exception as e:
            print(f"  ERROR: {e}")
            results["failed"].append((combo_name, str(e)))
            if not args.continue_on_error:
                break

    # Summary
    print(f"\n{'='*60}")
    print("GRID SUMMARY")
    print(f"{'='*60}")
    print(f"Passed:  {len(results['passed'])}")
    print(f"Failed:  {len(results['failed'])}")
    print(f"Skipped: {len(results['skipped'])}")

    if results["failed"]:
        print("\nFailed combinations:")
        for item in results["failed"]:
            if isinstance(item, tuple):
                print(f"  - {item[0]}: {item[1][:100]}")
            else:
                print(f"  - {item}")


if __name__ == "__main__":
    main()
