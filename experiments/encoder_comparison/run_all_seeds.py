"""Run every encoder family on several seeds, one subprocess per run.

Each run is a separate `run_comparison.py` process, so every seed starts from a clean state.
Runs that already have results are skipped, so the script can be stopped and started again and
only does what is left. Logs go to experiments/results/encoder_comparison/_logs/.

    python -m experiments.encoder_comparison.run_all_seeds
    python -m experiments.encoder_comparison.run_all_seeds --seeds 43 44 45 46
    python -m experiments.encoder_comparison.run_all_seeds --families morgan chemberta gat
    python -m experiments.encoder_comparison.run_all_seeds --dry-run

Seed 42 of all nine families is already done, so the default leaves 36 runs.
"""

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.encoder_comparison.families import ORDER  # noqa: E402
from experiments.encoder_comparison.run_comparison import RESULTS_ROOT, run_name  # noqa: E402

LOG_DIR = os.path.join(RESULTS_ROOT, "_logs")
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_comparison.py")


def is_done(family: str, seed: int) -> bool:
    """A run counts as done only when its LAST outputs exist, not its first.

    run_comparison.py writes val_metrics.json first and the sealed held-out files last, so keying
    on val_metrics.json alone marks a run killed in between as finished forever, with no held-out
    metrics. The grid runner (run_grid.py) uses an explicit COMPLETE sentinel; this older script
    stays as the record of how the 45 `none`-column runs were produced, so it only gains the
    stricter check.
    """
    run = run_name(family, seed)
    selection = os.path.join(RESULTS_ROOT, run)
    sealed = os.path.join(RESULTS_ROOT, "_heldout_sealed", run)
    required = [
        os.path.join(selection, "val_metrics.json"),
        os.path.join(selection, "run_info.json"),
        os.path.join(selection, "predictions_val.csv"),
        os.path.join(sealed, "heldout_metrics.json"),
        os.path.join(sealed, "predictions_heldout.csv"),
    ]
    return all(os.path.exists(p) for p in required)


def main():
    p = argparse.ArgumentParser(description="Run the encoder comparison over several seeds.")
    p.add_argument("--families", nargs="+", default=ORDER, help=f"default: all ({', '.join(ORDER)})")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--device", default="cuda", help="use ONE device for every run (default cuda)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="api_cluster")
    p.add_argument("--overwrite", action="store_true", help="rerun even if results exist")
    p.add_argument("--dry-run", action="store_true", help="list what would run, then stop")
    args = p.parse_args()

    for f in args.families:
        if f not in ORDER:
            raise SystemExit(f"unknown family '{f}'. Choices: {', '.join(ORDER)}")

    # seed-major order: a complete sweep of all families finishes before the next seed starts,
    # so stopping early still leaves whole seeds that can be compared
    planned = [(s, f) for s in args.seeds for f in args.families]
    todo = [(s, f) for s, f in planned if args.overwrite or not is_done(f, s)]
    done_already = len(planned) - len(todo)

    print(f"{len(args.families)} families x {len(args.seeds)} seeds = {len(planned)} runs")
    print(f"already done: {done_already} | to run now: {len(todo)}")
    if args.dry_run or not todo:
        for s, f in todo:
            print(f"  would run {run_name(f, s)}")
        if not todo:
            print("nothing to do. Collect with: python -m experiments.encoder_comparison.collect")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    t0 = time.time()
    failed = []
    for i, (seed, family) in enumerate(todo, 1):
        log = os.path.join(LOG_DIR, f"{run_name(family, seed)}.txt")
        cmd = [sys.executable, RUNNER, "--family", family, "--seed", str(seed),
               "--device", args.device, "--epochs", str(args.epochs), "--split", args.split]
        if args.data_root:
            cmd += ["--data-root", args.data_root]
        if args.overwrite:
            cmd += ["--overwrite"]
        print(f"[{i}/{len(todo)}] {run_name(family, seed)} ... (log {log})", flush=True)
        with open(log, "w") as fh:
            code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
        mins = (time.time() - t0) / 60
        rate = mins / i
        print(f"      exit {code} | {mins:.1f} min elapsed | ~{rate * (len(todo) - i):.0f} min left", flush=True)
        if code != 0:
            failed.append(run_name(family, seed))
            print(f"      FAILED, see {log}", flush=True)

    print(f"\nfinished {len(todo) - len(failed)}/{len(todo)} runs in {(time.time() - t0) / 60:.1f} min")
    for r in failed:
        print(f"  failed: {r}")
    print("\nNext: python -m experiments.encoder_comparison.collect")


if __name__ == "__main__":
    main()
