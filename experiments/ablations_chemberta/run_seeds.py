"""Run the planned ablations x seeds, one subprocess per run, resumable and shardable.

A run is done when its COMPLETE sentinel exists; everything else is (re)run. One failing run does
not stop the sweep (unless --stop-on-failure); re-running picks up exactly what is missing.

    python -m experiments.ablations_chemberta.run_seeds --dry-run
    python -m experiments.ablations_chemberta.run_seeds                       # the whole plan
    python -m experiments.ablations_chemberta.run_seeds --ablations full      # step 0 only
    python -m experiments.ablations_chemberta.run_seeds --ablations no_w no_prior --seeds 42 43
    CUDA_VISIBLE_DEVICES=1 python -m experiments.ablations_chemberta.run_seeds --shard 1/4

Step 0 (`full`, all seeds) always runs first in the queue: every comparison is against it.
"""

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.ablations_chemberta import variants as V  # noqa: E402
from experiments.ablations_chemberta.run_ablation import (RESULTS_ROOT, SEEDS,  # noqa: E402
                                                          is_complete, run_name)

LOG_DIR = os.path.join(RESULTS_ROOT, "logs")


def build_queue(ablations, seeds):
    for a in ablations:
        V.parse_ablation(a)
    ordered = (["full"] if "full" in ablations else []) + [a for a in ablations if a != "full"]
    return [(a, s) for a in ordered for s in seeds]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ablations", nargs="+", default=None, help="default: variants.PLAN")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--data-root", default=None)
    p.add_argument("--shard", default=None, metavar="I/N", help="run every N-th job starting at I")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-failure", action="store_true")
    args = p.parse_args()

    queue = build_queue(args.ablations or V.PLAN, args.seeds)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < n:
            raise SystemExit("--shard I/N needs 0 <= I < N")
        queue = queue[i::n]
    todo = [(a, s) for a, s in queue if args.overwrite or not is_complete(a, s)]
    done = len(queue) - len(todo)
    print(f"{len(queue)} runs in this queue | {done} already COMPLETE | {len(todo)} to run")
    if args.dry_run:
        for a, s in todo:
            print(f"   {run_name(a, s)}")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    failed = []
    t0 = time.time()
    for k, (a, s) in enumerate(todo, 1):
        name = run_name(a, s)
        log = os.path.join(LOG_DIR, f"{name}.txt")
        cmd = [sys.executable, "-m", "experiments.ablations_chemberta.run_ablation",
               "--ablation", a, "--seed", str(s), "--device", args.device,
               "--epochs", str(args.epochs)]
        if args.data_root:
            cmd += ["--data-root", args.data_root]
        if args.overwrite:
            cmd += ["--overwrite"]
        print(f"[{k}/{len(todo)}] {name} ... (log {log})", flush=True)
        with open(log, "w") as fh:
            code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
        if code != 0 or not is_complete(a, s):
            failed.append(name)
            print(f"      FAILED (exit {code}), see {log}", flush=True)
            if args.stop_on_failure:
                break
    mins = (time.time() - t0) / 60
    print(f"\nfinished {len(todo) - len(failed)} of {len(todo)} runs in {mins:.1f} min")
    if failed:
        print("failed: " + " ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
