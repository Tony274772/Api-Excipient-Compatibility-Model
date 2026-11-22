"""Run the planned ablations x seeds, one subprocess per run, resumable and shardable.

A run is done when its COMPLETE sentinel exists; everything else is (re)run. One failing run does
not stop the sweep (unless --stop-on-failure); re-running picks up exactly what is missing.

    python -m experiments.ablations_dmpnn_maccs.run_seeds --dry-run
    python -m experiments.ablations_dmpnn_maccs.run_seeds                       # the whole plan
    python -m experiments.ablations_dmpnn_maccs.run_seeds --ablations full      # step 0 only
    python -m experiments.ablations_dmpnn_maccs.run_seeds --ablations no_w no_prior --seeds 42 43
    CUDA_VISIBLE_DEVICES=1 python -m experiments.ablations_dmpnn_maccs.run_seeds --shard 1/4

Step 0 (`full`, all seeds) always runs first in the queue: every comparison is against it.
"""

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.ablations_dmpnn_maccs import variants as V  # noqa: E402

MODEL_CELL = "dmpnn__x_maccs"
SEEDS = [42, 43, 44, 45, 46]
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "results", "ablations_dmpnn_maccs")
RUNS_ROOT = os.path.join(RESULTS_ROOT, "runs")
SENTINEL = "COMPLETE"


def run_name(ablation: str, seed: int) -> str:
    return f"{ablation}_seed{seed}"


def is_complete(ablation: str, seed: int) -> bool:
    return os.path.exists(os.path.join(RUNS_ROOT, run_name(ablation, seed), SENTINEL))


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
        cmd = [sys.executable, "-m", "experiments.ablations_dmpnn_maccs.run_ablation",
               "--ablation", a, "--seed", str(s), "--device", args.device,
               "--epochs", str(args.epochs)]
        if args.data_root:
            cmd += ["--data-root", args.data_root]
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"[{k}/{len(todo)}] {name} -> {log}")
        with open(log, "w") as f:
            code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        if code != 0:
            print(f"   FAILED (exit {code}); see {log}")
            failed.append((name, code))
            if args.stop_on_failure:
                break
        else:
            with open(log) as f:
                last = [line.strip() for line in f if "DONE" in line or "val PR-AUC" in line]
                print(f"   {last[-1] if last else 'done'}")

    mins = round((time.time() - t0) / 60, 1)
    print(f"\nQueue finished in {mins} min. {len(todo) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        for name, code in failed:
            print(f"   failed: {name} (exit {code})")
        sys.exit(1)


if __name__ == "__main__":
    main()
