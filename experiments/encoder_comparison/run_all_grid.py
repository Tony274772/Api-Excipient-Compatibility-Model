"""Run the 101 grid cells x 5 seeds, one subprocess per run, resumable and multi-GPU.

Each run is a separate `run_grid.py` process, so every run starts from a clean state. A run counts
as finished only when its `COMPLETE` sentinel exists, so the sweep can be stopped and restarted and
will redo anything that was killed part-way. Logs go to
experiments/results/encoder_grid/_logs/.

Order follows Part 8 of paper/encoder_grid_plan.md, and it matters:

    1  floor + pooling        30 runs   pooling must be settled before any deep row runs
    2  shuffled + no-cross-attn 115 runs these decide how every later number is read
    3  fingerprint secondaries 160 runs
    4  encoder secondaries     200 runs

Stopping after block 3 still gives a complete, publishable answer with every control in place.

    # once, before anything else
    python -m experiments.encoder_comparison.precompute_embeddings

    python -m experiments.encoder_comparison.run_all_grid --dry-run
    python -m experiments.encoder_comparison.run_all_grid --blocks floor pooling
    python -m experiments.encoder_comparison.run_all_grid --blocks shuffled no_cross_attn
    python -m experiments.encoder_comparison.run_all_grid                       # everything left

Across several GPUs, give each process a disjoint slice of the same queue:

    CUDA_VISIBLE_DEVICES=0 python -m experiments.encoder_comparison.run_all_grid --shard 0/4 &
    CUDA_VISIBLE_DEVICES=1 python -m experiments.encoder_comparison.run_all_grid --shard 1/4 &
    CUDA_VISIBLE_DEVICES=2 python -m experiments.encoder_comparison.run_all_grid --shard 2/4 &
    CUDA_VISIBLE_DEVICES=3 python -m experiments.encoder_comparison.run_all_grid --shard 3/4 &

Shards never collide: the queue is built identically in every process and each takes only the
runs whose index mod N equals its own, so no two processes write the same folder.
"""

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.encoder_comparison.families import (BLOCK_ORDER, FAMILIES, all_cells,  # noqa: E402
                                                     grid_summary)
from experiments.encoder_comparison.reps import EMBEDDING_REPS  # noqa: E402
from experiments.encoder_comparison.data import embedding_table_path  # noqa: E402
from experiments.encoder_comparison.run_grid import (RESULTS_ROOT, is_complete,  # noqa: E402
                                                     run_name)

LOG_DIR = os.path.join(RESULTS_ROOT, "_logs")
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_grid.py")

SEEDS = [42, 43, 44, 45, 46]


def missing_embedding_tables(cells) -> list:
    """Which mean-pooled encoder tables these cells need and do not have."""
    needed = set()
    for c in cells:
        for r in c.representations():
            base = r.replace("shuf_", "")
            if base in EMBEDDING_REPS:
                needed.add(base)
    return [n for n in sorted(needed) if not os.path.exists(embedding_table_path(n))]


def unimportable_encoders(cells) -> dict:
    """Deep methods whose encoder class cannot even be imported, with the reason.

    A missing table only blocks the cells that read it. A method used as a cross-attended PRIMARY
    builds the live encoder instead, so it needs the library at run time: without dgl, the 9 cells
    reading gin's table are skipped by the table check while `gin`, `gin__pool_cls` and the 9
    `gin__x_*` cells would each still start, fail in the encoder constructor and be logged as a
    failure. They fail in seconds rather than wasting GPU hours, but a sweep that reports 55
    failures is a sweep nobody can read, so they are blocked here with one clear reason.

    Only the import is attempted - no weights are loaded, so this costs nothing.
    """
    from src.encoders import ENCODER_REGISTRY
    out = {}
    for name in sorted({c.primary for c in cells
                        if c.primary and not c.no_cross_attn
                        and FAMILIES.get(c.primary, {}).get("kind") == "seq"}):
        enc = FAMILIES[name]["encoder"]
        try:
            ENCODER_REGISTRY[enc]
        except Exception as e:
            out[name] = f"{type(e).__name__}: {e}"
    return out


def split_blocked(todo, missing: list, broken_encoders: dict = None):
    """Separate the runs that can start now from the ones waiting on a missing table.

    One unavailable encoder should not leave a DGX idle - gin needs dgl/dgllife, and without it 20
    of the 101 cells are blocked while the other 81 are perfectly runnable. But a grid with a hole
    in it must never look finished, so the blocked cells are named here, they are named again by
    every --dry-run, and collect_grid reports them as missing coverage rather than quietly
    averaging over what happens to exist.
    """
    broken_encoders = broken_encoders or {}
    if not missing and not broken_encoders:
        return list(todo), []
    blocked_reps = set(missing)

    def is_blocked(cell):
        if any(r.replace("shuf_", "") in blocked_reps for r in cell.representations()):
            return True
        # a live cross-attended primary whose encoder will not import
        return bool(cell.primary in broken_encoders and not cell.no_cross_attn
                    and FAMILIES.get(cell.primary, {}).get("kind") == "seq")

    runnable = [(c, s) for c, s in todo if not is_blocked(c)]
    blocked = [(c, s) for c, s in todo if is_blocked(c)]
    return runnable, blocked


def build_queue(blocks, seeds, cells_filter):
    """(cell, seed) pairs in block order, then cell order, then seed order.

    Cell-major within a block: all 5 seeds of a cell finish together, so a partial sweep still
    leaves whole cells that can be tested rather than a scattering of single seeds.
    """
    cells = all_cells(blocks)
    if cells_filter:
        wanted = set(cells_filter)
        unknown = wanted - {c.name for c in cells}
        if unknown:
            raise SystemExit(f"unknown cell(s): {', '.join(sorted(unknown))}")
        cells = [c for c in cells if c.name in wanted]
    order = {b: i for i, b in enumerate(BLOCK_ORDER)}
    cells.sort(key=lambda c: order.get(c.block, 99))
    return [(c, s) for c in cells for s in seeds]


def main():
    p = argparse.ArgumentParser(description="Run the encoder grid: 101 cells x 5 seeds.")
    p.add_argument("--blocks", nargs="+", default=None,
                   help=f"limit to these blocks, in order: {', '.join(BLOCK_ORDER)}")
    p.add_argument("--cells", nargs="+", default=None, help="limit to these cell names")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--device", default="cuda", help="device for every run (default cuda)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="api_cluster")
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="take only runs whose queue index mod N == I (for several GPUs)")
    p.add_argument("--overwrite", action="store_true", help="rerun even where COMPLETE exists")
    p.add_argument("--require-all-tables", action="store_true",
                   help="refuse to run anything while any encoder's embedding table is missing")
    p.add_argument("--dry-run", action="store_true", help="list what would run, then stop")
    p.add_argument("--stop-on-failure", action="store_true",
                   help="abort the sweep at the first failed run instead of carrying on")
    args = p.parse_args()

    if args.blocks:
        unknown = set(args.blocks) - set(BLOCK_ORDER)
        if unknown:
            raise SystemExit(f"unknown block(s): {', '.join(sorted(unknown))}. "
                             f"Choices: {', '.join(BLOCK_ORDER)}")

    queue = build_queue(args.blocks, args.seeds, args.cells)

    shard_i, shard_n = 0, 1
    if args.shard:
        try:
            shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit("--shard takes I/N, e.g. --shard 0/4")
        if not 0 <= shard_i < shard_n:
            raise SystemExit(f"--shard {args.shard}: need 0 <= I < N")
        queue = [item for i, item in enumerate(queue) if i % shard_n == shard_i]

    todo = [(c, s) for c, s in queue if args.overwrite or not is_complete(c.name, s)]
    done_already = len(queue) - len(todo)

    missing = missing_embedding_tables([c for c, _ in todo])
    broken = unimportable_encoders([c for c, _ in todo])
    todo, blocked = split_blocked(todo, missing, broken)

    print(grid_summary())
    print()
    if args.shard:
        print(f"shard {shard_i} of {shard_n}")
    print(f"runs in this queue: {len(queue)} | already COMPLETE: {done_already} | "
          f"to run now: {len(todo)}")

    if blocked:
        cells_blocked = sorted({c.name for c, _ in blocked})
        print()
        print(f"BLOCKED: {len(blocked)} run(s) across {len(cells_blocked)} cell(s) cannot run.")
        if missing:
            print(f"  no mean-pooled embedding table for: {', '.join(missing)}")
        for name, why in broken.items():
            print(f"  {name}: its encoder cannot be imported - {why}")
            if "dgl" in why:
                print("    pip install 'dgl>=1.1' 'dgllife>=0.3.2'      "
                      "(both are in requirements.txt)")
                print("    A CPU-only dgl wheel is fine: the encoder already runs the graph on "
                      "CPU and moves the output to the GPU.")
        print("  cells skipped:")
        for n in cells_blocked:
            print(f"    {n}")
        print("  Everything else proceeds. To fill these in later: install what is missing, run")
        print("    python -m experiments.encoder_comparison.precompute_embeddings")
        print("  then re-run this script - finished runs are skipped, so only the gaps are done.")
        if args.require_all_tables:
            raise SystemExit("--require-all-tables was passed, so nothing is run")

    if args.dry_run or not todo:
        for c, s in todo:
            print(f"  would run {run_name(c.name, s)}   [{c.block}]")
        if not todo:
            print("nothing runnable. Collect with: "
                  "python -m experiments.encoder_comparison.collect_grid")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    t0 = time.time()
    failed = []
    for i, (cell, seed) in enumerate(todo, 1):
        run = run_name(cell.name, seed)
        log = os.path.join(LOG_DIR, f"{run}.txt")
        cmd = [sys.executable, RUNNER, "--cell", cell.name, "--seed", str(seed),
               "--device", args.device, "--epochs", str(args.epochs), "--split", args.split]
        if args.data_root:
            cmd += ["--data-root", args.data_root]
        if args.overwrite:
            cmd += ["--overwrite"]
        print(f"[{i}/{len(todo)}] {run} [{cell.block}] ... (log {log})", flush=True)
        with open(log, "w") as fh:
            code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
        mins = (time.time() - t0) / 60
        rate = mins / i
        print(f"      exit {code} | {mins:.1f} min elapsed | "
              f"~{rate * (len(todo) - i):.0f} min left", flush=True)
        if code != 0:
            failed.append(run)
            print(f"      FAILED, see {log}", flush=True)
            if args.stop_on_failure:
                break
        elif not is_complete(cell.name, seed):
            # exit 0 without a sentinel means the run returned early; treat it as a failure so it
            # is retried rather than silently missing from the table.
            failed.append(run)
            print(f"      exit 0 but no COMPLETE sentinel, see {log}", flush=True)

    print(f"\nfinished {len(todo) - len(failed)}/{len(todo)} runs in "
          f"{(time.time() - t0) / 60:.1f} min")
    for r in failed:
        print(f"  failed: {r}")
    print("\nNext: python -m experiments.encoder_comparison.collect_grid")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
