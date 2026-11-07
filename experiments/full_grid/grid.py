"""The model grid: every configuration that has a folder in metrics/.

The folder names were produced by Config.resolve_checkpoint_paths(), so the grid is generated the
same way and matched back to those names rather than parsed by hand. 54 of the 63 folders match a
current configuration exactly. The other 9 use an older naming scheme that left the pooling out of
the name; they are listed in LEGACY with pooling set to the Config default
(global_gated_attention), and the ones that then collide with a 4-token name are dropped as
duplicates of it.

    python experiments/full_grid/grid.py            # list the grid
    python experiments/full_grid/grid.py --check    # also report folders nothing matches
"""

import argparse
import itertools
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config import Config  # noqa: E402

ENCODERS = ["molformer", "chemberta", "pretrained_gin", "pretrained_gat", "dmpnn_chemprop", "fixed_vector"]
SOURCES = ["morgan", "maccs", "pubchemfp", "mol2vec"]
FUSIONS = ["cross_attn", "concat"]
POOLINGS = ["global_gated_attention", "cls", "explicit_pairwise"]
LOSSES = ["asl", "bce", "focal", "weighted_bce"]

# folders in metrics/ that are not model runs
NOT_RUNS = {"oct6_resplit", "random_split", "split_8020", "_smoke_pair", "final_8020_noleak"}

# older runs whose folder name has no pooling token; pooling is the Config default
LEGACY_POOLING = "global_gated_attention"


def run_name(encoder, source, fusion, pooling, loss, pgb):
    """The folder name this configuration produces, from the project's own naming code."""
    c = Config()
    c.encoder, c.fusion, c.pooling, c.loss, c.use_pgb_head = encoder, fusion, pooling, loss, pgb
    if source:
        c.fixed_vector_source = source
    c.resolve_checkpoint_paths()
    return c.metrics_dir.split("/", 1)[1]


def metrics_folders():
    d = os.path.join(PROJECT_ROOT, "metrics")
    return {x for x in os.listdir(d) if x not in NOT_RUNS and os.path.isdir(os.path.join(d, x))}


def build_grid():
    """-> (grid, unmatched). grid maps run name -> config dict."""
    want = metrics_folders()
    grid = {}
    for enc, fus, pool, loss, pgb in itertools.product(ENCODERS, FUSIONS, POOLINGS, LOSSES, [False, True]):
        for src in (SOURCES if enc == "fixed_vector" else [None]):
            name = run_name(enc, src, fus, pool, loss, pgb)
            if name in want and name not in grid:
                grid[name] = dict(name=name, encoder=enc, source=src, fusion=fus,
                                  pooling=pool, loss=loss, pgb=pgb, legacy=False)
    unmatched = sorted(want - set(grid))

    # the 9 older names: same axes, pooling not in the name
    for name in unmatched:
        parts = name.split("_")
        pgb = parts[0] == "pgb"
        if pgb:
            parts = parts[1:]
        for enc in ENCODERS:
            e = enc.split("_")
            if parts[:len(e)] != e:
                continue
            rest = parts[len(e):]
            for fus in FUSIONS:
                f = fus.split("_")
                if rest[:len(f)] != f:
                    continue
                loss = "_".join(rest[len(f):])
                if loss not in LOSSES:
                    continue
                canonical = run_name(enc, None, fus, LEGACY_POOLING, loss, pgb)
                if canonical in grid:
                    break      # the 4-token name is already in the grid; this is the same model
                grid[name] = dict(name=name, encoder=enc, source=None, fusion=fus,
                                  pooling=LEGACY_POOLING, loss=loss, pgb=pgb, legacy=True)
                break
    still = sorted(want - set(grid))
    return grid, still


GRID, UNMATCHED = build_grid()
ORDER = sorted(GRID, key=lambda n: (GRID[n]["encoder"] != "fixed_vector", GRID[n]["encoder"], n))


def spec(name):
    if name not in GRID:
        raise SystemExit(f"unknown run '{name}'. {len(GRID)} available; see "
                         "python experiments/full_grid/grid.py")
    return GRID[name]


def main():
    p = argparse.ArgumentParser(description="List the model grid.")
    p.add_argument("--check", action="store_true", help="also report folders nothing matches")
    args = p.parse_args()
    legacy = [n for n in ORDER if GRID[n]["legacy"]]
    print(f"{len(GRID)} configurations ({len(legacy)} from the older naming scheme)\n")
    for i, n in enumerate(ORDER, 1):
        g = GRID[n]
        tag = "PGB " if g["pgb"] else "    "
        print(f"{i:3d}. {tag}{n}" + ("   [legacy name, pooling assumed default]" if g["legacy"] else ""))
    if args.check:
        print(f"\nmetrics/ folders no configuration reproduces: {UNMATCHED or 'none'}")


if __name__ == "__main__":
    main()
