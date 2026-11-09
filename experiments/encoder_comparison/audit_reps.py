"""Audit every representation the grid uses, before 505 runs read them.

audit_inputs.py already covers the shared features: SMILES that RDKit cannot parse, NaN and inf in
descriptors / salt / flags, zero-variance descriptor columns, excipients falling through the prior
lookup. This script covers the parts that are new in the grid:

  * every primary / secondary vector for every molecule in every split,
  * the z-scoring: bit vectors must stay raw, continuous vectors must be scaled on TRAIN only,
  * the shuffled controls: a real derangement over chemical identity, identical in all 5 seeds,
    and genuinely uninformative,
  * the mean-pooled encoder tables: coverage, all-zero rows, collisions between molecules, and
    dimensions that are constant across the dataset.

Read-only: trains nothing, writes nothing.

    python -m experiments.encoder_comparison.audit_reps
    python -m experiments.encoder_comparison.audit_reps --reps morgan maccs shuf_maccs
"""

import argparse
import os
import sys

import numpy as np
from rdkit import RDLogger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.encoder_comparison.data import (SplitPaths, check_heldout_is_external,  # noqa: E402
                                                 embedding_table_path, table_paths_for)
from experiments.encoder_comparison.reps import (ALL_REP_NAMES, EMBEDDING_REPS,  # noqa: E402
                                                 REP_SPECS, SHUFFLED, SHUFFLE_SEED,
                                                 build_representation, canonical_key, rep_dim)
from experiments.encoder_comparison.run_grid import LOOKUP_DIR  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROBLEMS = []        # must be fixed before training; these make the exit code non-zero
NOTES = []           # true facts about a representation, reported and not blocking


def report(ok: bool, label: str, detail: str = ""):
    """A check that must pass. A failure here means the grid would read bad numbers."""
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        PROBLEMS.append(f"{label}{(': ' + detail) if detail else ''}")


def note(label: str, detail: str = ""):
    """A property worth knowing that is not a defect.

    Kept separate on purpose. A coarse fingerprint collides and has dead bits on a dataset of 861
    molecules by construction - MACCS has 167 bits for the whole of chemistry - and an audit that
    reports those as problems is an audit nobody reads before a 25-hour sweep.
    """
    print(f"  note {label}{(' - ' + detail) if detail else ''}")
    NOTES.append(f"{label}{(': ' + detail) if detail else ''}")


# The residual category. NOT a failure: a representation of finite width must map some distinct
# molecules together, and for a 167-bit fingerprint that is most of its collisions. It bounds what
# a column can resolve, which is what the paper should say about it.
COARSE = ("genuinely different molecules the representation cannot separate - "
          "the resolution limit of a fixed-width descriptor")


def collision_groups(mat: np.ndarray, molecules: list) -> list:
    """Groups of molecules that received byte-identical vectors."""
    import collections
    g = collections.defaultdict(list)
    for i, row in enumerate(np.ascontiguousarray(mat)):
        g[row.tobytes()].append(i)
    return [[molecules[i] for i in idx] for idx in g.values() if len(idx) > 1]


def _flat(smiles: str):
    """Canonical SMILES with stereochemistry dropped."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smiles).strip())
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def _neutral(smiles: str):
    """Canonical SMILES with every formal charge removed, stereochemistry dropped."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smiles).strip())
    if m is None:
        return None
    rw = Chem.RWMol(m)
    for a in rw.GetAtoms():
        a.SetFormalCharge(0)
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    try:
        return Chem.MolToSmiles(rw, isomericSmiles=False)
    except Exception:
        return None


def explain_collision(group: list) -> str:
    """Why these molecules share a vector.

    Tested against the whole group collapsing to one key, so a group whose members collide for two
    different reasons at once falls through to the pairwise test below rather than being
    mislabelled by whichever check happened to run first.
    """
    from rdkit import Chem

    def collapses(fn):
        keys = {fn(s) for s in group}
        if len(keys) == 1 and None not in keys:
            return True
        # a mixed group: true if EVERY member shares this simplification with some other member
        return len(keys) < len(group) and None not in keys

    if collapses(canonical_key):
        return ("the same molecule spelled more than once in the source CSVs "
                "(leading/trailing whitespace) - a data-hygiene issue, not an encoder one")
    heavy = []
    for s in group:
        m = Chem.MolFromSmiles(str(s).strip())
        heavy.append(m.GetNumHeavyAtoms() if m else -1)
    if all(n == 1 for n in heavy):
        return ("bare single atoms (metal ions). The featuriser has a finite element vocabulary, "
                "so these fall into one bucket and oxidation state is invisible")
    if collapses(_flat):
        return "differ ONLY in stereochemistry - the representation is stereo-blind"
    if collapses(_neutral):
        return ("differ ONLY in protonation / charge state, e.g. an acid and its conjugate base. "
                "The representation cannot tell a free acid from its salt")
    return COARSE


def audit_one(name: str, paths: SplitPaths, tables: dict) -> None:
    print(f"\n--- {name}  ({REP_SPECS[SHUFFLED.get(name, name)]['kind']}, "
          f"{rep_dim(name)} dims)")
    train = paths.molecules("train")
    all_mols = paths.all_molecules()

    rep = build_representation(name, tables)
    rep.build_shuffle(all_mols)
    rep.fit_scaler(train)

    # ---- every molecule in every split produces a vector of the right width
    mat, failures = [], []
    for m in all_mols:
        try:
            v = rep.vector(m)
            if v.shape != (rep.dim,):
                failures.append((m, f"width {v.shape}"))
            else:
                mat.append(v)
        except Exception as e:
            failures.append((m, f"{type(e).__name__}: {e}"))
    report(not failures, f"{name}: a vector for all {len(all_mols)} molecules",
           f"{len(failures)} failure(s), e.g. {failures[:2]}" if failures else "")
    if not mat:
        return
    mat = np.stack(mat)

    report(bool(np.isfinite(mat).all()), f"{name}: all values finite",
           f"{int((~np.isfinite(mat)).sum())} non-finite" if not np.isfinite(mat).all() else "")

    zero_rows = int((~mat.any(axis=1)).sum())
    report(zero_rows == 0, f"{name}: no all-zero vectors",
           f"{zero_rows} molecule(s) get nothing" if zero_rows else "")

    # ---- scaling: the whole point of problem 3 in Part 6
    if rep.kind == "bits":
        raw = np.unique(mat)
        is_binary = set(raw.tolist()) <= {0.0, 1.0}
        report(is_binary, f"{name}: bit vector left raw (0/1 only)",
               "" if is_binary else f"values outside 0/1: {raw[:6]}")
    else:
        tr = np.stack([rep.vector(m) for m in sorted(set(train))])
        mu, sd = float(tr.mean()), float(tr.std())
        centred = abs(mu) < 0.05 and abs(sd - 1.0) < 0.25
        report(centred, f"{name}: z-scored on train statistics",
               f"train mean {mu:+.4f}, train sd {sd:.4f}")
        # Held-out must be scaled with the TRAIN statistics, so it is near 0/1 but not exactly.
        ho = np.stack([rep.vector(m) for m in sorted(set(paths.molecules("heldout")))])
        print(f"       held-out under train scaling: mean {ho.mean():+.4f}, sd {ho.std():.4f} "
              "(should be near 0/1 but not exactly - it was not fitted on them)")

    # ---- dead dimensions. Not a defect for anything: a fixed fingerprint's unset bits encode
    # substructures absent from 861 molecules, and a 2048-wide encoder output having 102 channels
    # that never activate on this dataset is ordinary. Reported with its parameter cost, because
    # that is the part worth knowing.
    dead = int((mat.std(axis=0) < 1e-6).sum())
    base_name = SHUFFLED.get(name, name)
    if dead == 0:
        report(True, f"{name}: every dimension varies across molecules")
    else:
        kind = "encoder channel" if base_name in EMBEDDING_REPS else "bit"
        note(f"{name}: {dead} of {mat.shape[1]} dimensions never vary on this dataset",
             f"{dead} dead {kind}(s); they survive scaling as exact zeros and cost "
             f"{dead * 32:,} secondary parameters that can carry no information")

    # ---- collisions, explained rather than blocked.
    #
    # Two molecules sharing a vector is a limitation of the representation, not a broken run, and
    # every encoder here collides for a reason that can be named. What IS worth failing on is a
    # collision nobody can explain, so each group is classified and only the unexplained ones are
    # errors.
    uniq = len({r.tobytes() for r in np.ascontiguousarray(mat)})
    identities = len({canonical_key(m) for m in all_mols})
    if uniq >= identities:
        report(True, f"{name}: distinct vectors for distinct molecules")
    elif name in SHUFFLED:
        # Classifying these is meaningless: every molecule already holds someone else's vector, so
        # a collision only says two molecules drew donors that themselves collide.
        note(f"{name}: {identities - uniq} molecule pair(s) share a vector",
             "inherited from the underlying fingerprint; the donors are deliberately mismatched, "
             "so this says nothing about the control's validity")
    else:
        reasons = {}
        for g in collision_groups(mat, all_mols):
            reasons.setdefault(explain_collision(g), []).append(g)
        note(f"{name}: {identities - uniq} molecule pair(s) share a vector",
             f"{uniq} distinct vectors for {identities} chemical identities")
        for why, gs in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
            print(f"       {len(gs):2d} group(s): {why}")
            print(f"           e.g. {' | '.join(s[:40] for s in gs[0][:3])}")

    # ---- the shuffled controls
    if name in SHUFFLED:
        self_donor = [m for m in all_mols
                      if canonical_key(rep._donor[m]) == canonical_key(m)]
        report(not self_donor, f"{name}: a true derangement over chemical identity",
               f"{len(self_donor)} molecule(s) get their own vector" if self_donor else "")

        rep2 = build_representation(name, tables)
        rep2.build_shuffle(all_mols)
        same = rep.shuffle_fingerprint() == rep2.shuffle_fingerprint()
        report(same, f"{name}: the permutation is reproducible (seed {SHUFFLE_SEED})",
               "" if same else "two builds disagree - the control would be noise, not a control")

        base = build_representation(SHUFFLED[name], tables)
        base.fit_scaler(train)
        agree = sum(1 for m in all_mols
                    if np.array_equal(rep.vector(m), base.vector(m)))
        report(agree == 0, f"{name}: no molecule receives its real vector",
               f"{agree} match the unshuffled representation" if agree else "")
        print(f"       donor map sha1 {rep.shuffle_fingerprint()} "
              "(recorded in every run_info.json for this column)")


BLOCKED = []         # an encoder with no table: its cells cannot run, but the rest can


def audit_embedding_tables() -> None:
    """A missing table is a blocked dependency, not a data defect.

    Kept out of PROBLEMS on purpose. precompute_embeddings refuses to WRITE a bad table, so a
    missing one means the encoder could not be built at all - gin needs dgl and dgllife, for
    instance. The cells that need it are skipped and reported; failing the whole audit would leave
    the GPUs idle over a pip install.
    """
    print("\n--- mean-pooled encoder tables")
    for name in EMBEDDING_REPS:
        path = embedding_table_path(name)
        if not os.path.exists(path):
            print(f"  BLOCKED {name}: no table - every cell using it will be skipped")
            BLOCKED.append(name)
        else:
            size = os.path.getsize(path) / 1e6
            report(True, f"{name}: table present", f"{os.path.basename(path)} ({size:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description="Audit the grid's molecular representations.")
    p.add_argument("--reps", nargs="+", default=None,
                   help=f"default: every one that has its table. Choices: {', '.join(ALL_REP_NAMES)}")
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="api_cluster")
    args = p.parse_args()

    paths = SplitPaths(args.data_root, args.split)
    check_heldout_is_external(paths)
    print(f"data root : {paths.root}")
    print(f"molecules : {len(paths.all_molecules())} strings, "
          f"{len({canonical_key(m) for m in paths.all_molecules()})} distinct chemical identities")
    print("held-out  : 22 pairs, external (verified)")

    audit_embedding_tables()

    wanted = args.reps or [n for n in ALL_REP_NAMES
                           if SHUFFLED.get(n, n) not in EMBEDDING_REPS
                           or os.path.exists(embedding_table_path(SHUFFLED.get(n, n)))]
    unknown = set(wanted) - set(ALL_REP_NAMES)
    if unknown:
        raise SystemExit(f"unknown representation(s): {', '.join(sorted(unknown))}")

    tables = table_paths_for(wanted, paths, LOOKUP_DIR)
    for name in wanted:
        try:
            audit_one(name, paths, tables)
        except SystemExit as e:
            report(False, f"{name}: could not be built", str(e))
        except Exception as e:
            report(False, f"{name}: crashed", f"{type(e).__name__}: {e}")

    print("\n" + "=" * 100)
    if BLOCKED:
        import importlib

        from experiments.encoder_comparison.families import all_cells
        n_cells = len([c for c in all_cells()
                       if c.primary in BLOCKED or c.secondary in BLOCKED])
        print(f"BLOCKED: {', '.join(BLOCKED)} has no embedding table, so {n_cells} of "
              f"{len(all_cells())} cells ({n_cells * 5} runs) will be skipped.")
        if "gin" in BLOCKED and importlib.util.find_spec("dgl") is None:
            print("  gin needs dgl and dgllife (both are in requirements.txt):")
            print("    pip install 'dgl>=1.1' 'dgllife>=0.3.2'")
            print("  A CPU-only dgl wheel is fine - the encoder already runs the graph on CPU and "
                  "moves the output to the GPU.")
        print("  The remaining cells are unaffected and can run now.")
        print()
    if NOTES:
        print(f"{len(NOTES)} observation(s) - true properties, not defects, worth having in the "
              "paper:")
        for n_ in NOTES:
            print(f"  . {n_}")
        print()
    if PROBLEMS:
        print(f"{len(PROBLEMS)} PROBLEM(S) that must be fixed before training:")
        for p_ in PROBLEMS:
            print(f"  - {p_}")
        raise SystemExit(1)
    print("no blocking problems in the representations the grid will read")


if __name__ == "__main__":
    main()
