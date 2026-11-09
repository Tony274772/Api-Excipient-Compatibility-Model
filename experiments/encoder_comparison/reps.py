"""Molecular representations for the 9x9 grid - one vector per molecule.

A *representation* here is anything that can be handed to a tower's primary (64-wide) or
secondary (32-wide) block as a single fixed-length vector. That covers:

  * the 2 fingerprints computed from SMILES on the fly (morgan, maccs),
  * the 2 fingerprints that come from a precomputed table (pubchemfp, mol2vec),
  * the 5 frozen encoders *averaged over their tokens / atoms* (molformer, chemberta, gin, gat,
    dmpnn), written once by precompute_embeddings.py and read back from a table here,

and the two capacity controls built on top of them:

  * shuf_maccs / shuf_morgan - real vectors handed to the WRONG molecule under a fixed
    derangement, so the parameter count is identical and the information content is zero.

Scaling rule (Part 6, problem 3 of paper/encoder_grid_plan.md): a bit vector is left raw, a
continuous vector is z-scored with TRAIN-ONLY statistics. Mol2vec runs about -0.35..+0.50 and the
encoder outputs are unbounded, so feeding them unscaled into a 32-wide block next to 0/1 bits is
the bug that was already fixed once in this project for raw `organic_heavy_atoms`.

Nothing here trains. Every vector is deterministic given the molecule.
"""

import hashlib
import os

import numpy as np
import pandas as pd

from src.features import maccs_bits, morgan_bits

# --------------------------------------------------------------------------- the registry
# kind: "bits" -> left raw; "continuous" -> z-scored with train-only mean/std.
# source: "smiles" -> computed per molecule; "table" -> read from a lookup CSV.
REP_SPECS = {
    "morgan":    {"dim": 1024, "kind": "bits",       "source": "smiles",
                  "label": "Morgan fingerprint, radius 2, 1024 bits"},
    "maccs":     {"dim": 167,  "kind": "bits",       "source": "smiles",
                  "label": "MACCS keys, 167 bits"},
    "pubchemfp": {"dim": 881,  "kind": "bits",       "source": "table",
                  "label": "PubChem fingerprint, 881 bits"},
    "mol2vec":   {"dim": 300,  "kind": "continuous", "source": "table",
                  "label": "Mol2vec embedding, 300-d"},
    "molformer": {"dim": 768,  "kind": "continuous", "source": "table",
                  "label": "MoLFormer-XL mean-pooled embedding, 768-d (frozen)"},
    "chemberta": {"dim": 384,  "kind": "continuous", "source": "table",
                  "label": "ChemBERTa mean-pooled embedding, 384-d (frozen)"},
    "gin":       {"dim": 300,  "kind": "continuous", "source": "table",
                  "label": "Pretrained GIN mean-pooled embedding, 300-d (frozen)"},
    "gat":       {"dim": 300,  "kind": "continuous", "source": "table",
                  "label": "Pretrained GAT mean-pooled embedding, 300-d (frozen)"},
    "dmpnn":     {"dim": 2048, "kind": "continuous", "source": "table",
                  "label": "D-MPNN (CheMeleon) mean-pooled embedding, 2048-d (frozen)"},
}

# The two shuffled capacity controls. They bracket the Linear(d,32) parameter range: 5,376
# parameters at 167 bits and 32,800 at 1024 bits, a 6x spread, so one control cannot stand in for
# the other.
SHUFFLED = {
    "shuf_maccs":  "maccs",
    "shuf_morgan": "morgan",
}

# Encoders whose averaged embedding needs precompute_embeddings.py to have been run.
EMBEDDING_REPS = ["molformer", "chemberta", "gin", "gat", "dmpnn"]

# Fingerprints that come from a shipped table (built by data.build_vector_lookup).
FINGERPRINT_TABLE_REPS = ["pubchemfp", "mol2vec"]

ALL_REP_NAMES = list(REP_SPECS) + list(SHUFFLED)

# The permutation seed is deliberately NOT the training seed: the control has to be the SAME
# control in every run, otherwise it is fresh noise each time and tests something else.
SHUFFLE_SEED = 20251110


def rep_dim(name: str) -> int:
    """Vector width of a representation, shuffled controls included."""
    base = SHUFFLED.get(name, name)
    if base not in REP_SPECS:
        raise SystemExit(f"unknown representation '{name}'. Choices: {', '.join(ALL_REP_NAMES)}")
    return REP_SPECS[base]["dim"]


def rep_kind(name: str) -> str:
    """'bits' (left raw) or 'continuous' (z-scored on train statistics)."""
    base = SHUFFLED.get(name, name)
    return REP_SPECS[base]["kind"]


def rep_label(name: str) -> str:
    if name in SHUFFLED:
        base = SHUFFLED[name]
        return (f"{REP_SPECS[base]['label']} - SHUFFLED control "
                "(fixed derangement, same parameter count, zero information)")
    return REP_SPECS[name]["label"]


def secondary_params(name: str) -> int:
    """Trainable parameters this representation adds in the 32-wide secondary block."""
    return rep_dim(name) * 32 + 32


def primary_params(name: str) -> int:
    """Trainable parameters this representation adds in the 64-wide primary block."""
    return rep_dim(name) * 64 + 64


# --------------------------------------------------------------------------- tables
def read_table(path: str) -> dict:
    """Lookup CSV (cid, smiles, f0..fN) -> {smiles: vector}. Keyed on OUR SMILES spelling."""
    if not os.path.exists(path):
        raise SystemExit(
            f"lookup table not found: {path}\n"
            "Embedding tables are written by:\n"
            "  python -m experiments.encoder_comparison.precompute_embeddings"
        )
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("cid", "smiles")]
    vals = df[feature_cols].to_numpy(dtype=np.float32)
    return {str(s): vals[i] for i, s in enumerate(df["smiles"].tolist())}


def canonical_key(smiles: str) -> str:
    """One key per chemical identity, so two spellings of a molecule never count as two molecules.

    RDKit's canonical SMILES where the molecule parses; the stripped string otherwise (a few
    entries are bare metals and salts RDKit may reject, and those still need a stable key).
    """
    from rdkit import Chem
    s = str(smiles).strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return s
    return Chem.MolToSmiles(mol)


def derangement(n: int, seed: int) -> np.ndarray:
    """A permutation with no fixed point, so no molecule can receive its own vector."""
    if n < 2:
        raise SystemExit("need at least 2 molecules to build a derangement")
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if not (perm == np.arange(n)).any():
            return perm


# --------------------------------------------------------------------------- representations
class Representation:
    """One molecule -> one fixed-length float32 vector.

    `vector(smiles)` is pure: the same SMILES always gives the same vector, in every epoch, every
    seed and every split. `fit_scaler` is called once with the training molecules only.
    """

    def __init__(self, name: str, table: dict = None):
        self.name = name
        self.base = SHUFFLED.get(name, name)
        self.dim = rep_dim(name)
        self.kind = rep_kind(name)
        self.label = rep_label(name)
        self._table = table
        self._mean = None
        self._std = None
        self._donor = {}                      # shuffled controls only
        self._zero = np.zeros(self.dim, dtype=np.float32)

    # ---- raw vector, before scaling
    def _raw(self, smiles: str) -> np.ndarray:
        if self.base == "morgan":
            return np.asarray(morgan_bits(smiles, n_bits=1024), dtype=np.float32)
        if self.base == "maccs":
            return np.asarray(maccs_bits(smiles), dtype=np.float32)
        v = self._table.get(str(smiles))
        if v is None:
            raise RuntimeError(f"{self.name}: no precomputed vector for SMILES {smiles!r}")
        return np.asarray(v, dtype=np.float32)

    # ---- the fixed derangement, for the two control columns
    def build_shuffle(self, molecules) -> None:
        """Assign every molecule a different molecule's vector, once and for all.

        The permutation covers train + validation + held-out together, so a molecule's donor never
        changes between splits, and is seeded by SHUFFLE_SEED rather than the run seed, so all 5
        seeds of a control cell see the identical mapping.

        The derangement runs over CHEMICAL IDENTITY, not over SMILES strings. The dataset spells
        some molecules two ways - a leading or trailing space, or a stereochemistry variant under
        the same CID - so a string-level derangement can hand a molecule back its own vector
        through its other spelling, which would quietly put real information into a control that
        is supposed to carry none.
        """
        if self.name not in SHUFFLED:
            return
        mols = sorted(set(str(m) for m in molecules if str(m)))

        groups = {}                               # canonical key -> [smiles, ...]
        key_of = {}
        for m in mols:
            k = canonical_key(m)
            key_of[m] = k
            groups.setdefault(k, []).append(m)

        keys = sorted(groups)
        perm = derangement(len(keys), SHUFFLE_SEED)
        donor_key = {keys[i]: keys[perm[i]] for i in range(len(keys))}

        # Each molecule takes the first spelling of its donor group, so the mapping is stable.
        self._donor = {m: groups[donor_key[key_of[m]]][0] for m in mols}

        same = [m for m in mols if canonical_key(self._donor[m]) == key_of[m]]
        if same:
            raise RuntimeError(f"{self.name}: {len(same)} molecule(s) would receive their own "
                               f"vector, e.g. {same[:3]}")

    def shuffle_fingerprint(self) -> str:
        """Short hash of the donor map, written into run_info so two runs can be compared."""
        if not self._donor:
            return ""
        joined = "\n".join(f"{k}\t{v}" for k, v in sorted(self._donor.items()))
        return hashlib.sha1(joined.encode()).hexdigest()[:12]

    # ---- train-only scaling
    def fit_scaler(self, train_molecules) -> None:
        """Mean / std over TRAIN molecules only. No-op for bit vectors."""
        if self.kind != "continuous":
            return
        mols = sorted(set(str(m) for m in train_molecules if str(m)))
        mat = np.stack([self._raw(m) for m in mols])
        self._mean = mat.mean(axis=0).astype(np.float32)
        std = mat.std(axis=0).astype(np.float32)
        # A constant column carries no information; 1.0 leaves it at exactly 0 after centring
        # instead of dividing by ~0 and exploding.
        self._std = np.where(std < 1e-6, np.float32(1.0), std).astype(np.float32)

    def save_scaler(self, path: str) -> None:
        if self.kind == "continuous" and self._mean is not None:
            np.savez(path, mean=self._mean, std=self._std)

    def load_scaler(self, path: str) -> None:
        if self.kind == "continuous":
            d = np.load(path)
            self._mean, self._std = d["mean"], d["std"]

    @property
    def scaled(self) -> bool:
        return self.kind == "continuous"

    # ---- the vector the tower sees
    def vector(self, smiles: str) -> np.ndarray:
        smiles = str(smiles)
        if not smiles:
            return self._zero.copy()
        if self.name in SHUFFLED:
            donor = self._donor.get(smiles)
            if donor is None:
                raise RuntimeError(
                    f"{self.name}: molecule {smiles!r} is not in the shuffle table; "
                    "build_shuffle() must see every molecule in every split")
            smiles = donor
        v = self._raw(smiles)
        if self.kind == "continuous":
            if self._mean is None:
                raise RuntimeError(f"{self.name}: fit_scaler() must be called before vector()")
            v = (v - self._mean) / self._std
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_representation(name: str, table_paths: dict) -> Representation:
    """Construct one representation, loading its lookup table if it needs one."""
    base = SHUFFLED.get(name, name)
    if base not in REP_SPECS:
        raise SystemExit(f"unknown representation '{name}'. Choices: {', '.join(ALL_REP_NAMES)}")
    table = None
    if REP_SPECS[base]["source"] == "table":
        path = table_paths.get(base)
        if path is None:
            raise SystemExit(
                f"representation '{name}' needs a lookup table for '{base}' but none was given")
        table = read_table(path)
        got = len(next(iter(table.values())))
        want = REP_SPECS[base]["dim"]
        if got != want:
            raise SystemExit(f"{base}: lookup table has {got} features, expected {want} ({path})")
    return Representation(name, table)
