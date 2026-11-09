"""The 9 encoder families, the 12 secondary columns, and the 101 grid cells left to train.

Read paper/encoder_grid_plan.md first. The short version:

  * One model = a PRIMARY molecular description (64-wide block) + an optional SECONDARY
    description (32-wide block) + the fixed tower + the fixed PGB head.
  * Rows are the 9 primaries, columns are the 12 secondaries. The primary and secondary slots
    are different widths, so `morgan+maccs` and `maccs+morgan` are different models and both run.
  * The `none` column is already trained (45 runs, 9 families x 5 seeds) and is reused as it is.
  * Everything that is not a molecular description is identical in every cell: 18 chemistry flags,
    12 PGB descriptors, 8-value salt context per molecule, 5-value excipient prior and 5 mechanism
    flags per pair.

Three control blocks are not columns:
  * the floor      - no molecular description at all, so we can say what any description is worth,
  * pooling        - each deep method with the OTHER pooling, because the current assignment was
                     picked on a single seed under a different recipe,
  * no-cross-attn  - each deep method as a precomputed mean-pooled vector in the PRIMARY slot, so
                     all 9 methods can be compared with the fusion held equal. Note that setting
                     fusion="concat" does NOT do this: src/model.py:_encode_struct returns
                     (None, None) when use_cross_attn is false and the encoder output is dropped
                     entirely, so the control has to run through the fixed-vector path.
"""

from experiments.encoder_comparison.reps import (SHUFFLED, primary_params,  # noqa: F401
                                                 rep_dim, rep_label, secondary_params)

FAMILIES = {
    # ---- fingerprint / fixed-vector families: concat fusion, no structural branch ----
    "morgan": {
        "kind": "fixed_vector",
        "encoder": "fixed_vector",
        "source": "morgan",
        "primary_dim": 1024,
        "fusion": "concat",
        "pooling": None,
        "needs_lookup": False,
        "label": "Morgan fingerprint, radius 2, 1024 bits (salt-aware OR-fold)",
    },
    "maccs": {
        "kind": "fixed_vector",
        "encoder": "fixed_vector",
        "source": "maccs",
        "primary_dim": 167,
        "fusion": "concat",
        "pooling": None,
        "needs_lookup": False,
        "label": "MACCS keys, 167 bits (salt-aware OR-fold)",
    },
    "pubchemfp": {
        "kind": "fixed_vector",
        "encoder": "fixed_vector",
        "source": "pubchemfp",
        "primary_dim": 881,
        "fusion": "concat",
        "pooling": None,
        "needs_lookup": True,
        "label": "PubChem fingerprint, 881 bits (precomputed lookup)",
    },
    "mol2vec": {
        "kind": "fixed_vector",
        "encoder": "fixed_vector",
        "source": "mol2vec",
        "primary_dim": 300,
        "fusion": "concat",
        "pooling": None,
        "needs_lookup": True,
        "label": "Mol2vec embedding, 300 dimensions (precomputed lookup)",
    },
    # ---- sequence / graph families: frozen pretrained encoder + cross-attention ----
    "molformer": {
        "kind": "seq",
        "encoder": "molformer",
        "source": None,
        "primary_dim": 0,
        "fusion": "cross_attn",
        "pooling": "global_gated_attention",
        "needs_lookup": False,
        "label": "MoLFormer-XL embedding (frozen) + cross-attention, gated-attention pooling",
    },
    "chemberta": {
        "kind": "seq",
        "encoder": "chemberta",
        "source": None,
        "primary_dim": 0,
        "fusion": "cross_attn",
        "pooling": "cls",
        "needs_lookup": False,
        "label": "ChemBERTa embedding (frozen) + cross-attention, CLS pooling",
    },
    "gin": {
        "kind": "seq",
        "encoder": "pretrained_gin",
        "source": None,
        "primary_dim": 0,
        "fusion": "cross_attn",
        "pooling": "global_gated_attention",
        "needs_lookup": False,
        "label": "Pretrained GIN (context-pred, frozen) + cross-attention, gated-attention pooling",
    },
    "gat": {
        "kind": "seq",
        "encoder": "pretrained_gat",
        "source": None,
        "primary_dim": 0,
        "fusion": "cross_attn",
        "pooling": "global_gated_attention",
        "needs_lookup": False,
        "label": "Pretrained GAT (Stanford context-pred, frozen) + cross-attention, gated-attention pooling",
    },
    "dmpnn": {
        "kind": "seq",
        "encoder": "dmpnn_chemprop",
        "source": None,
        "primary_dim": 0,
        "fusion": "cross_attn",
        "pooling": "cls",
        "needs_lookup": False,
        "label": "D-MPNN (CheMeleon, frozen) + cross-attention, CLS pooling",
    },
}

ORDER = ["morgan", "maccs", "pubchemfp", "mol2vec", "molformer", "chemberta", "gin", "gat", "dmpnn"]

FINGERPRINTS = ["morgan", "maccs", "pubchemfp", "mol2vec"]
DEEP = ["molformer", "chemberta", "gin", "gat", "dmpnn"]

# Which of the 5 deep methods are transformers and which are graph networks. Used by test 9
# (two-encoder models split into cross-architecture and same-architecture).
ARCHITECTURE = {"molformer": "transformer", "chemberta": "transformer",
                "gin": "graph", "gat": "graph", "dmpnn": "graph"}

# The 12 secondary columns, in table order. `None` is the already-trained column.
SECONDARY_COLUMNS = [None] + ORDER + ["shuf_maccs", "shuf_morgan"]

# Pooling currently used by each deep method, and the alternative the control trains.
# The current assignment came from single-seed full_grid runs under an older recipe
# (CLS won chemberta 0.763 vs 0.740 and dmpnn 0.732 vs 0.691; gated won gat 0.686 vs 0.634),
# which is exactly the kind of one-seed choice this grid exists to replace.
POOLINGS = ["global_gated_attention", "cls"]


def other_pooling(family: str) -> str:
    current = FAMILIES[family]["pooling"]
    return POOLINGS[1] if current == POOLINGS[0] else POOLINGS[0]


def spec(name: str) -> dict:
    if name not in FAMILIES:
        raise SystemExit(f"unknown family '{name}'. Choices: {', '.join(ORDER)}")
    return FAMILIES[name]


# --------------------------------------------------------------------------- cells
class Cell:
    """One point of the grid: a primary, a secondary, a pooling, and a block label.

    `name` is the folder name under experiments/results/encoder_grid/ and must be stable, because
    the resume logic keys on it. Seeds are appended by run_name().
    """

    def __init__(self, primary, secondary=None, pooling=None, block="pair",
                 no_cross_attn=False, note=""):
        self.primary = primary                  # a family name, or None for the floor
        self.secondary = secondary              # a representation name, or None
        self.block = block
        self.no_cross_attn = no_cross_attn
        self.note = note
        if primary is None:
            self.pooling = None
        elif pooling is not None:
            self.pooling = pooling
        else:
            self.pooling = FAMILIES[primary]["pooling"]

    @property
    def name(self) -> str:
        if self.primary is None:
            return "floor"
        parts = [self.primary]
        if self.no_cross_attn:
            parts.append("noca")
        if self.secondary:
            parts.append(f"x_{self.secondary}")
        if self.block == "pooling":
            parts.append(f"pool_{'gated' if self.pooling == 'global_gated_attention' else 'cls'}")
        return "__".join(parts)

    @property
    def kind(self) -> str:
        """'floor', 'fixed_vector' (no encoder is run) or 'seq' (cross-attention path)."""
        if self.primary is None:
            return "floor"
        if self.no_cross_attn:
            return "fixed_vector"
        return FAMILIES[self.primary]["kind"]

    def label(self) -> str:
        if self.primary is None:
            return ("FLOOR: no molecular description at all - descriptors, flags, salt, "
                    "excipient prior and mechanism flags only")
        if self.no_cross_attn:
            base = (f"{self.primary} mean-pooled embedding as a fixed primary vector, "
                    "NO cross-attention")
        else:
            base = FAMILIES[self.primary]["label"]
        if self.block == "pooling":
            base = f"{self.primary} + cross-attention, {self.pooling} pooling (pooling control)"
        if self.secondary:
            base = f"{base}  +  secondary: {rep_label(self.secondary)}"
        return base

    def representations(self) -> list:
        """Representation names this cell needs tables / shuffles for."""
        names = []
        if self.primary is not None:
            if self.no_cross_attn:
                names.append(self.primary)
            elif FAMILIES[self.primary]["kind"] == "fixed_vector":
                names.append(self.primary)
        if self.secondary:
            names.append(self.secondary)
        return names

    def extra_block_params(self) -> int:
        """Trainable parameters in the primary + secondary blocks, for the results table."""
        n = 0
        if self.primary is not None:
            if self.no_cross_attn:
                n += primary_params(self.primary)
            elif FAMILIES[self.primary]["kind"] == "fixed_vector":
                n += primary_params(self.primary)
        if self.secondary:
            n += secondary_params(self.secondary)
        return n

    def __repr__(self):
        return f"Cell({self.name})"


# --------------------------------------------------------------------------- the grid
def done_cells() -> list:
    """The 9 cells that are already trained: the `none` column, 5 seeds each.

    These live in experiments/results/encoder_comparison/<family>_seed<N>/ and are NOT retrained.
    """
    return [Cell(f, None, block="none") for f in ORDER]


def floor_cells() -> list:
    """Block 1a - 1 cell. No molecular description at all."""
    return [Cell(None, None, block="floor",
                 note="is any molecular description worth anything?")]


def pooling_cells() -> list:
    """Block 1b - 5 cells. Each deep method with the OTHER pooling.

    The five combinations that already exist are each method's current pooling (the `none`
    column), so only the alternative is new. Whatever wins per method is then used for that
    method everywhere else.
    """
    return [Cell(f, None, pooling=other_pooling(f), block="pooling",
                 note=f"{f}: {other_pooling(f)} instead of {FAMILIES[f]['pooling']}")
            for f in DEEP]


def shuffled_cells() -> list:
    """Block 2a - 18 cells. Every primary against both shuffled controls."""
    return [Cell(p, s, block="shuffled",
                 note="capacity control: same parameters, zero information")
            for p in ORDER for s in ("shuf_maccs", "shuf_morgan")]


def no_cross_attn_cells() -> list:
    """Block 2b - 5 cells. Each deep method with cross-attention removed.

    The encoder's mean-pooled vector goes into the PRIMARY slot through the fixed-vector path.
    This is the only way to keep the encoder's output: with fusion="concat" the model discards it.
    """
    return [Cell(f, None, block="no_cross_attn",
                 no_cross_attn=True,
                 note="is the encoder good, or is cross-attention good?")
            for f in DEEP]


def fingerprint_secondary_cells() -> list:
    """Block 3 - 32 cells. 9 primaries x 4 fingerprint secondaries, diagonal skipped."""
    return [Cell(p, s, block="fp_secondary") for p in ORDER for s in FINGERPRINTS if s != p]


def embedding_secondary_cells() -> list:
    """Block 4 - 40 cells. 9 primaries x 5 encoder secondaries, diagonal skipped.

    20 of these are fingerprint-primary + encoder-secondary; the other 20 put a cross-attended
    encoder together with a *different* encoder's mean-pooled output - two pretrained encoders in
    one model.
    """
    return [Cell(p, s, block="emb_secondary") for p in ORDER for s in DEEP if s != p]


BLOCKS = [
    ("1  floor",          floor_cells),
    ("1  pooling",        pooling_cells),
    ("2  shuffled",       shuffled_cells),
    ("2  no_cross_attn",  no_cross_attn_cells),
    ("3  fp_secondary",   fingerprint_secondary_cells),
    ("4  emb_secondary",  embedding_secondary_cells),
]

# Training order, and the order the plan's Part 8 prescribes: pooling must be settled before any
# deep row runs, and the controls decide how every later number is read.
BLOCK_ORDER = ["floor", "pooling", "shuffled", "no_cross_attn", "fp_secondary", "emb_secondary"]


def all_cells(blocks=None) -> list:
    """The 101 cells left to train, in training order. `blocks` filters by block name."""
    wanted = set(blocks) if blocks else None
    cells = []
    for _, fn in BLOCKS:
        for c in fn():
            if wanted is None or c.block in wanted:
                cells.append(c)
    return cells


def cell_by_name(name: str) -> Cell:
    for c in all_cells():
        if c.name == name:
            return c
    known = ", ".join(c.name for c in all_cells()[:6])
    raise SystemExit(f"unknown cell '{name}'. There are {len(all_cells())}; first few: {known} ...")


def grid_summary() -> str:
    lines = ["cells left to train, by block:"]
    total = 0
    for label, fn in BLOCKS:
        n = len(fn())
        total += n
        lines.append(f"  {label:18s} {n:4d} cells  {n * 5:5d} runs")
    lines.append(f"  {'TOTAL':18s} {total:4d} cells  {total * 5:5d} runs")
    lines.append(f"  already done        {len(done_cells()):4d} cells  "
                 f"{len(done_cells()) * 5:5d} runs  (the `none` column, reused as it is)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(grid_summary())
    print()
    for label, fn in BLOCKS:
        cells = fn()
        print(f"--- {label} ({len(cells)} cells)")
        for c in cells:
            print(f"    {c.name:34s} {c.label()[:96]}")
