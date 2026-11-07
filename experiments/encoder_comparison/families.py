"""The 9 encoder families, each with exactly ONE molecular representation.

This is the difference from the earlier runs. Previously every family also received a
secondary fingerprint, and which one it got varied by family (Morgan got MACCS, MACCS got
Morgan, the deep encoders got MACCS, ChemBERTa got Morgan). "Encoder" and "fingerprint" were
therefore confounded. Here:

  * a fingerprint family gets its own fingerprint and nothing else,
  * a sequence / graph family gets its own learned embedding and nothing else.

Everything that is NOT a molecular representation is identical for all 9 families and is part
of the fixed PGB design: 18 chemistry flags, 12 PGB descriptors, 8-value salt context per
molecule, and the 5-value excipient prior plus 5 mechanism flags per pair.
"""

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


def spec(name: str) -> dict:
    if name not in FAMILIES:
        raise SystemExit(f"unknown family '{name}'. Choices: {', '.join(ORDER)}")
    return FAMILIES[name]
