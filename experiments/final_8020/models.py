"""
Models trained on the final_project 80/20 split.

BASELINE_MODELS     the 54 CompatibilityModel runs in metrics/ (recipe of main.py; per-model settings
                    read from each run's own checkpoint, see experiments/split_8020/models.py).
PGB_MODELS          the 9 Prior-Gated Bilinear families in metrics/pgb_* (recipe of experiments/pgb).
PGB_GLOBAL_GATED    PGB ChemBERTa and D-MPNN with global_gated_attention pooling instead of cls
                    (experiments/pgb/global_gated).
GB_MODELS           the 9 Gated Bilinear families (experiments/gated_bilinear).

Names of the baseline and PGB models are the metrics/ folder names, so results line up with metrics/.

Concat models with a token encoder: src.model.CompatibilityModel ignores `pooling` under concat fusion, so in
metrics/ the four *_concat_global_gated_* runs were the same network as their *_concat_cls_* twin. Here the
global-gated ones use ConcatGatedPoolingModel (gated attention pooling over atom tokens, concat_pooling.py)
and the cls ones keep the encoder readout, so all 54 baseline models are different networks.
"""

from experiments.split_8020.models import BASELINE_MODELS
from src.config import Config

# concat fusion + token encoder, global gated pooling -> ConcatGatedPoolingModel
CONCAT_GATED_POOLING = {
    "dmpnn_chemprop_concat_global_gated_asl",
    "dmpnn_chemprop_concat_global_gated_bce",
    "pretrained_gat_concat_global_gated_asl",
    "pretrained_gat_concat_global_gated_bce",
}
# encoders whose encode() returns (tokens, readout, mask); every other encoder is forced to concat fusion
SEQUENCE_ENCODERS = {"molformer", "chemberta", "pretrained_gin", "pretrained_gat", "dmpnn_chemprop"}

# metrics/ folder name -> experiments/pgb family
PGB_MODELS = {
    "pgb_fixed_vector_maccs_concat_asl": "maccs",
    "pgb_fixed_vector_morgan_concat_asl": "morgan",
    "pgb_fixed_vector_pubchemfp_concat_asl": "pubchemfp",
    "pgb_fixed_vector_mol2vec_concat_asl": "mol2vec",
    "pgb_molformer_cross_attn_global_gated_asl": "molformer",
    "pgb_chemberta_cross_attn_cls_asl": "chemberta",
    "pgb_pretrained_gin_cross_attn_global_gated_asl": "gin",
    "pgb_pretrained_gat_cross_attn_global_gated_asl": "gat",
    "pgb_dmpnn_chemprop_cross_attn_cls_asl": "dmpnn",
}

# name -> experiments/pgb/global_gated family
PGB_GLOBAL_GATED_MODELS = {
    "pgb_chemberta_cross_attn_global_gated_asl": "chemberta",
    "pgb_dmpnn_chemprop_cross_attn_global_gated_asl": "dmpnn",
}

# name -> experiments/gated_bilinear family
GB_MODELS = {f"gb_{fam}": fam for fam in
             ["maccs", "morgan", "pubchem", "mol2vec", "molformer", "chemberta", "gin", "gat", "dmpnn"]}

GROUPS = {
    "baseline": list(BASELINE_MODELS),
    "pgb": list(PGB_MODELS),
    "pgb_global_gated": list(PGB_GLOBAL_GATED_MODELS),
    "gb": list(GB_MODELS),
}
ALL_MODELS = [m for names in GROUPS.values() for m in names]


def kind_of(name: str) -> str:
    for group, names in GROUPS.items():
        if name in names:
            return group
    raise KeyError(name)


def baseline_architecture(name: str) -> dict:
    """Effective network of a baseline model: what actually gets built, not just the config fields."""
    config = Config()
    overrides = BASELINE_MODELS[name]
    for k, v in overrides.items():
        setattr(config, k, v)
    if config.encoder not in SEQUENCE_ENCODERS:
        fusion, pooling = "concat", "none (fixed vector)"
    elif config.fusion == "concat":
        fusion = "concat"
        pooling = "gated_attention_over_tokens" if name in CONCAT_GATED_POOLING else "encoder_readout"
    else:
        fusion, pooling = "cross_attn", config.pooling
    if name in CONCAT_GATED_POOLING and (config.fusion != "concat" or config.pooling != "global_gated_attention"):
        raise ValueError(f"{name}: CONCAT_GATED_POOLING needs concat fusion with global_gated_attention pooling")
    arch = {k: v for k, v in overrides.items() if k not in ("fusion", "pooling")}
    arch.update(fusion=fusion, pooling=pooling,
                model_class="ConcatGatedPoolingModel" if name in CONCAT_GATED_POOLING else "CompatibilityModel")
    return arch


def assert_no_duplicate_models():
    """Two baseline names must never build the same network with the same recipe."""
    seen = {}
    for name in BASELINE_MODELS:
        key = tuple(sorted((k, str(v)) for k, v in baseline_architecture(name).items()))
        if key in seen:
            raise ValueError(f"{name} and {seen[key]} are the same model")
        seen[key] = name
