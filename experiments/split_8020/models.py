"""
Models trained on the 80/20 split.

BASELINE_MODELS: the 54 CompatibilityModel runs in metrics/ (trained with main.py on the
60/20/20 split). Each entry holds the settings that differ from src.config.Config defaults,
read from the run's own checkpoint (checkpoints/<name>/best_model.pt). Pooling is the
pooling the checkpoint actually used: six older cross-attention checkpoints have no
api_pool weights, i.e. they ran CLS pooling although their saved config says
global_gated_attention.

PGB_MODELS: the 9 Prior-Gated Bilinear families (experiments/pgb), trained with the
PGB recipe from experiments/pgb/config.py.
"""

BASELINE_MODELS = {
    "chemberta_concat_asl": {"encoder": "chemberta", "fusion": "concat"},
    "chemberta_cross_attn_asl": {"encoder": "chemberta", "pooling": "cls"},
    "chemberta_cross_attn_cls_bce": {"encoder": "chemberta", "pooling": "cls", "loss": "bce"},
    "chemberta_cross_attn_cls_focal": {"encoder": "chemberta", "pooling": "cls", "loss": "focal"},
    "chemberta_cross_attn_cls_weighted_bce": {"encoder": "chemberta", "pooling": "cls", "loss": "weighted_bce"},
    "chemberta_cross_attn_global_gated_asl": {"encoder": "chemberta"},
    "chemberta_cross_attn_global_gated_bce": {"encoder": "chemberta", "loss": "bce"},
    "chemberta_cross_attn_global_gated_focal": {"encoder": "chemberta", "loss": "focal"},
    "chemberta_cross_attn_global_gated_weighted_bce": {"encoder": "chemberta", "loss": "weighted_bce"},
    "chemberta_cross_attn_pairwise_asl": {"encoder": "chemberta", "pooling": "explicit_pairwise"},
    "chemberta_cross_attn_pairwise_bce": {"encoder": "chemberta", "pooling": "explicit_pairwise", "loss": "bce"},
    "dmpnn_chemprop_concat_cls_asl": {"encoder": "dmpnn_chemprop", "fusion": "concat", "pooling": "cls"},
    "dmpnn_chemprop_concat_cls_bce": {"encoder": "dmpnn_chemprop", "fusion": "concat", "pooling": "cls", "loss": "bce"},
    "dmpnn_chemprop_concat_global_gated_asl": {"encoder": "dmpnn_chemprop", "fusion": "concat"},
    "dmpnn_chemprop_concat_global_gated_bce": {"encoder": "dmpnn_chemprop", "fusion": "concat", "loss": "bce"},
    "dmpnn_chemprop_cross_attn_cls_asl": {"encoder": "dmpnn_chemprop", "pooling": "cls"},
    "dmpnn_chemprop_cross_attn_cls_bce": {"encoder": "dmpnn_chemprop", "pooling": "cls", "loss": "bce"},
    "dmpnn_chemprop_cross_attn_global_gated_asl": {"encoder": "dmpnn_chemprop"},
    "dmpnn_chemprop_cross_attn_global_gated_bce": {"encoder": "dmpnn_chemprop", "loss": "bce"},
    "fixed_vector_maccs_concat_asl": {"encoder": "fixed_vector", "fixed_vector_source": "maccs", "fusion": "concat"},
    "fixed_vector_maccs_concat_bce": {"encoder": "fixed_vector", "fixed_vector_source": "maccs", "fusion": "concat", "loss": "bce"},
    "fixed_vector_maccs_concat_focal": {"encoder": "fixed_vector", "fixed_vector_source": "maccs", "fusion": "concat", "loss": "focal"},
    "fixed_vector_maccs_concat_weighted_bce": {"encoder": "fixed_vector", "fixed_vector_source": "maccs", "fusion": "concat", "loss": "weighted_bce"},
    "fixed_vector_mol2vec_concat_asl": {"encoder": "fixed_vector", "fixed_vector_source": "mol2vec", "fusion": "concat"},
    "fixed_vector_mol2vec_concat_bce": {"encoder": "fixed_vector", "fixed_vector_source": "mol2vec", "fusion": "concat", "loss": "bce"},
    "fixed_vector_mol2vec_concat_focal": {"encoder": "fixed_vector", "fixed_vector_source": "mol2vec", "fusion": "concat", "loss": "focal"},
    "fixed_vector_mol2vec_concat_weighted_bce": {"encoder": "fixed_vector", "fixed_vector_source": "mol2vec", "fusion": "concat", "loss": "weighted_bce"},
    "fixed_vector_morgan_concat_asl": {"encoder": "fixed_vector", "fixed_vector_source": "morgan", "fusion": "concat"},
    "fixed_vector_morgan_concat_bce": {"encoder": "fixed_vector", "fixed_vector_source": "morgan", "fusion": "concat", "loss": "bce"},
    "fixed_vector_morgan_concat_focal": {"encoder": "fixed_vector", "fixed_vector_source": "morgan", "fusion": "concat", "loss": "focal"},
    "fixed_vector_morgan_concat_weighted_bce": {"encoder": "fixed_vector", "fixed_vector_source": "morgan", "fusion": "concat", "loss": "weighted_bce"},
    "fixed_vector_pubchemfp_concat_asl": {"encoder": "fixed_vector", "fixed_vector_source": "pubchemfp", "fusion": "concat"},
    "fixed_vector_pubchemfp_concat_bce": {"encoder": "fixed_vector", "fixed_vector_source": "pubchemfp", "fusion": "concat", "loss": "bce"},
    "fixed_vector_pubchemfp_concat_focal": {"encoder": "fixed_vector", "fixed_vector_source": "pubchemfp", "fusion": "concat", "loss": "focal"},
    "fixed_vector_pubchemfp_concat_weighted_bce": {"encoder": "fixed_vector", "fixed_vector_source": "pubchemfp", "fusion": "concat", "loss": "weighted_bce"},
    "molformer_concat_asl": {"encoder": "molformer", "fusion": "concat"},
    "molformer_cross_attn_asl": {"encoder": "molformer", "pooling": "cls", "use_descriptors": False},
    "molformer_cross_attn_bce": {"encoder": "molformer", "pooling": "cls", "loss": "bce"},
    "molformer_cross_attn_focal": {"encoder": "molformer", "pooling": "cls", "loss": "focal"},
    "molformer_cross_attn_global_gated_asl": {"encoder": "molformer"},
    "molformer_cross_attn_global_gated_bce": {"encoder": "molformer", "loss": "bce"},
    "molformer_cross_attn_pairwise_asl": {"encoder": "molformer", "pooling": "explicit_pairwise", "batch_size": 16},
    "molformer_cross_attn_pairwise_bce": {"encoder": "molformer", "pooling": "explicit_pairwise", "loss": "bce", "batch_size": 16},
    "molformer_cross_attn_weighted_bce": {"encoder": "molformer", "pooling": "cls", "loss": "weighted_bce"},
    "pretrained_gat_concat_cls_asl": {"encoder": "pretrained_gat", "fusion": "concat", "pooling": "cls"},
    "pretrained_gat_concat_cls_bce": {"encoder": "pretrained_gat", "fusion": "concat", "pooling": "cls", "loss": "bce"},
    "pretrained_gat_concat_global_gated_asl": {"encoder": "pretrained_gat", "fusion": "concat"},
    "pretrained_gat_concat_global_gated_bce": {"encoder": "pretrained_gat", "fusion": "concat", "loss": "bce"},
    "pretrained_gat_cross_attn_cls_asl": {"encoder": "pretrained_gat", "pooling": "cls"},
    "pretrained_gat_cross_attn_cls_bce": {"encoder": "pretrained_gat", "pooling": "cls", "loss": "bce"},
    "pretrained_gat_cross_attn_global_gated_asl": {"encoder": "pretrained_gat"},
    "pretrained_gat_cross_attn_global_gated_bce": {"encoder": "pretrained_gat", "loss": "bce"},
    "pretrained_gin_concat_asl": {"encoder": "pretrained_gin", "fusion": "concat"},
    "pretrained_gin_cross_attn_asl": {"encoder": "pretrained_gin", "pooling": "cls"},
}

# Concat fusion ignores pooling, so these train exactly the same model as their twin
# (their 60/20/20 metrics are identical). --skip_duplicates leaves them out.
DUPLICATE_OF = {
    "dmpnn_chemprop_concat_global_gated_asl": "dmpnn_chemprop_concat_cls_asl",
    "dmpnn_chemprop_concat_global_gated_bce": "dmpnn_chemprop_concat_cls_bce",
    "pretrained_gat_concat_global_gated_asl": "pretrained_gat_concat_cls_asl",
    "pretrained_gat_concat_global_gated_bce": "pretrained_gat_concat_cls_bce",
}

PGB_MODELS = {f"pgb_{fam}": fam for fam in
              ["maccs", "morgan", "pubchemfp", "mol2vec", "molformer", "chemberta", "gin", "gat", "dmpnn"]}

ALL_MODELS = list(BASELINE_MODELS) + list(PGB_MODELS)
