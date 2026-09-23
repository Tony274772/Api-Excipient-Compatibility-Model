import os
import torch
from src.config import Config
from src.encoders import ENCODER_REGISTRY
from src.model import CompatibilityModel

# Fallback only for checkpoints saved before "config" was embedded
LEGACY_MODEL_REGISTRY = {
    "molformer_concat_asl":                  ("molformer", "concat",     "global_gated_attention", "asl"),
    "molformer_cross_attn_asl":              ("molformer", "cross_attn", "global_gated_attention", "asl"),
    "molformer_cross_attn_bce":              ("molformer", "cross_attn", "global_gated_attention", "bce"),
    "molformer_cross_attn_focal":            ("molformer", "cross_attn", "global_gated_attention", "focal"),
    "molformer_cross_attn_weighted_bce":     ("molformer", "cross_attn", "global_gated_attention", "weighted_bce"),
    "molformer_cross_attn_global_gated_asl": ("molformer", "cross_attn", "global_gated_attention", "asl"),
    "molformer_cross_attn_global_gated_bce": ("molformer", "cross_attn", "global_gated_attention", "bce"),
    "molformer_cross_attn_pairwise_asl":     ("molformer", "cross_attn", "explicit_pairwise",      "asl"),
    "molformer_cross_attn_pairwise_bce":     ("molformer", "cross_attn", "explicit_pairwise",      "bce"),
    "chemberta_concat_asl":                  ("chemberta", "concat",     "global_gated_attention", "asl"),
    "chemberta_cross_attn_asl":              ("chemberta", "cross_attn", "global_gated_attention", "asl"),
    "chemberta_cross_attn_bce":              ("chemberta", "cross_attn", "global_gated_attention", "bce"),
    "chemberta_cross_attn_focal":            ("chemberta", "cross_attn", "global_gated_attention", "focal"),
    "chemberta_cross_attn_weighted_bce":     ("chemberta", "cross_attn", "global_gated_attention", "weighted_bce"),
    "chemberta_cross_attn_global_gated_asl": ("chemberta", "cross_attn", "global_gated_attention", "asl"),
    "chemberta_cross_attn_global_gated_bce": ("chemberta", "cross_attn", "global_gated_attention", "bce"),
    "chemberta_cross_attn_pairwise_asl":     ("chemberta", "cross_attn", "explicit_pairwise",      "asl"),
    "chemberta_cross_attn_pairwise_bce":     ("chemberta", "cross_attn", "explicit_pairwise",      "bce"),
    "pretrained_gin_cross_attn_asl":         ("pretrained_gin", "cross_attn", "global_gated_attention", "asl"),
    "pretrained_gin_concat_asl":             ("pretrained_gin", "concat",     "global_gated_attention", "asl"),
    "fixed_vector_pubchemfp_concat_asl":     ("fixed_vector",   "concat",     "global_gated_attention", "asl"),
}

def load_config_for_checkpoint(model_name: str, device, checkpoints_dir: str = "checkpoints") -> tuple[Config, dict]:
    """Build the Config for `model_name` by reading it out of the checkpoint itself."""
    ckpt_path = os.path.join(checkpoints_dir, model_name, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "config" in checkpoint and isinstance(checkpoint["config"], Config):
        config = checkpoint["config"]
        print(f"[{model_name}] loaded config from checkpoint: "
              f"encoder={config.encoder}, fusion={config.fusion}, "
              f"pooling={config.pooling}, loss={config.loss}")
    else:
        if model_name not in LEGACY_MODEL_REGISTRY:
            raise ValueError(
                f"Checkpoint '{model_name}' has no embedded config and is not in "
                f"LEGACY_MODEL_REGISTRY — cannot determine its architecture safely."
            )
        encoder, fusion, pooling, loss = LEGACY_MODEL_REGISTRY[model_name]
        print(f"[{model_name}] WARNING: no embedded config in checkpoint, "
              f"using LEGACY_MODEL_REGISTRY fallback: "
              f"encoder={encoder}, fusion={fusion}, pooling={pooling}, loss={loss}")
        config = Config()
        config.encoder = encoder
        config.fusion = fusion
        config.pooling = pooling
        config.loss = loss

    config.checkpoint_dir = os.path.join(checkpoints_dir, model_name)
    metrics_base = "metrics/random_split" if ("random_split" in checkpoints_dir or getattr(config, "split_type", "cluster") == "random") else "metrics"
    config.metrics_dir = os.path.join(metrics_base, model_name)
    data_path = "data/random_split" if ("random_split" in checkpoints_dir or getattr(config, "split_type", "cluster") == "random") else "data"
    config.train_csv = f"{data_path}/train.csv"
    config.val_csv = f"{data_path}/val.csv"
    config.test_csv = f"{data_path}/test.csv"

    return config, checkpoint


def build_model_from_checkpoint(model_name, device, checkpoints_dir: str = "checkpoints"):
    config, checkpoint = load_config_for_checkpoint(model_name, device, checkpoints_dir=checkpoints_dir)

    # Auto-correct older checkpoints that used "cls" pooling but the default config saved was "global_gated_attention"
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        # If it's a cross-attention model and configured for global gated attention, it should have api_pool
        if config.fusion == "cross_attn" and getattr(config, "pooling", "global_gated_attention") == "global_gated_attention":
            if "api_pool.tanh_proj.weight" not in state_dict:
                print(f"[{model_name}] Auto-correcting pooling to 'cls' because api_pool is missing from state_dict.")
                config.pooling = "cls"

    encoder_cls = ENCODER_REGISTRY[config.encoder]
    # Handle different encoder types correctly if we use this beyond molformer
    if config.encoder == "molformer":
        encoder = encoder_cls(config.molformer_model_path, device=str(device))
    elif config.encoder == "pretrained_gin":
        encoder = encoder_cls(config.gin_pretrained_name, device=str(device))
    elif config.encoder == "chemberta":
        encoder = encoder_cls(config.chemberta_model_path, device=str(device))
    elif config.encoder == "fixed_vector":
        encoder = encoder_cls(
            source=config.fixed_vector_source,
            vector_path=config.fixed_vector_path,
            device=str(device),
        )
    else:
        raise ValueError(f"Unknown encoder: {config.encoder}")
        
    encoder.to(device)
    config.encoder_output_dim = encoder.output_dim

    # Force concat fusion for non-sequence encoders, just like in training
    if not encoder.is_sequence_capable:
        config.fusion = "concat"

    model = CompatibilityModel(config, encoder)
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config
