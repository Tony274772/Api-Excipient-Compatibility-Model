"""Integration test: verify all 40 GNN encoder × fusion × loss combinations.

For each combination:
1. Instantiate config
2. Compute PNA degree histogram if needed
3. Instantiate encoder
4. Instantiate CompatibilityModel
5. Run one forward pass
6. Compute loss
7. Run backward()
8. Check finite logit/loss/gradient
"""

import sys
import os
import traceback
import gc

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ENCODERS = ["attentivefp", "gine", "gatv2", "pna"]
# dmpnn_chemprop excluded by default since it requires the chemeleon checkpoint
# Add "dmpnn_chemprop" if checkpoint is available, "dmpnn_scratch" for scratch D-MPNN

FUSIONS = ["concat", "cross_attn"]
LOSSES = ["bce", "weighted_bce", "focal", "asl"]


def test_combination(encoder_name, fusion, loss_name):
    """Test a single encoder/fusion/loss combination."""
    import copy
    from src.config import Config
    from src.model import CompatibilityModel
    from src.loss import get_loss_fn
    from src.encoders import ENCODER_REGISTRY
    from src.encoders.pna_utils import compute_degree_histogram

    config = Config()
    config.encoder = encoder_name
    config.fusion = fusion
    config.loss = loss_name
    config.pooling = "global_gated_attention"
    config.use_descriptors = True

    # PNA needs degree histogram
    if encoder_name == "pna":
        deg_hist = compute_degree_histogram(["CCO", "c1ccccc1", "CC(=O)O"])
        config.pna_degree_hist = deg_hist.tolist()

    # Build encoder
    encoder_cls = ENCODER_REGISTRY[config.encoder]
    encoder = encoder_cls(config, device="cpu")
    config.encoder_output_dim = encoder.output_dim

    # Build model
    model = CompatibilityModel(config, encoder)
    model.train()

    # Forward pass
    batch = {
        "api_smiles": ["CCO", "c1ccccc1"],
        "exc_smiles": ["CC(=O)O", "O"],
        "api_desc": torch.randn(2, 21),
        "exc_desc": torch.randn(2, 21),
        "exc_available": torch.tensor([1.0, 1.0]),
    }

    logits = model(batch)
    assert logits.shape == (2,), f"Expected [2], got {logits.shape}"
    assert torch.isfinite(logits).all(), "Non-finite logits"

    # Compute loss
    labels = torch.tensor([1.0, 0.0])
    sample_weight = torch.ones(2)

    loss_fn = get_loss_fn(config)
    if loss_name == "weighted_bce":
        loss = loss_fn(logits, labels, sample_weight, pw=torch.tensor(2.0))
    else:
        loss = loss_fn(logits, labels, sample_weight)

    assert torch.isfinite(loss), "Non-finite loss"

    # Backward
    loss.backward()

    # Check finite gradients exist
    has_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            if p.grad.abs().sum() > 0:
                has_grad = True
                break

    assert has_grad, "No finite gradients found in model parameters"

    return True


def main():
    # Check if dmpnn_scratch is available
    try:
        from src.encoders.dmpnn_encoder import ScratchDMPNNEncoder
        encoders = ENCODERS + ["dmpnn_scratch"]
    except ImportError:
        encoders = ENCODERS

    total = len(encoders) * len(FUSIONS) * len(LOSSES)
    print(f"Testing {total} combinations: {len(encoders)} encoders × {len(FUSIONS)} fusions × {len(LOSSES)} losses")

    passed = 0
    failed = 0
    errors = []

    for enc in encoders:
        for fusion in FUSIONS:
            for loss in LOSSES:
                combo = f"{enc}_{fusion}_{loss}"
                try:
                    test_combination(enc, fusion, loss)
                    print(f"  PASS: {combo}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL: {combo}: {e}")
                    errors.append((combo, str(e)))
                    failed += 1
                finally:
                    gc.collect()

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    if errors:
        print("\nFailed combinations:")
        for combo, err in errors:
            print(f"  {combo}: {err}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
