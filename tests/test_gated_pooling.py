"""Unit tests for Global + Gated Attention Pooling.

Verifies:
1. Truly safe all-masked pooling (no NaNs or spurious weights).
2. Padded tokens receive ~0 attention weight and changing their embeddings does not change pooled output.
3. Dimension verification with a DummySequenceEncoder (128-D struct, 609-D pair_vec via classifier hook).
4. Missing-excipient fallback behavior (exc_available == 0).
"""

import unittest
import torch
import torch.nn as nn

from src.config import Config
from src.model import CompatibilityModel, GatedAttentionPooling


class DummySequenceEncoder(nn.Module):
    """Fast dummy sequence-capable encoder for testing without loading MoLFormer."""
    is_sequence_capable = True
    output_dim = 768

    def __init__(self, fixed_len: int = 10):
        super().__init__()
        self.fixed_len = fixed_len

    def encode(self, smiles_batch: list[str]):
        B = len(smiles_batch)
        L = self.fixed_len
        D = self.output_dim
        # Deterministic dummy embeddings based on string lengths
        tokens = torch.randn(B, L, D)
        pooled = torch.randn(B, D)
        # Create varying length padding masks
        mask = torch.zeros(B, L, dtype=torch.bool)
        for i, s in enumerate(smiles_batch):
            valid_len = max(1, min(L, len(s) % L + 1))
            mask[i, valid_len:] = True  # True = padding
        return tokens, pooled, mask


class TestGlobalGatedAttentionPooling(unittest.TestCase):

    def test_truly_safe_all_masked_pooling(self):
        """Test that completely masked sequences yield finite, zeroed-weight outputs without NaNs."""
        pool_layer = GatedAttentionPooling(input_dim=128, hidden_dim=128)
        B, L, D = 2, 5, 128
        x = torch.randn(B, L, D)
        # First row normal, second row completely masked
        padding_mask = torch.zeros(B, L, dtype=torch.bool)
        padding_mask[1, :] = True

        pooled, weights = pool_layer(x, padding_mask)

        # Output checks
        self.assertTrue(torch.isfinite(pooled).all(), "Pooled output contains non-finite values")
        self.assertTrue(torch.isfinite(weights).all(), "Weights contain non-finite values")
        # Row 1 must have exactly zero weights and zero pooled output
        self.assertTrue((weights[1] == 0.0).all(), "All-masked row weights must be strictly zero")
        self.assertTrue((pooled[1] == 0.0).all(), "All-masked row pooled output must be strictly zero")

    def test_padding_mask_invariance(self):
        """Test that padded positions receive ~0 weight and changing them doesn't affect pooled vector."""
        pool_layer = GatedAttentionPooling(input_dim=128, hidden_dim=128)
        pool_layer.eval()
        B, L, D = 1, 6, 128
        x = torch.randn(B, L, D)
        padding_mask = torch.tensor([[False, False, False, True, True, True]])

        pooled1, weights1 = pool_layer(x, padding_mask)

        # Padded weights must be zero
        padded_weights = weights1[0, 3:]
        self.assertTrue((padded_weights < 1e-5).all(), f"Padded weights too high: {padded_weights}")

        # Modify values in padded positions heavily
        x_modified = x.clone()
        x_modified[0, 3:, :] += 1000.0

        pooled2, weights2 = pool_layer(x_modified, padding_mask)

        # Pooled vector must remain unchanged
        self.assertTrue(torch.allclose(pooled1, pooled2, atol=1e-5),
                        "Modifying padded tokens altered the pooled representation!")

    def test_full_model_dimensions_via_hooks(self):
        """Test full model forward pass with a dummy encoder and classifier hook for pair_vec."""
        config = Config()
        config.encoder = "molformer"
        config.fusion = "cross_attn"
        config.pooling = "global_gated_attention"
        config.pool_hidden_dim = 128

        encoder = DummySequenceEncoder(fixed_len=8)
        model = CompatibilityModel(config, encoder)
        model.eval()

        # Capture intermediate tensors via forward hooks
        captured = {}

        def struct_api_hook(mod, inp, out):
            captured["h_api_struct"] = out

        def struct_exc_hook(mod, inp, out):
            captured["h_exc_struct"] = out

        def classifier_hook(mod, inp, out):
            captured["pair_vec"] = inp[0]

        model.api_pool_fusion.register_forward_hook(struct_api_hook)
        model.exc_pool_fusion.register_forward_hook(struct_exc_hook)
        model.classifier.register_forward_hook(classifier_hook)

        B = 4
        batch = {
            "api_smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"] * B,
            "exc_smiles": ["CCO", "CC", "CCCO", "c1ccccc1"],
            "api_desc": torch.randn(B, 21),
            "exc_desc": torch.randn(B, 21),
            "exc_available": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        }

        logits = model(batch)

        # Assert shapes
        self.assertEqual(logits.shape, (B,))
        self.assertEqual(captured["h_api_struct"].shape, (B, 128))
        self.assertEqual(captured["h_exc_struct"].shape, (B, 128))
        self.assertEqual(captured["pair_vec"].shape, (B, 609))

        # Assert finite values
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(captured["h_api_struct"]).all())
        self.assertTrue(torch.isfinite(captured["h_exc_struct"]).all())
        self.assertTrue(torch.isfinite(captured["pair_vec"]).all())

        # Assert token weights stored and detached
        self.assertIsNotNone(model.api_token_weights)
        self.assertIsNotNone(model.exc_token_weights)
        self.assertFalse(model.api_token_weights.requires_grad)
        self.assertFalse(model.exc_token_weights.requires_grad)

    def test_missing_excipient_fallback(self):
        """Test missing-excipient batch (exc_available == 0) uses fallback and does not produce NaNs."""
        config = Config()
        config.encoder = "molformer"
        config.fusion = "cross_attn"
        config.pooling = "global_gated_attention"

        encoder = DummySequenceEncoder(fixed_len=8)
        model = CompatibilityModel(config, encoder)
        model.eval()

        captured = {}
        model.classifier.register_forward_hook(lambda mod, inp, out: captured.update({"pair_vec": inp[0]}))

        B = 3
        batch = {
            "api_smiles": ["CC(=O)O"] * B,
            "exc_smiles": ["", "CCO", ""],
            "api_desc": torch.randn(B, 21),
            "exc_desc": torch.randn(B, 21),
            "exc_available": torch.tensor([0.0, 1.0, 0.0]),  # index 0 and 2 are missing
        }

        logits = model(batch)

        self.assertEqual(logits.shape, (B,))
        self.assertEqual(captured["pair_vec"].shape, (B, 609))
        self.assertTrue(torch.isfinite(logits).all(), "Missing excipients caused NaN in logits!")
        self.assertTrue(torch.isfinite(captured["pair_vec"]).all(), "Missing excipients caused NaN in pair_vec!")

    def test_legacy_cls_pooling_mode(self):
        """Verify that setting pooling='cls' still runs and produces identical downstream shapes."""
        config = Config()
        config.encoder = "molformer"
        config.fusion = "cross_attn"
        config.pooling = "cls"

        encoder = DummySequenceEncoder(fixed_len=8)
        model = CompatibilityModel(config, encoder)
        model.eval()

        captured = {}
        model.classifier.register_forward_hook(lambda mod, inp, out: captured.update({"pair_vec": inp[0]}))

        B = 2
        batch = {
            "api_smiles": ["CC"] * B,
            "exc_smiles": ["CO"] * B,
            "api_desc": torch.randn(B, 21),
            "exc_desc": torch.randn(B, 21),
            "exc_available": torch.tensor([1.0, 1.0]),
        }

        logits = model(batch)
        self.assertEqual(logits.shape, (B,))
        self.assertEqual(captured["pair_vec"].shape, (B, 609))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
