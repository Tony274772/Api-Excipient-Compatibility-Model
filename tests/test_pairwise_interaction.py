"""Unit tests for Explicit Pairwise Token Interaction (Instructions/pre_pairwise.md).

Verifies:
1. Explicit pairwise feature construction ([B, L_a, L_e, 512] with exact ordering).
2. Pairwise padding mask semantics.
3. Attention padding weight suppression and sum-to-1 normalization.
4. Padding invariance (altering masked pair features doesn't affect pooled output).
5. All-pairs-masked behavior (zero weights, zero pooled vector, no NaNs).
6. Full model dimensions via forward hook on classifier (817-D pair_vector, [B] logits).
7. Missing-excipient fallback handling (exc_available == 0).
8. CLS baseline remains reproducible (609-D pair_vector).
9. Global + Gated attention remains default and isolated (609-D pair_vector).
"""

import unittest
import torch
import torch.nn as nn

from src.config import Config
from src.model import CompatibilityModel, PairwiseAttentionPooling


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
        tokens = torch.randn(B, L, D)
        pooled = torch.randn(B, D)
        mask = torch.zeros(B, L, dtype=torch.bool)
        for i, s in enumerate(smiles_batch):
            valid_len = max(1, min(L, len(s) % L + 1))
            mask[i, valid_len:] = True  # True = padding
        return tokens, pooled, mask


class TestPairwiseInteraction(unittest.TestCase):

    def test_pair_feature_construction(self):
        """Test 1: Vectorized pair feature construction matches the 512-D specification."""
        B, L_api, L_exc, D = 2, 3, 4, 128
        api = torch.randn(B, L_api, D)
        exc = torch.randn(B, L_exc, D)

        api_expanded = api.unsqueeze(2)
        exc_expanded = exc.unsqueeze(1)
        product = api_expanded * exc_expanded
        difference = torch.abs(api_expanded - exc_expanded)

        pair_tensor = torch.cat(
            [
                api_expanded.expand(-1, -1, L_exc, -1),
                exc_expanded.expand(-1, L_api, -1, -1),
                product,
                difference,
            ],
            dim=-1,
        )

        self.assertEqual(pair_tensor.shape, (B, L_api, L_exc, 512))

        # Check manual indexing for a sample (i=1, j=2)
        i, j = 1, 2
        torch.testing.assert_close(pair_tensor[0, i, j, :128], api[0, i])
        torch.testing.assert_close(pair_tensor[0, i, j, 128:256], exc[0, j])
        torch.testing.assert_close(pair_tensor[0, i, j, 256:384], api[0, i] * exc[0, j])
        torch.testing.assert_close(pair_tensor[0, i, j, 384:512], torch.abs(api[0, i] - exc[0, j]))

    def test_pair_mask_construction(self):
        """Test 2: Pair mask semantics: invalid if either token is invalid."""
        # API: [valid, valid, pad]
        api_mask = torch.tensor([[False, False, True]])  # [1, 3]
        # Exc: [valid, valid, valid, pad]
        exc_mask = torch.tensor([[False, False, False, True]])  # [1, 4]

        pair_mask = api_mask.unsqueeze(2) | exc_mask.unsqueeze(1)  # [1, 3, 4]

        # Valid pairs: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
        for i in range(2):
            for j in range(3):
                self.assertFalse(pair_mask[0, i, j].item())

        # Invalid pairs: any with i==2 or j==3
        for j in range(4):
            self.assertTrue(pair_mask[0, 2, j].item())
        for i in range(3):
            self.assertTrue(pair_mask[0, i, 3].item())

    def test_pairwise_attention_padding(self):
        """Test 3: Padding pairs receive ~0 weight and valid weights sum to 1."""
        pool = PairwiseAttentionPooling(input_dim=512, hidden_dim=256)
        pool.eval()
        B, N = 2, 6
        x = torch.randn(B, N, 512)
        mask = torch.tensor([
            [False, False, False, True, True, True],
            [False, False, True, True, True, True],
        ])

        pooled, weights = pool(x, mask)

        self.assertEqual(pooled.shape, (B, 512))
        self.assertEqual(weights.shape, (B, N))

        # Check padded weights are ~0
        self.assertTrue((weights[0, 3:] < 1e-5).all())
        self.assertTrue((weights[1, 2:] < 1e-5).all())

        # Check valid weights sum to 1.0
        torch.testing.assert_close(weights[0, :3].sum(), torch.tensor(1.0), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(weights[1, :2].sum(), torch.tensor(1.0), atol=1e-4, rtol=1e-4)

    def test_padding_invariance(self):
        """Test 4: Modifying padded pair features does not alter the pooled representation."""
        pool = PairwiseAttentionPooling(input_dim=512, hidden_dim=256)
        pool.eval()
        B, N = 1, 8
        x1 = torch.randn(B, N, 512)
        mask = torch.tensor([[False, False, False, False, True, True, True, True]])

        x2 = x1.clone()
        x2[0, 4:, :] += 5000.0  # massively perturb invalid pairs

        pooled1, _ = pool(x1, mask)
        pooled2, _ = pool(x2, mask)

        torch.testing.assert_close(pooled1, pooled2, atol=1e-5, rtol=1e-5)

    def test_all_pairs_masked_behavior(self):
        """Test 5: Completely masked pair grid produces zero weights and zero pooled vector without NaNs."""
        pool = PairwiseAttentionPooling(input_dim=512, hidden_dim=256)
        pool.eval()
        B, N = 2, 5
        x = torch.randn(B, N, 512)
        mask = torch.ones(B, N, dtype=torch.bool)  # all masked

        pooled, weights = pool(x, mask)

        self.assertTrue(torch.isfinite(pooled).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights == 0.0).all())
        self.assertTrue((pooled == 0.0).all())

    def test_full_model_dimensions_explicit_pairwise(self):
        """Test 6: Full model with explicit pairwise produces 817-D pair_vector and [B] logits."""
        config = Config()
        config.encoder = "molformer"
        config.fusion = "cross_attn"
        config.pooling = "explicit_pairwise"

        encoder = DummySequenceEncoder(fixed_len=6)
        model = CompatibilityModel(config, encoder)
        model.eval()

        captured = {}
        model.classifier.register_forward_hook(lambda m, inp, out: captured.update({"pair_vec": inp[0]}))

        B = 3
        batch = {
            "api_smiles": ["CC(=O)O"] * B,
            "exc_smiles": ["CCO", "CC", "CCCO"],
            "api_desc": torch.randn(B, 21),
            "exc_desc": torch.randn(B, 21),
            "exc_available": torch.tensor([1.0, 1.0, 1.0]),
        }

        logits = model(batch)

        self.assertEqual(logits.shape, (B,))
        self.assertEqual(captured["pair_vec"].shape, (B, 817))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(captured["pair_vec"]).all())

        # Verify detached weights stored
        self.assertIsNotNone(model.pair_token_weights)
        self.assertFalse(model.pair_token_weights.requires_grad)

    def test_missing_excipient_fallback(self):
        """Test 7: Missing excipient (exc_available == 0) handles zero valid pairs safely without NaNs."""
        config = Config()
        config.encoder = "molformer"
        config.fusion = "cross_attn"
        config.pooling = "explicit_pairwise"

        encoder = DummySequenceEncoder(fixed_len=6)
        model = CompatibilityModel(config, encoder)
        model.eval()

        captured = {}
        model.classifier.register_forward_hook(lambda m, inp, out: captured.update({"pair_vec": inp[0]}))

        B = 2
        batch = {
            "api_smiles": ["CC(=O)O", "c1ccccc1"],
            "exc_smiles": ["CCO", ""],
            "api_desc": torch.randn(B, 21),
            "exc_desc": torch.randn(B, 21),
            "exc_available": torch.tensor([1.0, 0.0]),  # second sample has missing excipient
        }

        logits = model(batch)

        self.assertEqual(logits.shape, (B,))
        self.assertEqual(captured["pair_vec"].shape, (B, 817))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(captured["pair_vec"]).all())

    def test_cls_baseline_remains_reproducible(self):
        """Test 8: Setting pooling='cls' still produces 609-D pair_vector and works as expected."""
        config = Config()
        config.encoder = "molformer"
        config.fusion = "cross_attn"
        config.pooling = "cls"

        encoder = DummySequenceEncoder(fixed_len=6)
        model = CompatibilityModel(config, encoder)
        model.eval()

        captured = {}
        model.classifier.register_forward_hook(lambda m, inp, out: captured.update({"pair_vec": inp[0]}))

        B = 2
        batch = {
            "api_smiles": ["CC", "CCC"],
            "exc_smiles": ["CO", "CCO"],
            "api_desc": torch.randn(B, 21),
            "exc_desc": torch.randn(B, 21),
            "exc_available": torch.tensor([1.0, 1.0]),
        }

        logits = model(batch)
        self.assertEqual(logits.shape, (B,))
        self.assertEqual(captured["pair_vec"].shape, (B, 609))
        self.assertTrue(torch.isfinite(logits).all())

    def test_global_gated_mode_remains_default_and_isolated(self):
        """Test 9: Default pooling remains 'global_gated_attention' with 609-D pair_vector."""
        config = Config()
        self.assertEqual(config.pooling, "global_gated_attention")

        encoder = DummySequenceEncoder(fixed_len=6)
        model = CompatibilityModel(config, encoder)
        model.eval()

        captured = {}
        model.classifier.register_forward_hook(lambda m, inp, out: captured.update({"pair_vec": inp[0]}))

        B = 2
        batch = {
            "api_smiles": ["CC", "CCC"],
            "exc_smiles": ["CO", "CCO"],
            "api_desc": torch.randn(B, 21),
            "exc_desc": torch.randn(B, 21),
            "exc_available": torch.tensor([1.0, 1.0]),
        }

        logits = model(batch)
        self.assertEqual(logits.shape, (B,))
        self.assertEqual(captured["pair_vec"].shape, (B, 609))
        self.assertTrue(torch.isfinite(logits).all())

    def test_path_resolution(self):
        """Test path derivation for all pooling modes."""
        cfg = Config()
        cfg.encoder = "molformer"
        cfg.fusion = "cross_attn"
        cfg.loss = "asl"

        # Default: global_gated_attention
        cfg.pooling = "global_gated_attention"
        cfg.resolve_paths()
        self.assertEqual(cfg.metrics_dir, "metrics/molformer_cross_attn_global_gated_asl")

        # Explicit pairwise
        cfg.pooling = "explicit_pairwise"
        cfg.resolve_paths()
        self.assertEqual(cfg.metrics_dir, "metrics/molformer_cross_attn_pairwise_asl")
        self.assertEqual(cfg.checkpoint_dir, "checkpoints/molformer_cross_attn_pairwise_asl")

        # CLS
        cfg.pooling = "cls"
        cfg.resolve_paths()
        self.assertEqual(cfg.metrics_dir, "metrics/molformer_cross_attn_cls_asl")


if __name__ == "__main__":
    unittest.main()
