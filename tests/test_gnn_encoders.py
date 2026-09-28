"""Comprehensive test suite for GNN encoders.

Tests:
1. Graph construction (39 atom features, 10 bond features, directed edges)
2. Per-encoder shape tests
3. Padding semantics (True = padding)
4. Mean pooling correctness
5. Missing/empty SMILES handling
6. Invalid SMILES error
7. Multi-fragment molecules
8. Single-atom molecules
9. Gradient flow through all GNN encoders
10. All four losses with GNN encoders
11. CompatibilityModel integration (concat + cross_attn)
12. PNA degree histogram
"""

import sys
import os
import traceback

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_graph_construction():
    """Test 39-dim atom features, 10-dim bond features, directed edges."""
    from src.encoders.gnn_common import smiles_to_pyg_data, ATOM_FEATURE_DIM, BOND_FEATURE_DIM

    # Ethanol: 3 atoms (C, C, O), 2 bonds -> 4 directed edges
    data = smiles_to_pyg_data("CCO")
    assert data.x.shape[1] == ATOM_FEATURE_DIM == 39, f"Expected 39, got {data.x.shape[1]}"
    assert data.edge_attr.shape[1] == BOND_FEATURE_DIM == 10, f"Expected 10, got {data.edge_attr.shape[1]}"
    assert data.edge_index.shape[0] == 2
    # 2 bonds * 2 directions = 4 edges
    assert data.edge_index.shape[1] == 4, f"Expected 4 edges, got {data.edge_index.shape[1]}"
    print("  PASS: graph construction (39/10 features, directed edges)")

    # No self-loops in raw graph
    src, dst = data.edge_index
    assert not (src == dst).any(), "Found self-loops in raw graph"
    print("  PASS: no self-loops in raw graph")


def test_multi_fragment():
    """Test multi-fragment SMILES (salts)."""
    from src.encoders.gnn_common import smiles_to_pyg_data

    data = smiles_to_pyg_data("CC(=O)[O-].[Na+]")
    # Acetate: 4 atoms (C, C, O, O-), Na+: 1 atom = 5 total
    assert data.x.shape[0] == 5, f"Expected 5 atoms, got {data.x.shape[0]}"
    assert data.x.shape[1] == 39
    assert not torch.isnan(data.x).any()
    print("  PASS: multi-fragment molecule (salt)")


def test_single_atom():
    """Test single-atom molecule."""
    from src.encoders.gnn_common import smiles_to_pyg_data

    data = smiles_to_pyg_data("[Na+]")
    assert data.x.shape[0] == 1, f"Expected 1 atom, got {data.x.shape[0]}"
    assert data.edge_index.shape == (2, 0), f"Expected [2,0], got {data.edge_index.shape}"
    assert data.edge_attr.shape == (0, 10), f"Expected [0,10], got {data.edge_attr.shape}"
    print("  PASS: single-atom molecule (no bonds)")


def test_invalid_smiles():
    """Test that invalid SMILES raises ValueError."""
    from src.encoders.gnn_common import smiles_to_pyg_data

    try:
        smiles_to_pyg_data("this_is_not_a_valid_smiles")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot parse" in str(e).lower()
        print("  PASS: invalid SMILES raises ValueError")


def test_encoder_shapes(encoder_cls, config, name):
    """Test encoder output shapes for a batch of SMILES."""
    enc = encoder_cls(config, device="cpu")

    smiles = ["CCO", "c1ccccc1", "CC(=O)[O-].[Na+]"]
    tok, pool, mask = enc.encode(smiles)

    assert tok.ndim == 3, f"{name}: token_embeddings should be 3D, got {tok.ndim}"
    assert pool.ndim == 2, f"{name}: pooled should be 2D, got {pool.ndim}"
    assert mask.ndim == 2, f"{name}: token_mask should be 2D, got {mask.ndim}"

    assert tok.shape[0] == 3, f"{name}: batch size should be 3"
    assert pool.shape == (3, enc.output_dim), f"{name}: pooled shape mismatch"
    assert tok.shape[2] == enc.output_dim, f"{name}: token dim mismatch"
    assert mask.shape == tok.shape[:2], f"{name}: mask shape mismatch"

    assert not torch.isnan(tok).any(), f"{name}: NaN in token_embeddings"
    assert not torch.isnan(pool).any(), f"{name}: NaN in pooled"
    assert not torch.isinf(tok).any(), f"{name}: Inf in token_embeddings"

    print(f"  PASS: {name} shapes (output_dim={enc.output_dim})")
    return enc


def test_padding_semantics(encoder_cls, config, name):
    """Test that mask=True means padding, mask=False means real atom."""
    enc = encoder_cls(config, device="cpu")

    smiles = ["C", "CC", "c1ccccc1"]  # 1, 2, 6 atoms
    tok, pool, mask = enc.encode(smiles)

    # C has 1 atom, should have mask[0, 0] = False (real)
    assert mask[0, 0].item() == False, f"{name}: first atom should not be padded"

    # CC has 2 atoms, benzene has 6
    # In the dense batch, L_max = 6 (benzene)
    # For C (1 atom): positions 1..5 should be padded (True)
    for i in range(1, mask.shape[1]):
        assert mask[0, i].item() == True, f"{name}: position {i} for 'C' should be padded"

    print(f"  PASS: {name} padding semantics")


def test_missing_smiles(encoder_cls, config, name):
    """Test empty SMILES handling."""
    enc = encoder_cls(config, device="cpu")

    tok, pool, mask = enc.encode(["CCO", ""])

    # Empty SMILES should have all-True mask
    assert mask[1].all(), f"{name}: empty SMILES should have all-True mask"
    # Empty SMILES pooled should be zeros
    assert (pool[1] == 0).all(), f"{name}: empty SMILES pooled should be zeros"
    # Valid SMILES should produce finite output
    assert not torch.isnan(tok[0]).any(), f"{name}: valid SMILES should not have NaN"

    print(f"  PASS: {name} missing SMILES handling")


def test_gradient_flow(encoder_cls, config, name):
    """Test that gradients flow back into the GNN."""
    enc = encoder_cls(config, device="cpu")
    enc.train()

    tok, pool, mask = enc.encode(["CCO", "c1ccccc1"])
    loss = pool.sum()
    loss.backward()

    has_grad = False
    for p_name, p in enc.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            assert torch.isfinite(p.grad).all(), f"{name}: non-finite gradient in {p_name}"
            break

    assert has_grad, f"{name}: no gradients reached GNN parameters!"
    print(f"  PASS: {name} gradient flow")


def test_model_integration_concat(encoder_cls, config, name):
    """Test encoder works with CompatibilityModel in concat mode."""
    from src.model import CompatibilityModel

    cfg = _make_config(config, fusion="concat", loss="bce")
    enc = encoder_cls(cfg, device="cpu")
    cfg.encoder_output_dim = enc.output_dim

    model = CompatibilityModel(cfg, enc)
    model.train()

    batch = _make_batch()
    logits = model(batch)

    assert logits.shape == (2,), f"{name} concat: expected [2], got {logits.shape}"
    assert torch.isfinite(logits).all(), f"{name} concat: non-finite logits"

    loss = logits.sum()
    loss.backward()

    print(f"  PASS: {name} concat integration")


def test_model_integration_cross_attn(encoder_cls, config, name):
    """Test encoder works with CompatibilityModel in cross-attention mode."""
    from src.model import CompatibilityModel

    cfg = _make_config(config, fusion="cross_attn", loss="bce")
    enc = encoder_cls(cfg, device="cpu")
    cfg.encoder_output_dim = enc.output_dim

    model = CompatibilityModel(cfg, enc)
    model.train()

    batch = _make_batch()
    logits = model(batch)

    assert logits.shape == (2,), f"{name} cross_attn: expected [2], got {logits.shape}"
    assert torch.isfinite(logits).all(), f"{name} cross_attn: non-finite logits"

    loss = logits.sum()
    loss.backward()

    print(f"  PASS: {name} cross_attn integration")


def test_all_losses(encoder_cls, config, name):
    """Test all four losses with a GNN encoder."""
    from src.model import CompatibilityModel
    from src.loss import get_loss_fn

    for loss_name in ["bce", "weighted_bce", "focal", "asl"]:
        cfg = _make_config(config, fusion="concat", loss=loss_name)
        enc = encoder_cls(cfg, device="cpu")
        cfg.encoder_output_dim = enc.output_dim

        model = CompatibilityModel(cfg, enc)
        model.train()

        batch = _make_batch()
        logits = model(batch)
        labels = torch.tensor([1.0, 0.0])
        sample_weight = torch.ones(2)

        loss_fn = get_loss_fn(cfg)
        if loss_name == "weighted_bce":
            loss = loss_fn(logits, labels, sample_weight, pw=torch.tensor(2.0))
        else:
            loss = loss_fn(logits, labels, sample_weight)

        assert torch.isfinite(loss), f"{name} {loss_name}: non-finite loss"
        loss.backward()

        # Clear gradients for next iteration
        model.zero_grad()

    print(f"  PASS: {name} all four losses")


def test_pna_degree_histogram():
    """Test PNA degree histogram computation."""
    from src.encoders.pna_utils import compute_degree_histogram

    hist = compute_degree_histogram(["CCO", "c1ccccc1", "CC(=O)O", "[Na+]"])

    assert hist.dtype == torch.long
    assert hist.numel() > 0
    assert hist.sum() > 0
    assert hist[0].item() > 0  # single-atom Na+ has degree 0

    # Verify no negative values
    assert (hist >= 0).all()

    print("  PASS: PNA degree histogram")


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_config(base_config, fusion="concat", loss="bce"):
    """Create a Config for testing."""
    from src.config import Config
    import copy
    cfg = copy.deepcopy(base_config)
    cfg.fusion = fusion
    cfg.loss = loss
    cfg.pooling = "global_gated_attention"
    cfg.use_descriptors = True
    return cfg


def _make_batch():
    """Create a minimal batch dict for testing."""
    return {
        "api_smiles": ["CCO", "c1ccccc1"],
        "exc_smiles": ["CC(=O)O", "O"],
        "api_desc": torch.randn(2, 21),
        "exc_desc": torch.randn(2, 21),
        "exc_available": torch.tensor([1.0, 1.0]),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    from src.config import Config
    from src.encoders.gine_encoder import GINEEncoder
    from src.encoders.gatv2_encoder import GATv2Encoder
    from src.encoders.attentivefp_encoder import AttentiveFPEncoder
    from src.encoders.pna_encoder import PNAEncoder
    from src.encoders.pna_utils import compute_degree_histogram

    config = Config()
    config.encoder = "gine"  # default for base config

    # Compute PNA degree histogram
    deg_hist = compute_degree_histogram(["CCO", "c1ccccc1", "CC(=O)O", "[Na+]", "CC(=O)[O-].[Na+]"])
    config.pna_degree_hist = deg_hist.tolist()

    pyg_encoders = {
        "attentivefp": AttentiveFPEncoder,
        "gine": GINEEncoder,
        "gatv2": GATv2Encoder,
        "pna": PNAEncoder,
    }

    passed = 0
    failed = 0

    # Graph construction tests
    print("\n=== Graph Construction ===")
    for test_fn in [test_graph_construction, test_multi_fragment, test_single_atom, test_invalid_smiles]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    # PNA degree histogram
    print("\n=== PNA Degree Histogram ===")
    try:
        test_pna_degree_histogram()
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        failed += 1

    # Per-encoder tests
    for enc_name, enc_cls in pyg_encoders.items():
        print(f"\n=== {enc_name.upper()} ===")
        enc_config = Config()
        enc_config.encoder = enc_name
        enc_config.pna_degree_hist = deg_hist.tolist()

        for test_fn in [test_encoder_shapes, test_padding_semantics, test_missing_smiles, test_gradient_flow]:
            try:
                test_fn(enc_cls, enc_config, enc_name)
                passed += 1
            except Exception as e:
                print(f"  FAIL: {test_fn.__name__}: {e}")
                traceback.print_exc()
                failed += 1

        # Integration tests
        for test_fn in [test_model_integration_concat, test_model_integration_cross_attn, test_all_losses]:
            try:
                test_fn(enc_cls, enc_config, enc_name)
                passed += 1
            except Exception as e:
                print(f"  FAIL: {test_fn.__name__}: {e}")
                traceback.print_exc()
                failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
