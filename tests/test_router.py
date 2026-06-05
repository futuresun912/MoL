"""Tests for the TriRouter primitive."""
import torch
from mol.router import TriRouter, gumbel_softmax


def test_router_shape():
    H, RD, RH, L = 64, 16, 32, 8
    router = TriRouter(H, RD, RH, num_positions=L)
    b = 2
    pooled = torch.randn(b, H)
    z = router.init_state(b, pooled.device)
    logits, z_new = router(pooled, z, pos_idx=3)
    assert logits.shape == (b, 3)
    assert z_new.shape == (b, RD)


def test_router_layer_bias_indexable():
    """Per-position bias should affect output at the matching position only."""
    router = TriRouter(64, 16, 32, num_positions=8)
    router.layer_bias.data[3, 0] = 5.0    # large skip bias at position 3 only
    z = torch.zeros(1, 16)
    pooled = torch.zeros(1, 64)
    logits_3, _ = router(pooled, z, pos_idx=3)
    logits_2, _ = router(pooled, z, pos_idx=2)
    assert logits_3[0, 0] > logits_2[0, 0]   # skip favoured at position 3


def test_gumbel_softmax_train_mode():
    """In train mode (requires_grad=True) the output sums to 1 along last dim."""
    logits = torch.randn(2, 3, requires_grad=True)
    out = gumbel_softmax(logits, tau=1.0, hard=False)
    assert out.shape == (2, 3)
    assert torch.allclose(out.sum(-1), torch.ones(2), atol=1e-5)


def test_gumbel_softmax_hard_one_hot():
    logits = torch.randn(2, 3, requires_grad=True)
    out = gumbel_softmax(logits, tau=1.0, hard=True)
    # Forward value should be one-hot (sum=1, argmax in {0,1,2}, exactly one 1)
    is_one_hot = ((out > 0.99).sum(-1) == 1).all().item()
    assert is_one_hot


def test_gumbel_softmax_inference_deterministic():
    """Without grad, returns deterministic argmax one-hot."""
    logits = torch.tensor([[0.1, 0.9, 0.2], [0.0, 0.0, 1.5]])
    out = gumbel_softmax(logits, tau=1.0, hard=False)
    expected = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(out, expected)


def test_alpha_in_unit_interval():
    router = TriRouter(8, 4, 8, num_positions=4)
    a = float(router.alpha.item())
    assert 0.0 <= a <= 1.0
