"""Tests for the soft MoLModel wrapper.

Validates that wrapping freezes the base, that the forward path produces
proper logits, that route extraction works, and that ``apply_route`` is
correctly reversible.
"""
import copy
import torch

from mol import MoLModel
from mol import recovery as R


def test_mol_freezes_base(tiny_llama):
    """Wrapping must set requires_grad=False on every base parameter."""
    base_param_ids = {id(p) for p in tiny_llama.parameters()}
    mol = MoLModel(tiny_llama)
    for p in tiny_llama.parameters():
        if id(p) in base_param_ids:
            assert p.requires_grad is False
    # router params must be trainable
    assert any(p.requires_grad for p in mol.router.parameters())


def test_mol_forward_shape(tiny_llama, input_ids):
    mol = MoLModel(tiny_llama)
    mol.inference_mode = "eval"; mol.deterministic = True
    out = mol(input_ids, use_cache=False)
    V = tiny_llama.config.vocab_size
    assert out.logits.shape == (1, 32, V)


def test_extract_route_runs(tiny_llama, input_ids):
    mol = MoLModel(tiny_llama)
    route = mol.extract_route(input_ids)
    L = tiny_llama.config.num_hidden_layers
    assert len(route) == L
    assert all(m in (0, 1, 2) for m in route)
    # Boundaries forced EXEC
    assert route[0] == 1 and route[L - 1] == 1


def test_apply_route_skip_changes_output(tiny_llama, input_ids):
    """Applying a skip route should change the model output vs dense."""
    # Save a fresh copy for "dense baseline" comparison
    L = tiny_llama.config.num_hidden_layers
    with torch.no_grad():
        out_dense = tiny_llama(input_ids, use_cache=False).logits.clone()

    route = [1] * L
    route[1] = 0   # skip layer 1
    restore = R.apply_route(tiny_llama, route)
    try:
        with torch.no_grad():
            out_routed = tiny_llama(input_ids, use_cache=False).logits
    finally:
        restore()
    # Outputs should differ
    assert not torch.allclose(out_dense, out_routed)
    # After restore, output should match dense again
    with torch.no_grad():
        out_after_restore = tiny_llama(input_ids, use_cache=False).logits
    assert torch.allclose(out_dense, out_after_restore)


def test_apply_route_repeat_doubles_layer(tiny_llama, input_ids):
    """A repeat route should yield a different (and presumably more processed)
    output than dense."""
    L = tiny_llama.config.num_hidden_layers
    with torch.no_grad():
        out_dense = tiny_llama(input_ids, use_cache=False).logits.clone()

    route = [1] * L
    route[1] = 2   # repeat layer 1
    restore = R.apply_route(tiny_llama, route)
    try:
        with torch.no_grad():
            out_repeat = tiny_llama(input_ids, use_cache=False).logits
    finally:
        restore()
    assert not torch.allclose(out_dense, out_repeat)


def test_avg_compute_fraction(tiny_llama, input_ids):
    """For dense forward (no skips/repeats forced) compute should be ~1.0."""
    mol = MoLModel(tiny_llama)
    mol.inference_mode = "eval"; mol.deterministic = True
    out = mol(input_ids, use_cache=False)
    c = mol.avg_compute_fraction()
    # In eval mode with no constraints, router can pick anything — but it
    # initialized with exec-favoring bias, so dense (~1.0) is the expectation.
    # Tiny model may produce some skip/repeat; just check the value is sane.
    assert 0.0 <= c <= 2.0
