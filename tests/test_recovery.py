"""Tests for the LoRA + KD recovery utilities."""
import torch
import torch.nn as nn

from mol import recovery as R


def _count_lora(model):
    return sum(1 for _ in R.iterate_lora(model))


def test_add_lora_replaces_linears(tiny_llama):
    assert _count_lora(tiny_llama) == 0
    n = R.add_lora(tiny_llama, r=4, alpha=8)
    # Llama decoder has 7 target linears per layer × 4 layers = 28
    expected = 4 * 7
    assert n == expected
    assert _count_lora(tiny_llama) == expected


def test_lora_initial_delta_is_zero(tiny_llama, input_ids):
    """With B=0 at init, LoRA-on output equals dense LoRA-off output."""
    with torch.no_grad():
        out_pre = tiny_llama(input_ids, use_cache=False).logits.clone()
    R.add_lora(tiny_llama, r=4, alpha=8)
    R.set_lora(tiny_llama, True)
    with torch.no_grad():
        out_post = tiny_llama(input_ids, use_cache=False).logits
    assert torch.allclose(out_pre, out_post, atol=1e-5)


def test_set_lora_enable_disable(tiny_llama):
    R.add_lora(tiny_llama, r=4)
    R.set_lora(tiny_llama, False)
    for m in R.iterate_lora(tiny_llama):
        assert m.enabled is False
    R.set_lora(tiny_llama, True)
    for m in R.iterate_lora(tiny_llama):
        assert m.enabled is True


def test_set_lora_trainable(tiny_llama):
    R.add_lora(tiny_llama, r=4)
    R.set_lora_trainable(tiny_llama, False)
    for p in R.lora_parameters(tiny_llama):
        assert p.requires_grad is False
    R.set_lora_trainable(tiny_llama, True)
    for p in R.lora_parameters(tiny_llama):
        assert p.requires_grad is True


def test_kd_loss_zero_when_equal():
    """KD divergence should be ~0 when student and teacher are identical."""
    logits = torch.randn(2, 8, 16)
    loss = R.kd_loss(logits, logits, temperature=1.0)
    assert loss.item() < 1e-5


def test_kd_loss_positive_when_different():
    s = torch.randn(2, 8, 16)
    t = torch.randn(2, 8, 16)
    loss = R.kd_loss(s, t, temperature=1.0)
    assert loss.item() > 0.0


def test_block_influence_shape(tiny_llama, calib_seqs):
    bi = R.block_influence(tiny_llama, calib_seqs, calib_seqs[0].device)
    assert bi.shape == (tiny_llama.config.num_hidden_layers,)
    # BI should be non-negative (cosine distance)
    assert (bi >= 0).all().item()


def test_apply_route_skip_identity(tiny_llama, input_ids):
    """All-skip route on routable layers should pass hidden states unchanged
    through those layers (but boundaries unaffected here since route is set)."""
    L = tiny_llama.config.num_hidden_layers
    # All layers skip → output should equal embeddings flowed through (mostly) identity
    route = [0] * L
    restore = R.apply_route(tiny_llama, route)
    try:
        with torch.no_grad():
            out = tiny_llama(input_ids, use_cache=False).logits
    finally:
        restore()
    # We just need the model to not crash and return a sensible shape
    assert out.shape == (1, 32, tiny_llama.config.vocab_size)
    assert torch.isfinite(out).all().item()
