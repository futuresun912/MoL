"""Tests for the TopK-MoL deterministic variant.

Validates:
  * top-k assignment satisfies the budget exactly
  * boundary protection works
  * routes vary with input (input-adaptive)
  * forward produces logits with the same vocab as the base
"""
import torch

from mol import TopKMoLModel, SKIP, EXEC, REPEAT


def test_topk_budget_compliance_under_budget(tiny_llama, input_ids):
    """At c⋆=0.75, the router should skip exactly 1 of 4 layers (boundary protected)."""
    tk = TopKMoLModel(tiny_llama, c_star=0.75, repeat_frac=0.0,
                       protect_boundaries=1)
    # n_skip = round(4 * 0.25) = 1; layers {0, 3} are protected boundaries
    assert tk.n_skip == 1
    assert tk.n_repeat == 0
    route = tk.predict_route(input_ids)
    assert (route == SKIP).sum().item() == 1
    assert (route == REPEAT).sum().item() == 0
    # boundaries forced EXEC
    assert route[0].item() == EXEC
    assert route[3].item() == EXEC


def test_topk_budget_compliance_over_budget(tiny_llama, input_ids):
    """At c⋆=1.25, exactly 1 of 4 layers should repeat."""
    tk = TopKMoLModel(tiny_llama, c_star=1.25, repeat_frac=0.25,
                       protect_boundaries=1)
    assert tk.n_skip == 0
    assert tk.n_repeat == 1
    route = tk.predict_route(input_ids)
    assert (route == SKIP).sum().item() == 0
    assert (route == REPEAT).sum().item() == 1


def test_topk_input_adaptive(tiny_llama, device):
    """Different inputs → potentially different routes (or at least same shape)."""
    tk = TopKMoLModel(tiny_llama, c_star=0.75, repeat_frac=0.0).to(device)
    a = torch.randint(0, 200, (1, 32)).to(device)
    b = torch.randint(0, 200, (1, 32)).to(device)
    ra = tk.predict_route(a)
    rb = tk.predict_route(b)
    assert ra.shape == (4,) == rb.shape
    # Both routes must satisfy the budget (1 skip each)
    assert (ra == SKIP).sum().item() == 1
    assert (rb == SKIP).sum().item() == 1


def test_topk_forward_shape(tiny_llama, input_ids):
    """Forward pass returns vocab-shaped logits same as the base."""
    tk = TopKMoLModel(tiny_llama, c_star=0.75, repeat_frac=0.0)
    out = tk(input_ids, labels=input_ids)
    assert out.logits.shape == (1, 32, tiny_llama.config.vocab_size)
    # loss should be finite
    assert torch.isfinite(out.loss).item()


def test_topk_assignment_correctness():
    """topk_assignment should put SKIP at high skip_pref layers and REPEAT at
    high repeat_pref layers, respecting boundary protection."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=6, num_attention_heads=4,
                       num_key_value_heads=4, max_position_embeddings=64)
    base = LlamaForCausalLM(cfg)
    tk = TopKMoLModel(base, c_star=0.667, repeat_frac=0.0, protect_boundaries=1)
    # n_skip = round(6 * 0.333) = 2, n_repeat = 0
    assert tk.n_skip == 2
    # Construct a synthetic logits tensor: large skip-pref at layers 2,3
    logits = torch.zeros(6, 3)
    logits[2, SKIP] = 5.0
    logits[3, SKIP] = 4.0
    logits[1, SKIP] = 1.0    # should NOT be picked over 2,3
    route, _ = tk.topk_assignment(logits)
    assert route[2].item() == SKIP
    assert route[3].item() == SKIP
    assert route[0].item() == EXEC   # boundary
    assert route[5].item() == EXEC   # boundary
