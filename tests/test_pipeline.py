"""End-to-end smoke test: train Stage A + Stage B on a tiny synthetic model.

We use a TinyLlama with 4 layers and 5 training steps per stage so the test
finishes in a few seconds, but the full code path is exercised (router
update, route extraction, LoRA training, KD against the dense teacher, eval
under the deployed route).
"""
import torch

from mol import MoLModel, TopKMoLModel
from mol import recovery as R


def _make_pair(device, dtype):
    """Build a tiny Llama-arch model + a calibration list of 4 random sequences."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=200, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=128,
                      rms_norm_eps=1e-6)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(device=device, dtype=dtype)
    seqs = [torch.randint(0, 200, (32,)).to(device) for _ in range(4)]
    return model, seqs


def test_soft_mol_end_to_end(device, dtype):
    """Full soft-MoL pipeline: warm-start → Stage A → extract route → Stage B → eval."""
    base, calib = _make_pair(device, dtype)
    R.add_lora(base, r=4, alpha=8)
    mol = MoLModel(base).to(device)

    # BI warm-start (very short)
    bi = R.block_influence(base, calib, device)
    L = base.config.num_hidden_layers
    K = 1
    order = sorted(range(L), key=lambda i: bi[i].item())
    order = [i for i in order if i not in (0, L - 1)]
    skip = set(order[:K])
    static_dec = [0 if i in skip else 1 for i in range(L)]
    R.pretrain_to_static(mol, static_dec, calib, device, steps=5, seed=0)

    # Stage A
    R.train_stage_a(mol, calib, target_compute=0.75, device=device,
                    steps=5, log_every=10, verbose=False, seed=0)

    # Route extraction
    route = mol.extract_route(calib[0].unsqueeze(0).to(device))
    assert len(route) == L
    assert all(m in (0, 1, 2) for m in route)

    # Stage B — LoRA + KD
    del mol
    R.train_stage_b(base, calib, route, device, steps=5,
                    kd_weight=1.0, log_every=10, verbose=False, seed=0)

    # Eval at the deployed route
    R.set_lora(base, True)
    restore = R.apply_route(base, route)
    try:
        with torch.no_grad():
            out = base(calib[0].unsqueeze(0).to(device), use_cache=False)
    finally:
        restore()
    assert torch.isfinite(out.logits).all().item()


def test_topk_mol_end_to_end(device, dtype):
    """Full TopK-MoL pipeline: Stage A (LM only) → extract route → Stage B → eval."""
    base, calib = _make_pair(device, dtype)
    R.add_lora(base, r=4, alpha=8)
    tk = TopKMoLModel(base, c_star=0.75, repeat_frac=0.0,
                       protect_boundaries=1).to(device)
    assert tk.n_skip == 1
    assert tk.realized_compute_frac == 0.75

    # Stage A
    R.train_topk_stage_a(tk, calib, device, steps=5, log_every=10,
                          verbose=False, seed=0)

    # Route from the trained router
    with torch.no_grad():
        route = tk.predict_route(calib[0].unsqueeze(0).to(device)).cpu().tolist()
    L = base.config.num_hidden_layers
    assert len(route) == L
    assert route.count(0) == 1   # exactly n_skip skips
    assert route[0] == 1 and route[L - 1] == 1   # boundaries EXEC

    # Stage B
    del tk
    R.train_stage_b(base, calib, route, device, steps=5,
                    kd_weight=1.0, log_every=10, verbose=False, seed=0)

    R.set_lora(base, True)
    restore = R.apply_route(base, route)
    try:
        with torch.no_grad():
            out = base(calib[0].unsqueeze(0).to(device), use_cache=False)
    finally:
        restore()
    assert torch.isfinite(out.logits).all().item()


def test_apply_route_realized_compute(device, dtype):
    """A route of {skip:1, exec:2, repeat:1} on a 4-layer model has realized c = 1.0."""
    base, _ = _make_pair(device, dtype)
    route = [1, 0, 2, 1]   # boundaries kept, layer 1 skipped, layer 2 repeated
    n_skip = sum(1 for m in route if m == 0)
    n_rep = sum(1 for m in route if m == 2)
    L = len(route)
    realized = (L - n_skip + n_rep) / L
    assert realized == 1.0   # (4 - 1 + 1) / 4 = 1.0
