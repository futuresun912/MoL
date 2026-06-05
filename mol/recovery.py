"""
Recovery utilities — LoRA + KD + two-stage training.

The trainer is *separable*. Stage A trains the router only (LoRA off). Stage B
freezes the router (or its post-extracted route) and trains LoRA only against
the dense teacher via KD. Decoupling Stage A and Stage B is critical: joint
training of router + LoRA + KD collapses the router to easy-to-compensate
positions (early layers) rather than genuinely redundant ones (late layers).

See ``train_stage_a`` and ``train_stage_b`` for the loss functions used in
the paper.
"""
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- LoRA -----------------

class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank delta.

    A is initialized ~ N(0, 0.01); B is zero (so initial Δ is zero and the
    LoRA-on model exactly matches the original at start).
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.enabled = True
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B)
            out = out + self.scaling * delta.to(out.dtype)
        return out


_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj")


def add_lora(base_model, r: int = 8, alpha: int = 16, targets=_TARGETS) -> int:
    """Replace target Linears in every decoder layer with LoRALinear in place.

    Must be called BEFORE wrapping ``base_model`` with ``MoLModel`` or
    ``TopKMoLModel`` (the wrapper will then freeze all base params including
    LoRA; you re-enable LoRA params for Stage B via ``set_lora_trainable``).
    """
    n = 0
    for layer in base_model.model.layers:
        dl = getattr(layer, "layer", layer)
        for sub_name in ("self_attn", "mlp"):
            sub = getattr(dl, sub_name, None)
            if sub is None:
                continue
            for t in targets:
                lin = getattr(sub, t, None)
                if isinstance(lin, nn.Linear):
                    setattr(sub, t, LoRALinear(lin, r, alpha).to(lin.weight.device))
                    n += 1
    return n


def iterate_lora(base_model):
    for m in base_model.modules():
        if isinstance(m, LoRALinear):
            yield m


def lora_parameters(base_model):
    """Iterator over LoRA tensors (lora_A, lora_B for every LoRALinear)."""
    for m in iterate_lora(base_model):
        yield m.lora_A
        yield m.lora_B


def set_lora(base_model, enabled: bool):
    """Enable/disable the LoRA delta on every LoRALinear (forward only — params unchanged)."""
    for m in iterate_lora(base_model):
        m.enabled = enabled


def set_lora_trainable(base_model, trainable: bool):
    """Set ``requires_grad`` on every LoRA parameter."""
    for p in lora_parameters(base_model):
        p.requires_grad_(trainable)


# ----------------- KD loss -----------------

def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            temperature: float = 1.0) -> torch.Tensor:
    """Forward KL on next-token distributions."""
    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.softmax(teacher_logits.float() / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


# ----------------- Block influence (for BI warm-start) -----------------

@torch.no_grad()
def block_influence(base_model, calibration_seqs, device) -> torch.Tensor:
    """Per-layer cosine-distance change of hidden states (ShortGPT-style BI).

    Lower BI = more "redundant" layer (safer to skip). Used to warm-start the
    router via ``pretrain_to_static`` before Stage A.
    """
    L = base_model.config.num_hidden_layers
    influence = torch.zeros(L, device=device, dtype=torch.float32)
    n_seqs = 0
    for ids in calibration_seqs:
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        out = base_model(ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states                                  # len L+1
        for i in range(L):
            x_in = hs[i].float()
            x_out = hs[i + 1].float()
            cos = F.cosine_similarity(x_in, x_out, dim=-1).mean()
            influence[i] += (1.0 - cos)
        n_seqs += 1
    return influence / max(n_seqs, 1)


# ----------------- Soft MoL Stage A (router only, no LoRA) -----------------

def pretrain_to_static(mol_model, static_dec, train_seqs, device,
                        steps: int = 300, lr: float = 1e-3, seed: int = 0):
    """Warm-start: supervised CE training of the router to predict ``static_dec``
    (BI-ordered skip pattern) at every layer position. Recommended before
    Stage A for aggressive sparsity (c⋆ ≤ 0.85).
    """
    torch.manual_seed(seed)
    mol_model.routing_enabled = True
    mol_model.inference_mode = "train"
    mol_model.hard_train = False
    router_params = list(mol_model.get_trainable_params())
    opt = torch.optim.AdamW(router_params, lr=lr)
    tgt = torch.tensor(static_dec, device=device, dtype=torch.long)
    for step in range(steps):
        ids = train_seqs[step % len(train_seqs)].to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        mol_model(ids, use_cache=False)
        lg = mol_model.last_gate_logits
        n = min(len(lg), len(static_dec))
        loss = sum(F.cross_entropy(lg[p], tgt[p:p + 1]) for p in range(n)) / n
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(router_params, 1.0)
        opt.step()


def train_stage_a(mol_model, train_seqs, target_compute: float, device,
                  steps: int = 1000, lam: float = 25.0,
                  hard_frac: float = 0.35, lr: float = 1e-3, seed: int = 0,
                  log_every: int = 200, verbose: bool = True):
    """Stage A — router-only training under Lagrangian budget hinge.

    Loss = LM_CE + λ · max(0, c_live − target)²    (one-sided hinge)

    Temperature anneal τ: 1.0 → 0.3 across ``steps``; hard-route STE
    (``hard_train=True``) enabled for the last ``hard_frac`` fraction of steps.

    For ``MoLModel`` only. ``TopKMoLModel`` uses ``train_topk_stage_a`` instead.
    """
    torch.manual_seed(seed)
    mol_model.routing_enabled = True
    mol_model.inference_mode = "train"
    set_lora(mol_model.base_model, False)
    set_lora_trainable(mol_model.base_model, False)

    router_params = list(mol_model.get_trainable_params())
    opt = torch.optim.AdamW(router_params, lr=lr, weight_decay=0.01)
    hard_start = int(steps * (1 - hard_frac))
    ramp = int(steps * 0.6)
    t0 = time.time()
    for step in range(steps):
        mol_model.hard_train = (step >= hard_start)
        mol_model.tau = 1.0 + (0.3 - 1.0) * (step / max(steps - 1, 1))
        tgt_t = 1.0 - (1.0 - target_compute) * min(1.0, step / max(ramp, 1))

        ids = train_seqs[step % len(train_seqs)].to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        out = mol_model(ids, labels=ids, use_cache=False)
        gl = mol_model.last_gate_values_live
        c_live = mol_model.ncost_live(gl)
        loss = out.loss + lam * torch.clamp(c_live - tgt_t, min=0.0)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(router_params, 1.0)
        opt.step()
        if verbose and step % log_every == 0:
            print(f"    [A] step {step:4d}  tau={mol_model.tau:.3f}  "
                  f"hard={mol_model.hard_train}  LM={out.loss.item():.3f}  "
                  f"c={float(c_live.item()):.3f}  tgt={tgt_t:.3f}")
    return time.time() - t0


# ----------------- TopK-MoL Stage A (LM-only) -----------------

def train_topk_stage_a(topk_model, train_seqs, device,
                        steps: int = 1000, lr: float = 1e-3,
                        seed: int = 0, log_every: int = 200,
                        verbose: bool = True):
    """Stage A for ``TopKMoLModel`` — LM-only training. The budget is enforced
    structurally by the top-k assignment, so no Lagrangian is needed.
    """
    torch.manual_seed(seed)
    set_lora(topk_model.base_model, False)
    set_lora_trainable(topk_model.base_model, False)
    router_params = list(topk_model.get_trainable_params())
    opt = torch.optim.AdamW(router_params, lr=lr, weight_decay=0.01)
    t0 = time.time()
    for step in range(steps):
        ids = train_seqs[step % len(train_seqs)].to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        ids = ids[:1]
        out = topk_model(ids, labels=ids)
        loss = out.loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(router_params, 1.0)
        opt.step()
        if verbose and step % log_every == 0:
            print(f"    [A] step {step:4d}  LM={loss.item():.3f}")
    return time.time() - t0


# ----------------- Stage B — LoRA + KD with fixed route -----------------

def train_stage_b(base_model, train_seqs, fixed_route, device,
                  steps: int = 1000, kd_weight: float = 1.0,
                  lr: float = 2e-4, seed: int = 0, log_every: int = 200,
                  verbose: bool = True):
    """Stage B — train LoRA only, with KD from the dense (LoRA-off) teacher.

    ``fixed_route`` is a list of length L with values in {0=SKIP, 1=EXEC, 2=REPEAT}
    (e.g. obtained from ``MoLModel.extract_route(...)`` or
    ``TopKMoLModel.predict_route(...)``). The route is applied via ``apply_route``
    so that training and evaluation use the SAME deterministic skip pattern.

    Returns wall-clock training seconds.
    """
    torch.manual_seed(seed + 1)
    set_lora_trainable(base_model, True)
    set_lora(base_model, True)
    lp = list(lora_parameters(base_model))
    opt = torch.optim.AdamW(lp, lr=lr, weight_decay=0.0)

    skip_set = {i for i, m in enumerate(fixed_route) if m == 0}

    t0 = time.time()
    for step in range(steps):
        ids = train_seqs[step % len(train_seqs)].to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        # Teacher: LoRA OFF + dense
        set_lora(base_model, False)
        with torch.no_grad():
            teacher = base_model(ids, use_cache=False).logits
        set_lora(base_model, True)

        # Student: routed with LoRA ON
        restore = apply_route(base_model, fixed_route)
        try:
            out = base_model(ids, labels=ids, use_cache=False)
        finally:
            restore()
        loss = out.loss + kd_weight * kd_loss(out.logits, teacher)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(lp, 1.0)
        opt.step()
        if verbose and step % log_every == 0:
            print(f"    [B] step {step:4d}  LM={out.loss.item():.3f}  "
                  f"total={loss.item():.3f}")
    return time.time() - t0


# ----------------- Apply a fixed route via in-place layer surgery -----------------

def apply_route(base_model, route):
    """Apply a fixed deployment route to ``base_model``. Returns a callable
    ``restore()`` that undoes the change. Layers with route value 0 are
    skipped (replaced with identity); value 1 executes normally; value 2
    runs the layer twice in sequence.

    Note: this performs in-place surgery on ``base_model.model.layers`` and
    must be undone by calling the returned closure.
    """
    original_layers = list(base_model.model.layers)
    wrapped = []
    for i, lyr in enumerate(original_layers):
        m = int(route[i]) if not isinstance(route[i], int) else route[i]
        if m == 0:
            wrapped.append(_IdentityLayer(lyr))
        elif m == 2:
            wrapped.append(_RepeatLayer(lyr))
        else:
            wrapped.append(lyr)
    base_model.model.layers = nn.ModuleList(wrapped)

    def restore():
        base_model.model.layers = nn.ModuleList(original_layers)
    return restore


class _IdentityLayer(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.layer = base_layer
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _RepeatLayer(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.layer = base_layer
    def forward(self, hidden_states, **kwargs):
        x = self.layer(hidden_states, **kwargs)
        x = x[0] if isinstance(x, tuple) else x
        ck = dict(kwargs); ck["past_key_values"] = None; ck["use_cache"] = False
        y = self.layer(x, **ck)
        return y[0] if isinstance(y, tuple) else y
