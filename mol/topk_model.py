"""
TopKMoLModel — deterministic top-k variant of MoL.

Instead of per-layer Gumbel sampling + argmax (which can drift away from the
target budget at extraction), TopKMoLModel uses a **global top-k selection**
on per-layer 3-way logits to produce a budget-compliant route by construction:

    Given budget c⋆, n_skip = ⌊L · (1−c⋆+φ)⌋, n_repeat = ⌊L · φ⌋
    Realized compute = L − n_skip + n_repeat = L · c⋆ (exactly).

    Route assignment (greedy):
      1. Rank non-boundary layers by skip_pref = score[:, SKIP] - score[:, EXEC]
      2. Top n_skip → SKIP
      3. Rank remaining by repeat_pref = score[:, REPEAT] - score[:, EXEC]
      4. Top n_repeat → REPEAT, rest → EXEC

Training is two-pass per step:
  - Pass 1: dense forward; recurrent shared router collects per-layer logits.
  - Pass 2: routed forward (skip/exec/repeat) with the top-k assignment;
            LM loss flows back through a STE phantom on the soft policy.

This variant works best for mild perturbations (c⋆ in [0.875, 1.25]). For
aggressive depth pruning (c⋆ ≤ 0.75) the default ``MoLModel`` with full
warm-start + Lagrangian + KL prior is recommended.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from mol.router import TriRouter

SKIP, EXEC, REPEAT = 0, 1, 2


class _RoutedLayer(nn.Module):
    """Wraps a base layer to apply a fixed routing action with STE phantom."""
    def __init__(self, layer, action: int, layer_idx: int, phantom=None):
        super().__init__()
        self.layer = layer
        self.action = action
        self.layer_idx = layer_idx
        self.phantom = phantom

    def _ste_scale(self, x):
        if self.phantom is None:
            return x
        scale = self.phantom / self.phantom.detach().clamp(min=1e-8)
        return x * scale

    def forward(self, hidden_states, **kwargs):
        if self.action == SKIP:
            return self._ste_scale(hidden_states)
        x = self.layer(hidden_states, **kwargs)
        x = x[0] if isinstance(x, tuple) else x
        if self.action == EXEC:
            return self._ste_scale(x)
        # REPEAT
        ck = dict(kwargs); ck["past_key_values"] = None; ck["use_cache"] = False
        y = self.layer(x, **ck)
        y = y[0] if isinstance(y, tuple) else y
        return self._ste_scale(y)


class _ScoreCollectingLayer(nn.Module):
    """Pass-1 wrapper: invoke the router on layer input to collect a 3-way score,
    then run the layer dense (no routing)."""
    def __init__(self, layer, router: TriRouter, layer_idx: int,
                 score_buffer: list):
        super().__init__()
        self.layer = layer
        self.router = router
        self.layer_idx = layer_idx
        self.score_buffer = score_buffer
        self._state = None

    def _init_state(self, hidden_states):
        b = hidden_states.shape[0]
        dev = hidden_states.device
        if (self._state is None
                or self._state["b"] != b
                or self._state["dev"] != dev
                or self._state["fid"] != id(self.score_buffer)):
            self._state = {
                "b": b, "dev": dev,
                "z": self.router.init_state(b, dev),
                "fid": id(self.score_buffer),
            }

    def forward(self, hidden_states, **kwargs):
        self._init_state(hidden_states)
        pooled = hidden_states.mean(dim=1)
        logits, z_new = self.router(pooled, self._state["z"], self.layer_idx)
        self._state["z"] = z_new
        self.score_buffer.append(logits)
        return self.layer(hidden_states, **kwargs)


class TopKMoLModel(nn.Module):
    """Top-k Mixture-of-Layers wrapper.

    Args:
        base_model:        HF causal LM (frozen on construction).
        c_star:            target compute fraction. Determines n_skip and n_repeat.
        repeat_frac:       fraction of layers to repeat (default 0). For c⋆ > 1
                           this is typically c⋆ − 1.
        protect_boundaries: number of boundary layers forced to EXEC at both ends.
        router_dim,
        router_hidden:     same as TriRouter defaults.
        use_ste:           use straight-through estimator for backward (default True).
    """
    def __init__(self,
                 base_model,
                 c_star: float = 0.875,
                 repeat_frac: float = 0.0,
                 protect_boundaries: int = 1,
                 router_dim: int = 64,
                 router_hidden: int = 128,
                 use_ste: bool = True):
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = base_model.config.hidden_size
        self.L = base_model.config.num_hidden_layers
        self.protect_boundaries = protect_boundaries
        self.c_star = c_star
        self.use_ste = use_ste
        self.tau = 1.0

        L = self.L
        self.n_repeat = max(0, int(round(L * repeat_frac)))
        self.n_skip = max(0, self.n_repeat + int(round(L * (1.0 - c_star))))
        max_routable = L - 2 * protect_boundaries
        if self.n_skip + self.n_repeat > max_routable:
            scale = max_routable / max(self.n_skip + self.n_repeat, 1)
            self.n_skip = int(self.n_skip * scale)
            self.n_repeat = int(self.n_repeat * scale)
        self.realized_compute = L - self.n_skip + self.n_repeat
        self.realized_compute_frac = self.realized_compute / L

        self.router = TriRouter(self.hidden_dim, router_dim, router_hidden,
                                num_positions=L)
        # Match the router device/dtype to the base LLM.
        _p = next(base_model.parameters())
        self.router = self.router.to(device=_p.device, dtype=_p.dtype)

        for p in self.base_model.parameters():
            p.requires_grad = False

        # Last-forward bookkeeping
        self.last_route = None
        self.last_score_logits = None

    # ---- core algorithm ----

    def topk_assignment(self, score_logits: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Greedy top-k assignment from per-layer 3-way logits.

        Args:
            score_logits: [L, 3] logits (cols: SKIP, EXEC, REPEAT).
        Returns:
            route: [L] long tensor in {0, 1, 2}; exactly ``n_skip`` SKIPs and
                   ``n_repeat`` REPEATs (subject to boundary protection).
            info:  dict with intermediate preference scores.
        """
        L = self.L
        skip_pref   = score_logits[:, SKIP]   - score_logits[:, EXEC]
        repeat_pref = score_logits[:, REPEAT] - score_logits[:, EXEC]

        mask = torch.zeros(L, dtype=torch.bool, device=score_logits.device)
        for i in range(self.protect_boundaries):
            mask[i] = True
            mask[L - 1 - i] = True
        skip_pref_masked = skip_pref.masked_fill(mask, -1e9)
        repeat_pref_masked = repeat_pref.masked_fill(mask, -1e9)

        route = torch.full((L,), EXEC, dtype=torch.long, device=score_logits.device)
        if self.n_skip > 0:
            _, sidx = skip_pref_masked.topk(self.n_skip)
            route[sidx] = SKIP
        if self.n_repeat > 0:
            r_masked = repeat_pref_masked.masked_fill(route == SKIP, -1e9)
            _, ridx = r_masked.topk(self.n_repeat)
            route[ridx] = REPEAT
        return route, {"skip_pref": skip_pref, "repeat_pref": repeat_pref}

    def _soft_one_hot(self, score_logits: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        """STE phantom: forward = hard one-hot, backward = soft gradient."""
        hard = F.one_hot(route, num_classes=3).to(score_logits.dtype)
        if not self.use_ste:
            return hard
        soft = F.softmax(score_logits / max(self.tau, 1e-4), dim=-1)
        return hard.detach() + (soft - soft.detach())

    # ---- two-pass forward ----

    def _collect_scores(self, input_ids, kwargs) -> torch.Tensor:
        """Pass 1: dense forward; collect [L, 3] per-layer router logits."""
        base = self.base_model
        original = list(base.model.layers)
        score_buffer = []
        wrapped = [_ScoreCollectingLayer(orig, self.router, i, score_buffer)
                   for i, orig in enumerate(original)]
        base.model.layers = nn.ModuleList(wrapped)
        try:
            ctx = torch.enable_grad() if self.training else torch.no_grad()
            with ctx:
                _ = base(input_ids, **{**kwargs, "labels": None, "use_cache": False})
        finally:
            base.model.layers = nn.ModuleList(original)
        return torch.stack(score_buffer, dim=0).mean(dim=1)   # [L, 3]

    def _forward_routed(self, input_ids, route, kwargs, phantom=None):
        """Pass 2: routed forward with the given route. Optional STE phantom."""
        base = self.base_model
        original = list(base.model.layers)
        wrapped = []
        for i, orig in enumerate(original):
            ph = phantom[i, int(route[i].item())] if phantom is not None else None
            wrapped.append(_RoutedLayer(orig, int(route[i].item()), i, ph))
        base.model.layers = nn.ModuleList(wrapped)
        try:
            out = base(input_ids, **kwargs)
        finally:
            base.model.layers = nn.ModuleList(original)
        return out

    def forward(self, input_ids, labels=None, **kwargs):
        # Pass 1: collect router scores
        score_logits = self._collect_scores(input_ids, kwargs)        # [L, 3]
        self.last_score_logits = score_logits.detach()

        # Top-k assignment
        route, _ = self.topk_assignment(score_logits)
        self.last_route = route.detach()

        # STE phantom for backprop
        phantom = self._soft_one_hot(score_logits, route)             # [L, 3]

        # Pass 2: routed forward (loss-bearing)
        out = self._forward_routed(input_ids, route,
                                    {**kwargs, "labels": labels},
                                    phantom=phantom)
        return out

    @torch.no_grad()
    def predict_route(self, input_ids) -> torch.Tensor:
        """Return the deployment route for a prompt (Pass 1 + top-k, no Pass 2)."""
        score_logits = self._collect_scores(input_ids, {})
        route, _ = self.topk_assignment(score_logits)
        return route

    def get_trainable_params(self):
        return self.router.parameters()
