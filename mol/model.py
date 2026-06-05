"""
MoLModel — soft Mixture-of-Layers (the default).

Wraps a frozen HuggingFace causal LM with a recurrent trinary router. At each
decoder layer the router produces a 3-way distribution over {skip, exec,
repeat}; the forward blends the three outcomes via a (Gumbel-)softmax weight,
giving a differentiable computation graph. The MoLModel is trained with:

    Loss = LM_CE  +  Lagrangian budget hinge  +  μ · KL(π_θ ‖ π_prior)

(see ``mol.recovery.train_stage_a`` for the training loop). At deployment the
argmax route is read out and re-applied as a static layer-skip pattern.

This is the variant used to produce Tables I and II in the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mol.router import TriRouter, gumbel_softmax


class _RoutingState:
    """Mutable per-forward state shared between the wrapper layers."""
    def __init__(self):
        self.z = None
        self.router = None
        self.enabled = False
        self.mode = "train"                # "train" or "eval"
        self.tau = 1.0
        self.hard_train = False
        self.deterministic = False
        self.collect_logits = False
        self.gate_values = []              # list of [B, 3], detached
        self.gate_values_live = []         # list of [B, 3], with grad
        self.gate_logits = []              # list of [B, 3], raw router output
        self.static_dec = None             # warm-start blend target per layer
        self.blend_alpha = 1.0             # 0 = pure static, 1 = pure router


class _MoLLayer(nn.Module):
    """Wraps a single base decoder layer with a routed skip / exec / repeat
    transform driven by the shared TriRouter."""
    def __init__(self, base_layer, layer_idx: int, always_execute: bool = False):
        super().__init__()
        self.layer = base_layer
        self.layer_idx = layer_idx
        self.always_execute = always_execute
        object.__setattr__(self, "state", None)

    def forward(self, hidden_states, **kwargs):
        st = self.state
        if st is None or not st.enabled:
            return self.layer(hidden_states, **kwargs)

        # Per-layer 3-way logits
        pooled = hidden_states.mean(dim=1).to(st.z.dtype)
        logits, z_new = st.router(pooled, st.z, pos_idx=self.layer_idx)
        st.z = z_new
        st.gate_logits.append(logits)

        batch = hidden_states.shape[0]
        if self.always_execute:
            gate = torch.zeros(batch, 3, device=hidden_states.device)
            gate[:, 1] = 1.0
        elif st.deterministic:
            gate = F.one_hot(logits.detach().argmax(-1), 3).float()
        elif st.mode == "train":
            gate = gumbel_softmax(logits, tau=st.tau, hard=st.hard_train)
        else:
            gate = gumbel_softmax(logits, tau=st.tau, hard=True)

        # Warm-start blend: gate = (1-a) * static_onehot + a * gate
        if st.static_dec is not None and not self.always_execute:
            a = st.blend_alpha
            tgt = torch.tensor(st.static_dec[self.layer_idx],
                               device=hidden_states.device)
            so = F.one_hot(tgt, 3).float()
            gate = (1 - a) * so.unsqueeze(0).expand(batch, -1) + a * gate

        st.gate_values.append(gate.detach())
        st.gate_values_live.append(gate)

        # Fast path for deterministic single-sample inference
        if st.mode != "train" and batch == 1:
            d = int(gate[0].argmax().item())
            if d == 0:
                return hidden_states
            if d == 1:
                return self._exec(hidden_states, kwargs)
            return self._exec(self._exec(hidden_states, kwargs), kwargs)

        # Blended training path: weighted sum of {skip, exec, repeat}
        e = self._exec(hidden_states, kwargs)
        r = self._exec(e, kwargs)
        g = gate.to(hidden_states.dtype)
        return (g[:, 0:1].unsqueeze(-1) * hidden_states
                + g[:, 1:2].unsqueeze(-1) * e
                + g[:, 2:3].unsqueeze(-1) * r)

    def _exec(self, x, kwargs):
        # Disable cache for repeated/blended forwards
        ck = dict(kwargs); ck["use_cache"] = False; ck["past_key_values"] = None
        out = self.layer(x, **ck)
        return out[0] if isinstance(out, tuple) else out


class MoLModel(nn.Module):
    """Soft MoL wrapper for a frozen HuggingFace causal LM.

    Args:
        base_model:        HF causal LM whose decoder layers will be wrapped.
        router_dim:        recurrent state dimension (default 64).
        router_hidden:     router MLP width (default 128).
        protect_boundaries: number of boundary layers (start AND end) forced to EXEC.

    After construction, the base model's parameters are frozen; only
    ``self.router`` is trainable. To add LoRA on the base, call
    ``mol.recovery.add_lora(base_model, ...)`` **before** constructing
    ``MoLModel``, then ``set_lora(base_model, True)`` during Stage B.

    Training/inference toggles (set as attributes before forward):
        routing_enabled   bool  enable routing (else acts as the dense base)
        inference_mode    str   "train" or "eval"
        tau               float Gumbel temperature (anneal 1.0→0.3 in Stage A)
        hard_train        bool  apply STE on argmax (last 35% of Stage A)
        deterministic     bool  use argmax directly (eval-only)
        static_dec        list  per-layer {0,1,2} for warm-start blending
        blend_alpha       float 0=pure static, 1=pure router
    """
    def __init__(self,
                 base_model,
                 router_dim: int = 64,
                 router_hidden: int = 128,
                 protect_boundaries: int = 1):
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = base_model.config.hidden_size
        self.num_layers = base_model.config.num_hidden_layers
        self.protect_boundaries = protect_boundaries

        self.router = TriRouter(self.hidden_dim, router_dim, router_hidden,
                                num_positions=self.num_layers)
        # Move the router to the same device/dtype as the base LLM so the
        # forward path doesn't trip on cross-device tensor ops.
        _p = next(base_model.parameters())
        self.router = self.router.to(device=_p.device, dtype=_p.dtype)

        for p in self.base_model.parameters():
            p.requires_grad = False

        # Wrap each decoder layer
        self._state = _RoutingState()
        self._state.router = self.router
        layers = base_model.model.layers
        L = self.num_layers
        for i in range(L):
            always = (i < protect_boundaries) or (i >= L - protect_boundaries)
            w = _MoLLayer(layers[i], i, always_execute=always)
            w.state = self._state
            layers[i] = w

        # Public toggles (mirror onto state in forward)
        self.routing_enabled = True
        self.inference_mode = "train"
        self.tau = 1.0
        self.hard_train = False
        self.deterministic = False
        self.static_dec = None
        self.blend_alpha = 1.0
        # Last-forward bookkeeping
        self.last_gate_values = []
        self.last_gate_values_live = []
        self.last_gate_logits = []

    def forward(self, input_ids, **kwargs):
        st = self._state
        b = input_ids.shape[0]; dev = input_ids.device
        st.z = self.router.init_state(b, dev)
        st.gate_values = []; st.gate_values_live = []; st.gate_logits = []
        st.enabled = self.routing_enabled
        st.mode = self.inference_mode
        st.tau = self.tau
        st.hard_train = self.hard_train
        st.deterministic = self.deterministic
        st.static_dec = self.static_dec
        st.blend_alpha = self.blend_alpha

        out = self.base_model(input_ids, **kwargs)

        self.last_gate_values = list(st.gate_values)
        self.last_gate_values_live = list(st.gate_values_live)
        self.last_gate_logits = list(st.gate_logits)
        return out

    # ---- introspection helpers used by the training loop ----

    def get_trainable_params(self):
        """Iterator over router parameters (the only trainables on this module)."""
        return self.router.parameters()

    def ncost_live(self, gates_live):
        """Differentiable expected compute fraction (1.0 = full dense)."""
        L = self.num_layers
        cost = torch.stack([g[:, 1] + 2 * g[:, 2] for g in gates_live])   # (L, B)
        return cost.mean() / 1.0    # already a fraction of single-layer cost; mean over L,B

    @torch.no_grad()
    def avg_compute_fraction(self):
        """Realized compute fraction from the most recent forward (0..2; 1.0 = dense)."""
        if not self.last_gate_values:
            return 0.0
        g = torch.stack(self.last_gate_values)
        cost = g[..., 1] + 2 * g[..., 2]
        return cost.mean().item()

    @torch.no_grad()
    def extract_route(self, input_ids) -> list[int]:
        """Run a single deterministic forward and return the argmax route per layer."""
        prev = (self.inference_mode, self.deterministic, self.routing_enabled)
        self.inference_mode = "eval"
        self.deterministic = True
        self.routing_enabled = True
        try:
            self.eval()
            _ = self.forward(input_ids, use_cache=False)
            route = [int(g.argmax(-1).item()) for g in self.last_gate_logits]
        finally:
            (self.inference_mode, self.deterministic, self.routing_enabled) = prev
        # Force boundaries
        L = self.num_layers
        for i in range(self.protect_boundaries):
            route[i] = 1
            route[L - 1 - i] = 1
        return route
