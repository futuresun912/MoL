"""
TriRouter — recurrent shared 3-way (skip / exec / repeat) router head.

The router takes a per-layer pooled hidden state plus a recurrent state ``z``
and emits a 3-way logit per layer. Used by both ``MoLModel`` (soft policy via
Gumbel-Softmax) and ``TopKMoLModel`` (hard top-k assignment).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0,
                    hard: bool = False) -> torch.Tensor:
    """3-class Gumbel-Softmax. In inference mode (no grad) returns a deterministic
    one-hot argmax; otherwise samples and optionally applies straight-through."""
    if not logits.requires_grad:
        idx = logits.argmax(dim=-1)
        return F.one_hot(idx, num_classes=3).float()
    u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    g = -torch.log(-torch.log(u))
    y = F.softmax((logits + g) / tau, dim=-1)
    if hard:
        idx = y.argmax(dim=-1)
        y_hard = F.one_hot(idx, num_classes=3).float()
        return y_hard - y.detach() + y     # straight-through estimator
    return y


class TriRouter(nn.Module):
    """Recurrent shared trinary router.

    Args:
        hidden_dim:    LLM hidden size (e.g. 4096 for Llama-2-7B).
        router_dim:    dimension of the recurrent state.
        router_hidden: width of the router MLP.
        num_positions: number of layer-id slots (= L for layer granularity).
    """
    def __init__(self, hidden_dim: int, router_dim: int = 64,
                 router_hidden: int = 128, num_positions: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.router_dim = router_dim
        self.router_hidden = router_hidden
        self.num_positions = num_positions

        self.input_proj = nn.Linear(hidden_dim + router_dim, router_hidden)
        self.hidden_layer = nn.Linear(router_hidden, router_hidden)
        self.gate_head = nn.Linear(router_hidden, 3)
        self.state_head = nn.Linear(router_hidden, router_dim)

        # Favor execute at init: skip/repeat disfavored
        nn.init.constant_(self.gate_head.bias[0], -1.0)
        nn.init.constant_(self.gate_head.bias[1], 1.0)
        nn.init.constant_(self.gate_head.bias[2], -1.0)

        self.layer_bias = nn.Parameter(torch.zeros(num_positions, 3))
        self._alpha_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        """Damping of the recurrent state, in (0, 1) via sigmoid."""
        return torch.sigmoid(self._alpha_raw)

    def forward(self, pooled_hidden: torch.Tensor, z_prev: torch.Tensor,
                pos_idx: int = 0):
        """
        Args:
            pooled_hidden: (batch, hidden_dim) pooled layer input.
            z_prev:        (batch, router_dim) prior recurrent state.
            pos_idx:       layer index for per-position bias.

        Returns:
            logits: (batch, 3) raw 3-way logits.
            z_new:  (batch, router_dim) updated recurrent state.
        """
        x = torch.cat([pooled_hidden, z_prev], dim=-1)
        h = F.silu(self.input_proj(x))
        h = F.silu(self.hidden_layer(h))
        logits = self.gate_head(h)
        if pos_idx < self.num_positions:
            logits = logits + self.layer_bias[pos_idx]
        z_new = self.alpha * torch.tanh(self.state_head(h)) + (1 - self.alpha) * z_prev
        return logits, z_new

    def init_state(self, batch_size: int, device) -> torch.Tensor:
        """Initialize the recurrent state to zeros for a new forward."""
        return torch.zeros(batch_size, self.router_dim, device=device)
