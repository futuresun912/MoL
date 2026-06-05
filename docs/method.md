# Method note (release)

This note summarizes the MoL training algorithm at the level needed to use the
release. Full derivations and ablations are in the paper.

## Trinary routing

For a base LLM with `L` decoder layers, MoL associates each layer `ℓ ∈ [0, L)`
with a discrete decision `m_ℓ ∈ {skip, exec, repeat}`. The routed forward
applies, per layer:

* `skip(x)   = x`                      (identity)
* `exec(x)   = f_ℓ(x)`                 (one base-layer call)
* `repeat(x) = f_ℓ(f_ℓ(x))`            (apply the layer twice)

Total compute over a forward is `sum_ℓ c(m_ℓ)` with `c(skip)=0, c(exec)=1,
c(repeat)=2`. We define the realized compute fraction `c̄ = (Σ c(m_ℓ)) / L`.
The user picks a target `c⋆` (e.g. 0.75 = 25% layers removed).

A small recurrent shared router (`TriRouter`) produces per-layer 3-way logits
from the layer-input hidden state and a recurrent context `z_ℓ`.

## Default variant — soft MoL

Training proceeds in two decoupled stages.

**Stage A — router only.** LoRA disabled; only the router updates. At each
step:
```
gate_ℓ ~ Gumbel-Softmax(logits_ℓ, τ)        (τ anneals 1.0 → 0.3)
hidden = gate_ℓ[0] · x + gate_ℓ[1] · f(x) + gate_ℓ[2] · f(f(x))
Loss_A = LM_CE  +  λ · ReLU(c̄ − c⋆)²
```
Optionally a KL prior pulls the router toward a Block-Influence-informed
3-way distribution `π_ℓ`. For the last 35 % of steps we set `hard_train=True`,
using straight-through estimation on argmax samples. Before Stage A we
warm-start with 300 supervised steps where the router is trained via
cross-entropy to predict a BI-ordered static skip pattern.

**Stage B — LoRA recovery.** Extract the deployment route by running the
router once in deterministic mode (`extract_route`). Freeze the router and
apply the route as a fixed layer-surgery pattern (`apply_route`). Train LoRA
adapters on every linear projection of every layer with KD from the dense
(LoRA-off) teacher:
```
Loss_B = LM_CE(student)  +  kd_weight · KL(student ‖ teacher)
```

## Variant — TopK-MoL

The soft variant's argmax route can drift away from the target budget. The
TopK variant enforces the budget by construction:

```
Pass 1 (dense forward): collect score_ℓ ∈ ℝ³ from the router at each layer.
Pass 2 (routed):       route ← topk_assignment(score, n_skip, n_repeat)
                       apply route, compute LM_CE on the routed output.
n_skip   = round(L · (1 − c⋆ + φ))
n_repeat = round(L · φ)             (φ ≥ 0; φ = max(0, c⋆ − 1))
```

Backward uses a straight-through phantom (`hard + (soft − soft.detach())`).
Stage B is unchanged.

## Why two stages

Joint training of router + LoRA + KD has a known failure mode: LoRA can
absorb the impact of *any* route, so the router is rewarded for picking
routes that are *easy for LoRA to repair* rather than routes that are
intrinsically least-disruptive. Empirically this collapses the router to
early-layer routing (layers 1–6), where LoRA capacity is most effective,
even though late layers are the ones with low Block Influence and are
genuinely the right ones to skip.

Decoupling Stage A and Stage B forces the router to commit to a route
based purely on LM loss, then lets LoRA adapt FROM the fixed route.

## Choosing a budget

* `c⋆ ≤ 0.75` (aggressive depth pruning, `K ≥ 8` on a 32-layer LLM):
  use the default soft variant with BI warm-start and the full
  Lagrangian+KL recipe.
* `c⋆ ∈ [0.875, 1.25]` (mild perturbations, under- or over-budget):
  `TopKMoLModel` is competitive and has exact budget compliance.
