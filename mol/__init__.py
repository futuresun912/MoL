"""
MoL — Mixture of Layers for input-adaptive depth routing in pretrained LLMs.

Public API:

    from mol import MoLModel, TopKMoLModel
    from mol.recovery import add_lora, train_stage_a, train_stage_b
    from mol.eval import wikitext_ppl, eight_task_accuracy

Two routing variants are exposed:

* ``MoLModel`` — the default. Soft Gumbel-Softmax 3-way router, Lagrangian
  budget hinge, KL prior toward a Block-Influence-informed distribution, and
  temperature anneal. With ``add_lora`` + ``train_stage_{a,b}`` this is the
  strongest setup used to report Tables I and II in the paper.

* ``TopKMoLModel`` — a deterministic variant that uses global top-k selection
  on per-layer 3-way logits to enforce the budget by construction. Best for
  mild perturbations (c⋆ in [0.875, 1.25]) and for the input-adaptive Fig 6
  panels.

Reference: *Input-Adaptive Depth Routing in Pretrained Large Language Models
with Mixture of Layers*, anonymous, ICDM 2026.
"""
from mol.router import TriRouter
from mol.model import MoLModel
from mol.topk_model import TopKMoLModel, SKIP, EXEC, REPEAT

__all__ = [
    "TriRouter",
    "MoLModel",
    "TopKMoLModel",
    "SKIP", "EXEC", "REPEAT",
]
__version__ = "1.0.0-icdm2026"
