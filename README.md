# MoL — Input-Adaptive Depth Routing with Mixture of Layers

Code release accompanying the paper
**“Input-Adaptive Depth Routing in Pretrained Large Language Models with Mixture of Layers”** (ICDM 2026 submission).

MoL replaces each decoder layer of a pretrained LLM with a routed module that
either **skips**, **executes**, or **repeats** the layer per input, under a
learned recurrent router. The result is a single multi-budget model whose
per-input depth allocation can be tuned at deployment without retraining.

This package provides:

* **`MoLModel`** — the default soft variant (Gumbel-Softmax router, Lagrangian
  budget hinge, KL prior toward a Block-Influence-informed distribution, BI
  warm-start). Used to produce all main-table results in the paper.
* **`TopKMoLModel`** — a deterministic variant that selects routes by global
  top-k assignment on per-layer logits, enforcing the compute budget by
  construction. Recommended for over-budget regimes and the input-adaptive
  routing study (Figure 6 in the paper).
* **Two-stage trainer** — `train_stage_a` trains the router; `train_stage_b`
  freezes the router and trains LoRA adapters with KD against the dense
  teacher.
* **Eval suite** — WikiText-2 perplexity and 8-task zero-shot accuracy
  (ARC-E/C, HellaSwag, Winogrande, OpenBookQA, PIQA, BoolQ, SciQ).

## Installation

The release targets Python 3.10+, PyTorch 2.x, and HuggingFace Transformers 5.x.

```bash
git clone <anonymous repo URL>
cd MoL
pip install -r requirements.txt
```

A single NVIDIA GPU with at least 24 GB of memory is needed for the default
7B-parameter setting (48 GB recommended). All experiments use `bfloat16`.

## Quick start

End-to-end training and evaluation on Llama-2-7B at 75 % of dense compute
(`L_eff = 24`):

```bash
python -m scripts.train --model NousResearch/Llama-2-7b-hf --c-star 0.75
```

The script:
1. Loads the base LLM and builds a calibration mix (C4 + ARC-Easy prompts).
2. Computes per-layer Block Influence and warm-starts the router with a
   BI-ordered static skip pattern.
3. Stage A — trains the router for 1000 steps under the Lagrangian budget
   hinge with τ-anneal from 1.0 → 0.3.
4. Stage B — freezes the router on the extracted deployment route, adds
   LoRA adapters (rank 8) on all linear projections of every decoder layer,
   trains for 1000 steps with `LM_CE + 1.0 · KL(student ‖ dense_teacher)`.
5. Evaluates on WikiText-2 (32 windows) and the 8-task accuracy suite.

Expected wall time on a single H100/RTX-PRO 6000 class GPU: **~20 minutes**.

### Variant: TopK-MoL

```bash
python -m scripts.train --model NousResearch/Llama-2-7b-hf --c-star 0.875 \
    --variant topk
```

TopK-MoL is recommended for budgets near `c⋆ = 1` (under- or over-budget)
where its structural budget-compliance is most useful. For aggressive
depth pruning (`c⋆ ≤ 0.75`) the default soft variant is stronger.

### Router-only ablation (no Stage B)

```bash
python -m scripts.train --c-star 0.75 --no-kd
```

## Module overview

| Path | Purpose |
| --- | --- |
| `mol/router.py` | `TriRouter` — recurrent shared 3-way router head |
| `mol/model.py` | `MoLModel` — soft variant with Gumbel routing |
| `mol/topk_model.py` | `TopKMoLModel` — deterministic top-k variant |
| `mol/recovery.py` | LoRA + KD + two-stage training (Stage A + Stage B) |
| `mol/data.py` | C4 / WikiText-2 / ARC / 8-task loaders |
| `mol/eval.py` | WikiText-2 PPL + multiple-choice accuracy |
| `scripts/train.py` | End-to-end trainer + evaluator |
| `tests/` | Unit tests (run with `pytest`) |
| `examples/` | Minimal example scripts |

## Programmatic usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mol import MoLModel
from mol import recovery as R
from mol.data import load_c4_sequences, load_wikitext2_eval
from mol.eval import wikitext_ppl

tok = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
base = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Llama-2-7b-hf", dtype=torch.bfloat16).to("cuda")

calib = load_c4_sequences(tok, n_sequences=32, seed=0)

R.add_lora(base, r=8)             # LoRA adapters for Stage B
mol = MoLModel(base).to("cuda")    # wrap with the router

# Stage A — router training under Lagrangian budget hinge
R.train_stage_a(mol, calib, target_compute=0.75, device="cuda", steps=1000)

# Extract the deployment route
route = mol.extract_route(calib[0].unsqueeze(0).to("cuda"))

# Stage B — LoRA + KD with the fixed route (KD teacher = base with LoRA off)
R.train_stage_b(base, calib, route, device="cuda", steps=1000, kd_weight=1.0)

# Eval at the deployed route
R.set_lora(base, True)
restore = R.apply_route(base, route)
ppl = wikitext_ppl(base, load_wikitext2_eval(tok), "cuda")
restore()
print(f"PPL: {ppl:.2f}")
```

## Reproducing the headline result

```bash
# 20 min on a single 48 GB GPU.
python -m scripts.train --model NousResearch/Llama-2-7b-hf --c-star 0.75 \
    --variant soft --steps 2000
```

This reproduces the `MoL-soft + KD` entry of Table I at `L_eff = 24`
on Llama-2-7B (target PPL ~11.7, 8-task acc ~57.3).

## Tests

```bash
pytest -q tests/
```

The test suite verifies module imports, router behaviour, TopK assignment
correctness, and a 5-step end-to-end smoke test (uses a tiny synthetic
Llama config — runs in under a minute, no large model download required).

## License

Released under the MIT License for academic review and follow-up research.

## Note for reviewers

This release contains the **core** code path used to produce Tables I/II
and Figures 5/6 in the submission, refactored for clarity. The full
research codebase (additional baselines, plotting utilities, raw logs)
is available upon de-anonymization.
