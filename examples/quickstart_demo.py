"""
quickstart_demo.py — programmatic example, 50 lines.

Builds a MoL model on Llama-2-7B at c⋆=0.75 and runs 100 Stage-A + 100
Stage-B steps. Intended as a smoke-test of the install (uses a real 7B model,
takes ~3 minutes on a 48 GB GPU).
"""
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mol import MoLModel
from mol import recovery as R
from mol.data import load_c4_sequences, load_wikitext2_eval
from mol.eval import wikitext_ppl

MODEL = "NousResearch/Llama-2-7b-hf"
DEVICE = "cuda"
C_STAR = 0.75

random.seed(0); torch.manual_seed(0)
print(f"Loading {MODEL}")
tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
L = base.config.num_hidden_layers
calib = load_c4_sequences(tok, 16, seed=0)
val = load_wikitext2_eval(tok, 8, seed=0)

R.add_lora(base, r=8)
mol = MoLModel(base).to(DEVICE)

print("BI warm-start")
bi = R.block_influence(base, calib, DEVICE)
K = L - int(round(L * C_STAR))
order = sorted(range(L), key=lambda i: bi[i].item())
order = [i for i in order if i not in (0, L - 1)]
sd = [0 if i in set(order[:K]) else 1 for i in range(L)]
R.pretrain_to_static(mol, sd, calib, DEVICE, steps=100)

print("Stage A (router, 100 steps)")
R.train_stage_a(mol, calib, target_compute=C_STAR, device=DEVICE,
                steps=100, log_every=50)

route = mol.extract_route(calib[0].unsqueeze(0).to(DEVICE))
print(f"Route: skip={sum(1 for m in route if m==0)} layers at "
      f"{[i for i,m in enumerate(route) if m==0]}")

print("Stage B (LoRA + KD, 100 steps)")
del mol; torch.cuda.empty_cache()
R.train_stage_b(base, calib, route, DEVICE, steps=100,
                kd_weight=1.0, log_every=50)

print("Eval (WikiText-2 PPL)")
R.set_lora(base, True)
restore = R.apply_route(base, route)
try:
    ppl = wikitext_ppl(base, val, DEVICE)
finally:
    restore()
print(f"PPL: {ppl:.2f}")
