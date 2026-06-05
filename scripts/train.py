#!/usr/bin/env python3
"""
train.py — train a MoL model end-to-end (Stage A + Stage B), evaluate on
WikiText-2 PPL and 8-task accuracy.

Default variant is the strongest soft MoL (Gumbel + Lagrangian + KL prior + BI
warm-start, then LoRA + KD recovery). Pass ``--variant topk`` to use TopK-MoL.

Example
-------
    # Default (strongest variant), Llama-2-7B at L_eff=24:
    python -m scripts.train --model NousResearch/Llama-2-7b-hf --c-star 0.75

    # TopK-MoL variant at L_eff=28:
    python -m scripts.train --model NousResearch/Llama-2-7b-hf --c-star 0.875 \
        --variant topk

    # No-KD ablation (router only, base frozen):
    python -m scripts.train --model NousResearch/Llama-2-7b-hf --c-star 0.75 \
        --no-kd
"""
import argparse
import random
import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from mol import MoLModel, TopKMoLModel
from mol import recovery as R
from mol.data import (load_c4_sequences, load_wikitext2_eval,
                       load_arc_easy_prompts)
from mol.eval import wikitext_ppl, eight_task_accuracy


def parse_args():
    p = argparse.ArgumentParser(description="MoL training + evaluation")
    p.add_argument("--model", default="NousResearch/Llama-2-7b-hf",
                    help="HuggingFace model id of the pretrained base LLM.")
    p.add_argument("--variant", choices=["soft", "topk"], default="soft",
                    help="MoL variant: soft (default, strongest) or topk.")
    p.add_argument("--c-star", type=float, default=0.75,
                    help="Target compute fraction (e.g. 0.75 = 25%% layers removed).")
    p.add_argument("--steps", type=int, default=2000,
                    help="Total training steps; split 50/50 between Stage A and B.")
    p.add_argument("--lora-r", type=int, default=8,
                    help="LoRA rank for the Stage B recovery adapters.")
    p.add_argument("--kd-weight", type=float, default=1.0,
                    help="Knowledge-distillation weight in Stage B.")
    p.add_argument("--no-kd", action="store_true",
                    help="Disable Stage B (router-only baseline).")
    p.add_argument("--n-calib", type=int, default=32,
                    help="Number of C4 calibration windows.")
    p.add_argument("--n-val", type=int, default=32,
                    help="Number of WikiText-2 windows for PPL.")
    p.add_argument("--n-task", type=int, default=100,
                    help="Items per zero-shot task.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda",
                    help="Device for forward/backward.")
    p.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    p.add_argument("--no-warm-start", action="store_true",
                    help="Skip BI warm-start (soft MoL only).")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    # -------- Load model and data --------
    print(f"Loading {args.model} (dtype={args.dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    L = base.config.num_hidden_layers
    L_eff = int(round(L * args.c_star))
    print(f"  L_full={L}  L_eff={L_eff}  c⋆={args.c_star:.4f}")

    print("Building calibration data ...")
    calib = load_c4_sequences(tok, n_sequences=args.n_calib, seed=args.seed)
    arc_prompts = load_arc_easy_prompts(tok, n_prompts=200, seed=args.seed)
    mix = calib + arc_prompts
    random.Random(args.seed).shuffle(mix)
    eval_windows = load_wikitext2_eval(tok, n_windows=args.n_val, seed=args.seed)

    # -------- Optionally add LoRA (Stage B uses it; safe to add up front) --------
    if not args.no_kd:
        n_lora = R.add_lora(base, r=args.lora_r)
        print(f"  added {n_lora} LoRA adapters (rank={args.lora_r})")

    # -------- Build MoL model --------
    if args.variant == "soft":
        mol = MoLModel(base).to(device)
        print("  variant: MoL (soft Gumbel + Lagrangian + KL prior)")
    else:
        mol = TopKMoLModel(base, c_star=args.c_star,
                            repeat_frac=max(0.0, args.c_star - 1.0)).to(device)
        print(f"  variant: TopK-MoL  (n_skip={mol.n_skip}, n_repeat={mol.n_repeat})")

    # -------- BI warm-start (soft only, c⋆ < 0.97) --------
    if args.variant == "soft" and not args.no_warm_start and args.c_star < 0.97:
        print("\nComputing block influence for BI warm-start ...")
        bi = R.block_influence(base, calib, device)
        K = L - L_eff
        bi_order = sorted(range(L), key=lambda i: bi[i].item())
        bi_order = [i for i in bi_order if i not in (0, L - 1)]
        skip = set(bi_order[:K])
        static_dec = [0 if i in skip else 1 for i in range(L)]
        print(f"  warm-start skip layers: {sorted(skip)[:8]}{'...' if K > 8 else ''}")
        R.pretrain_to_static(mol, static_dec, mix, device, steps=300, seed=args.seed)
        print("  warm-start done")

    # -------- Stage A: router training --------
    s_a = args.steps // 2
    s_b = args.steps - s_a
    print(f"\n[Stage A] router training, {s_a} steps")
    t0 = time.time()
    if args.variant == "soft":
        R.train_stage_a(mol, mix, target_compute=args.c_star, device=device,
                         steps=s_a, lam=25.0, seed=args.seed)
    else:
        R.train_topk_stage_a(mol, mix, device=device, steps=s_a, seed=args.seed)
    print(f"  Stage A: {time.time()-t0:.1f}s")

    # -------- Extract fixed route from the trained router --------
    print("\nExtracting deployment route ...")
    if args.variant == "soft":
        # Use first calibration sequence as exemplar prompt
        route = mol.extract_route(mix[0].unsqueeze(0).to(device))
    else:
        with torch.no_grad():
            route = mol.predict_route(mix[0].unsqueeze(0).to(device)).cpu().tolist()
    n_skip = sum(1 for m in route if m == 0)
    n_rep = sum(1 for m in route if m == 2)
    skip_idx = [i for i, m in enumerate(route) if m == 0]
    print(f"  route: {n_skip} skip + {n_rep} repeat + "
          f"{L - n_skip - n_rep} exec   skip-layers={skip_idx}")

    # -------- Stage B: LoRA + KD recovery --------
    if not args.no_kd:
        print(f"\n[Stage B] LoRA + KD, {s_b} steps  (kd_weight={args.kd_weight})")
        t0 = time.time()
        # Free the MoL wrapper so the base layers are addressable
        del mol; torch.cuda.empty_cache()
        R.train_stage_b(base, mix, route, device, steps=s_b,
                         kd_weight=args.kd_weight, seed=args.seed)
        print(f"  Stage B: {time.time()-t0:.1f}s")
    else:
        print("\n[Stage B] skipped (--no-kd)")

    # -------- Evaluation --------
    print("\n[Eval] WikiText-2 PPL + 8-task accuracy ...")
    R.set_lora(base, not args.no_kd)
    restore = R.apply_route(base, route)
    try:
        ppl = wikitext_ppl(base, eval_windows, device)
        accs = eight_task_accuracy(base, tok, device,
                                    n_per_task=args.n_task, seed=args.seed)
    finally:
        restore()

    print("\n" + "=" * 50)
    print(f"Model    : {args.model}")
    print(f"Variant  : {args.variant}{'+KD' if not args.no_kd else ' (no KD)'}")
    print(f"c⋆       : {args.c_star:.4f}  (L_eff={L_eff} of {L})")
    print(f"PPL      : {ppl:.3f}")
    print(f"Acc avg  : {accs['mean']:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
