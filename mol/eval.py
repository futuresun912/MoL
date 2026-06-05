"""
Evaluation — WikiText-2 perplexity and 8-task zero-shot accuracy.
"""
import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from mol.data import load_mc_task, AVAILABLE_TASKS


@torch.no_grad()
def wikitext_ppl(model, eval_windows: List[torch.Tensor], device) -> float:
    """Standard PPL on tokenized WikiText-2 windows."""
    model.eval()
    losses = []
    for ids in eval_windows:
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        out = model(ids, labels=ids, use_cache=False)
        losses.append(out.loss.item())
    return min(math.exp(sum(losses) / max(len(losses), 1)), 1e8)


@torch.no_grad()
def mc_accuracy(model, tokenizer, items: List[Tuple[str, list, int]],
                device) -> float:
    """Length-normalized log-likelihood multiple-choice scoring."""
    model.eval()
    correct = 0
    for prompt, choices, label in items:
        scores = []
        for ch in choices:
            full = tokenizer(prompt + " " + ch, return_tensors="pt").input_ids.to(device)
            plen = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            if full.shape[1] <= plen:
                scores.append(-1e9); continue
            out = model(full, use_cache=False)
            targets = full[0, plen:]
            pred = out.logits[0, plen - 1: full.shape[1] - 1, :]
            ll = -F.cross_entropy(pred.float(), targets, reduction="sum").item()
            scores.append(ll / max(1, full.shape[1] - plen))
        if scores.index(max(scores)) == label:
            correct += 1
    return 100.0 * correct / max(len(items), 1)


@torch.no_grad()
def eight_task_accuracy(model, tokenizer, device, n_per_task: int = 100,
                        seed: int = 0, tasks: List[str] = AVAILABLE_TASKS,
                        verbose: bool = True) -> Dict[str, float]:
    """Run all 8 multiple-choice tasks. Returns a dict ``{task: accuracy}``
    plus the average under key ``'mean'``.
    """
    out = {}
    for t in tasks:
        items = load_mc_task(t, n_per_task, seed)
        acc = mc_accuracy(model, tokenizer, items, device)
        out[t] = acc
        if verbose:
            print(f"  {t:<16}  acc={acc:5.2f}  (n={len(items)})")
    out["mean"] = sum(out[t] for t in tasks) / len(tasks)
    if verbose:
        print(f"  {'mean':<16}  acc={out['mean']:5.2f}")
    return out
