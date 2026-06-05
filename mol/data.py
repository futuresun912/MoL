"""
Minimal data loaders for calibration and zero-shot evaluation.

To keep the release small we ship just the loaders we actually use:
  - ``load_c4_sequences``    : tokenized C4 training sequences for calibration
  - ``load_wikitext2_eval``  : tokenized WikiText-2 windows for PPL eval
  - ``load_arc_easy_prompts``: ARC-Easy QA prompts for diverse calibration
  - ``load_mc_task``         : multiple-choice items for the 8-task eval suite
"""
from typing import List, Tuple

import torch
from datasets import load_dataset


# ----------------- Calibration sequences -----------------

def load_c4_sequences(tokenizer, n_sequences: int = 32, seq_len: int = 2048,
                      split: str = "train", seed: int = 0) -> List[torch.Tensor]:
    """Load ``n_sequences`` tokenized C4 windows. Returns a list of long tensors,
    each of length ``seq_len``."""
    import random
    rng = random.Random(seed)
    ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
    seqs = []
    accumulator: List[int] = []
    for ex in ds:
        toks = tokenizer.encode(ex["text"], add_special_tokens=False)
        accumulator.extend(toks)
        while len(accumulator) >= seq_len:
            seqs.append(torch.tensor(accumulator[:seq_len], dtype=torch.long))
            accumulator = accumulator[seq_len:]
            if len(seqs) >= n_sequences:
                rng.shuffle(seqs)
                return seqs[:n_sequences]
    return seqs


def load_wikitext2_eval(tokenizer, n_windows: int = 32,
                        seq_len: int = 2048, seed: int = 0) -> List[torch.Tensor]:
    """Load ``n_windows`` WikiText-2 raw test windows for perplexity evaluation."""
    import random
    rng = random.Random(seed)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    toks = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(toks) - seq_len, seq_len):
        chunks.append(torch.tensor(toks[i:i + seq_len], dtype=torch.long))
    rng.shuffle(chunks)
    return chunks[:n_windows]


# ----------------- ARC-Easy training prompts -----------------

def load_arc_easy_prompts(tokenizer, n_prompts: int = 200,
                           max_len: int = 256, seed: int = 0) -> List[torch.Tensor]:
    """ARC-Easy training questions as tokenized prompts for diverse calibration."""
    import random
    rng = random.Random(seed)
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    items = list(ds); rng.shuffle(items)
    prompts = []
    for it in items:
        q = it["question"]
        toks = tokenizer.encode(q, add_special_tokens=False)[:max_len]
        if len(toks) < 8:
            continue
        prompts.append(torch.tensor(toks, dtype=torch.long))
        if len(prompts) >= n_prompts:
            break
    return prompts


# ----------------- 8-task multiple-choice eval items -----------------

_TASK_LOADERS = {}


def _register(name):
    def deco(fn):
        _TASK_LOADERS[name] = fn
        return fn
    return deco


@_register("arc_easy")
def _arc_easy(n: int, seed: int):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    items = []
    for it in ds:
        prompt = f"Question: {it['question']}\nAnswer:"
        choices = it["choices"]["text"]
        try:
            label = it["choices"]["label"].index(it["answerKey"])
        except ValueError:
            continue
        items.append((prompt, choices, label))
    import random
    random.Random(seed).shuffle(items)
    return items[:n]


@_register("arc_challenge")
def _arc_challenge(n: int, seed: int):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for it in ds:
        prompt = f"Question: {it['question']}\nAnswer:"
        choices = it["choices"]["text"]
        try:
            label = it["choices"]["label"].index(it["answerKey"])
        except ValueError:
            continue
        items.append((prompt, choices, label))
    import random
    random.Random(seed).shuffle(items)
    return items[:n]


@_register("hellaswag")
def _hellaswag(n: int, seed: int):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    items = []
    for it in ds:
        prompt = it["ctx"]
        choices = it["endings"]
        try:
            label = int(it["label"])
        except (ValueError, TypeError):
            continue
        items.append((prompt, choices, label))
    import random
    random.Random(seed).shuffle(items)
    return items[:n]


@_register("winogrande")
def _winogrande(n: int, seed: int):
    ds = load_dataset("allenai/winogrande", "winogrande_xs",
                       split="validation", trust_remote_code=True)
    items = []
    for it in ds:
        prompt = it["sentence"]
        choices = [it["option1"], it["option2"]]
        label = int(it["answer"]) - 1
        items.append((prompt, choices, label))
    import random
    random.Random(seed).shuffle(items)
    return items[:n]


@_register("openbookqa")
def _openbookqa(n: int, seed: int):
    ds = load_dataset("allenai/openbookqa", "main", split="test")
    items = []
    for it in ds:
        prompt = f"Question: {it['question_stem']}\nAnswer:"
        choices = it["choices"]["text"]
        label = it["choices"]["label"].index(it["answerKey"])
        items.append((prompt, choices, label))
    import random
    random.Random(seed).shuffle(items)
    return items[:n]


@_register("piqa")
def _piqa(n: int, seed: int):
    ds = load_dataset("agicorp/piqa", split="validation")
    items = []
    for it in ds:
        prompt = it["goal"]
        choices = [it["sol1"], it["sol2"]]
        label = int(it["label"])
        items.append((prompt, choices, label))
    import random
    random.Random(seed).shuffle(items)
    return items[:n]


@_register("boolq")
def _boolq(n: int, seed: int):
    ds = load_dataset("aps/super_glue", "boolq", split="validation")
    items = []
    for it in ds:
        prompt = f"{it['passage']}\nQuestion: {it['question']}\nAnswer:"
        choices = ["No", "Yes"]
        label = int(it["label"])
        items.append((prompt, choices, label))
    import random
    random.Random(seed).shuffle(items)
    return items[:n]


@_register("sciq")
def _sciq(n: int, seed: int):
    ds = load_dataset("allenai/sciq", split="test")
    items = []
    for it in ds:
        prompt = f"Question: {it['question']}\nAnswer:"
        choices = [it["correct_answer"], it["distractor1"],
                   it["distractor2"], it["distractor3"]]
        import random
        local_rng = random.Random(0)
        order = list(range(4)); local_rng.shuffle(order)
        choices = [choices[k] for k in order]
        label = order.index(0)
        items.append((prompt, choices, label))
    import random
    random.Random(seed).shuffle(items)
    return items[:n]


def load_mc_task(task: str, n: int = 100, seed: int = 0) -> List[Tuple[str, list, int]]:
    """Load multiple-choice items for one of the 8 evaluation tasks.

    Returns a list of (prompt, [choices...], label_int) triples.
    """
    if task not in _TASK_LOADERS:
        raise ValueError(f"Unknown task '{task}'. "
                          f"Available: {sorted(_TASK_LOADERS)}")
    return _TASK_LOADERS[task](n, seed)


AVAILABLE_TASKS = ("arc_easy", "arc_challenge", "hellaswag", "winogrande",
                   "openbookqa", "piqa", "boolq", "sciq")
