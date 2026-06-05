#!/usr/bin/env bash
# quickstart.sh — minimal end-to-end MoL run on Llama-2-7B at L_eff = 24.
#
# Runs Stage A (router training, ~1000 steps) + Stage B (LoRA + KD, ~1000 steps)
# and evaluates on WikiText-2 PPL plus the 8-task accuracy suite. Wall time is
# roughly 20 minutes on a 48 GB GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m scripts.train \
    --model NousResearch/Llama-2-7b-hf \
    --variant soft \
    --c-star 0.75 \
    --steps 2000 \
    --lora-r 8 \
    --kd-weight 1.0 \
    --seed 0
