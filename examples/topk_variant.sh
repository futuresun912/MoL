#!/usr/bin/env bash
# topk_variant.sh — TopK-MoL at c⋆ = 0.875 (an under-budget regime where TopK
# is competitive with the soft variant) on Llama-2-7B.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m scripts.train \
    --model NousResearch/Llama-2-7b-hf \
    --variant topk \
    --c-star 0.875 \
    --steps 2000 \
    --lora-r 8 \
    --kd-weight 1.0 \
    --seed 0
