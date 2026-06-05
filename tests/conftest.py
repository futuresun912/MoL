"""
Pytest fixtures — provides a tiny synthetic Llama-arch model for fast tests.

We deliberately avoid downloading multi-GB checkpoints in unit tests. A
TinyLlama (hidden=64, L=4, vocab=200) is enough to exercise every code path
in MoL while keeping each test under one second.
"""
import torch
import pytest

from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def dtype():
    return torch.float32


@pytest.fixture
def tiny_llama(device, dtype):
    """A 4-layer Llama-arch model with hidden=64, vocab=200. ~50 K params."""
    cfg = LlamaConfig(
        vocab_size=200,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(device=device, dtype=dtype)
    model.eval()
    return model


@pytest.fixture
def calib_seqs(device):
    """A short calibration list — 4 random sequences of 32 tokens (vocab=200)."""
    torch.manual_seed(0)
    return [torch.randint(0, 200, (32,)).to(device) for _ in range(4)]


@pytest.fixture
def input_ids(device):
    torch.manual_seed(0)
    return torch.randint(0, 200, (1, 32)).to(device)
