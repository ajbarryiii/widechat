import pytest
import torch

from nanochat.checkpoint_manager import _patch_missing_config_keys
from nanochat.gpt import GPT, GPTConfig


def test_patch_missing_config_keys_adds_n_branches_and_window_pattern():
    config = {
        "sequence_len": 16,
        "vocab_size": 32,
        "n_layer": 2,
        "n_head": 2,
        "n_kv_head": 2,
        "n_embd": 16,
    }

    _patch_missing_config_keys(config)

    assert config["window_pattern"] == "L"
    assert config["n_branches"] == 1


def test_gpt_config_exposes_n_branches():
    assert GPTConfig().n_branches == 1
    assert GPTConfig(n_branches=3).n_branches == 3


def test_gpt_rejects_unimplemented_multi_branch_model():
    config = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        n_branches=2,
    )

    with pytest.raises(NotImplementedError, match="n_branches>1"):
        with torch.device("meta"):
            GPT(config)
