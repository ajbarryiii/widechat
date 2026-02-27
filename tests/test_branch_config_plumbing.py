import torch

from nanochat.engine import KVCache
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


def test_gpt_multi_branch_forward_and_backward_smoke():
    config = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        n_branches=2,
    )

    model = GPT(config)
    model.init_weights()

    idx = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
    targets = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
    loss = model(idx, targets)
    assert torch.isfinite(loss)

    loss.backward()
    assert model.branch_proj is not None
    assert model.branch_proj["linear_in"].weight.grad is not None
    assert model.branch_proj["linear_out"].weight.grad is not None


def test_n_branches_1_uses_baseline_block_path_and_kv_cache():
    config = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        n_branches=1,
    )

    model = GPT(config)
    model.init_weights()

    assert model.branch_proj is None
    assert model.uses_parallel_branches is False

    idx = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
    kv_cache = KVCache(
        batch_size=2,
        num_heads=config.n_kv_head,
        seq_len=config.sequence_len,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
        device=idx.device,
        dtype=torch.float32,
    )
    logits = model(idx, kv_cache=kv_cache)

    assert logits.shape == (2, 8, config.vocab_size)
    assert kv_cache.get_pos() == 8


def test_n_branches_gt_1_supports_kv_cache_path():
    config = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        n_branches=2,
    )

    model = GPT(config)
    model.init_weights()

    batch_size = 2
    prefill_len = 6
    kv_cache = KVCache(
        batch_size=batch_size * config.n_branches,
        num_heads=config.n_kv_head,
        seq_len=config.sequence_len,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    idx = torch.randint(0, config.vocab_size, (batch_size, prefill_len), dtype=torch.long)
    logits = model(idx, kv_cache=kv_cache)
    assert logits.shape == (batch_size, prefill_len, config.vocab_size)
    assert kv_cache.get_pos() == prefill_len

    next_idx = torch.randint(0, config.vocab_size, (batch_size, 1), dtype=torch.long)
    next_logits = model(next_idx, kv_cache=kv_cache)
    assert next_logits.shape == (batch_size, 1, config.vocab_size)
    assert kv_cache.get_pos() == prefill_len + 1
