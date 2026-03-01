import json

import torch

import nanochat.checkpoint_manager as checkpoint_manager
from nanochat.gpt import GPT, GPTConfig


class _FakeTokenizer:
    def __init__(self, vocab_size):
        self._vocab_size = vocab_size

    def get_vocab_size(self):
        return self._vocab_size


def test_build_model_loads_old_checkpoint_without_n_branches(monkeypatch, tmp_path):
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

    step = 1
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    torch.save(model.state_dict(), checkpoint_dir / f"model_{step:06d}.pt")
    old_style_model_config = {
        "sequence_len": config.sequence_len,
        "vocab_size": config.vocab_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_kv_head": config.n_kv_head,
        "n_embd": config.n_embd,
    }
    with open(checkpoint_dir / f"meta_{step:06d}.json", "w", encoding="utf-8") as f:
        json.dump({"model_config": old_style_model_config}, f)

    monkeypatch.setattr(
        checkpoint_manager,
        "get_tokenizer",
        lambda: _FakeTokenizer(config.vocab_size),
    )

    loaded_model, _, loaded_meta = checkpoint_manager.build_model(
        str(checkpoint_dir),
        step,
        torch.device("cpu"),
        phase="eval",
    )

    assert loaded_model.config.n_branches == 1
    assert loaded_meta["model_config"]["n_branches"] == 1

    idx = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)
    logits = loaded_model(idx)
    assert logits.shape == (1, 8, config.vocab_size)
