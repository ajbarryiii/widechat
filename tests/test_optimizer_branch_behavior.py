import os
import tempfile

import pytest
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import DistMuonAdamW, MuonAdamW


def _tiny_branched_model():
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        n_branches=2,
        window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()
    return model


def _one_train_step(model, optimizer):
    idx = torch.randint(0, model.config.vocab_size, (2, 8), dtype=torch.long)
    targets = torch.randint(0, model.config.vocab_size, (2, 8), dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    loss = model(idx, targets)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()


def test_muonadamw_handles_branched_parameter_groups():
    model = _tiny_branched_model()
    optimizer = model.setup_optimizer(matrix_lr=1e-3, embedding_lr=1e-3, unembedding_lr=1e-3, scalar_lr=1e-3)
    assert isinstance(optimizer, MuonAdamW)
    _one_train_step(model, optimizer)


def test_distmuonadamw_handles_branched_parameter_groups_world_size1():
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    if dist.is_initialized():
        pytest.skip("process group already initialized")

    model = _tiny_branched_model()
    old_env = {k: os.environ.get(k) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}

    with tempfile.NamedTemporaryFile() as init_file:
        try:
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            dist.init_process_group(backend="gloo", init_method=f"file://{init_file.name}", rank=0, world_size=1)
            optimizer = model.setup_optimizer(matrix_lr=1e-3, embedding_lr=1e-3, unembedding_lr=1e-3, scalar_lr=1e-3)
            assert isinstance(optimizer, DistMuonAdamW)
            _one_train_step(model, optimizer)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
