import torch
import pytest

from nanochat.gpt import GPT, GPTConfig


def test_cpu_short_train_smoke_multi_branch_no_crash_and_finite_loss():
    config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        n_branches=2,
        window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()
    optimizer = model.setup_optimizer(
        matrix_lr=1e-3,
        embedding_lr=1e-3,
        unembedding_lr=1e-3,
        scalar_lr=1e-3,
        weight_decay=0.0,
    )

    batch_size, seq_len = 2, 8
    torch.manual_seed(0)
    losses = []

    for _ in range(3):
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        loss = model(idx, targets)
        assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses))), "All training-step losses must be finite"


@pytest.mark.parametrize("n_layer,n_branches", [(12, 1), (6, 2)])
def test_cpu_stage0_short_checks_for_depth_branch_candidates(n_layer, n_branches):
    config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=n_layer,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        n_branches=n_branches,
        window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()

    batch_size, seq_len = 2, 8
    torch.manual_seed(0)
    losses = []

    for _ in range(2):
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
        model.zero_grad(set_to_none=True)
        loss = model(idx, targets)
        assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "Model gradients must be finite"
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses))), "All training-step losses must be finite"
