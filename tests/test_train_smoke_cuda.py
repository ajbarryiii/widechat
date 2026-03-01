import pytest
import torch

from nanochat.gpt import GPT, GPTConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this smoke test")
def test_cuda_short_train_smoke_multi_branch_compile_path_finite_loss():
    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

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

    model = GPT(config).to(device)
    model.init_weights()
    model = torch.compile(model, fullgraph=True)
    optimizer = model.setup_optimizer(
        matrix_lr=1e-3,
        embedding_lr=1e-3,
        unembedding_lr=1e-3,
        scalar_lr=1e-3,
        weight_decay=0.0,
    )

    losses = []
    for _ in range(2):
        idx = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long, device=device)
        targets = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(idx, targets)
        assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().float().item())

    assert all(torch.isfinite(torch.tensor(losses))), "All CUDA training-step losses must be finite"
