import torch

from nanochat.gpt import Block, GPTConfig, ParallelBlock


def _make_cos_sin(seq_len, head_dim, device, dtype):
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000 ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype=dtype)[None, :, None, :]
    sin = freqs.sin().to(dtype=dtype)[None, :, None, :]
    return cos, sin


def test_parallel_block_matches_per_branch_reference():
    torch.manual_seed(0)
    n, t, r, c, n_head = 2, 5, 3, 16, 4
    config = GPTConfig(n_layer=2, n_embd=c, n_head=n_head, n_kv_head=n_head, n_branches=r)
    x = torch.randn(n, t, r, c, dtype=torch.float32)
    cos_sin = _make_cos_sin(t, c // n_head, x.device, x.dtype)
    window_size = (t, 0)

    parallel_block = ParallelBlock(config, layer_idx=0)
    with torch.no_grad():
        for param in parallel_block.parameters():
            torch.nn.init.normal_(param, mean=0.0, std=0.1)

    y_parallel = parallel_block(x, ve=None, cos_sin=cos_sin, window_size=window_size, kv_cache=None)

    refs = [Block(config, layer_idx=0) for _ in range(r)]
    with torch.no_grad():
        for branch_idx, ref in enumerate(refs):
            ref.attn.c_q.weight.copy_(parallel_block.attn.c_q.weight[branch_idx])
            ref.attn.c_k.weight.copy_(parallel_block.attn.c_k.weight[branch_idx])
            ref.attn.c_v.weight.copy_(parallel_block.attn.c_v.weight[branch_idx])
            ref.attn.c_proj.weight.copy_(parallel_block.attn.c_proj.weight[branch_idx])
            if ref.attn.ve_gate is not None:
                assert parallel_block.attn.ve_gate is not None
                ref.attn.ve_gate.weight.copy_(parallel_block.attn.ve_gate[branch_idx])
            ref.mlp.c_fc.weight.copy_(parallel_block.mlp.c_fc.weight[branch_idx])
            ref.mlp.c_proj.weight.copy_(parallel_block.mlp.c_proj.weight[branch_idx])

    y_ref = torch.stack(
        [ref(x[:, :, branch_idx, :], ve=None, cos_sin=cos_sin, window_size=window_size, kv_cache=None) for branch_idx, ref in enumerate(refs)],
        dim=2,
    )

    assert y_parallel.shape == (n, t, r, c)
    torch.testing.assert_close(y_parallel, y_ref, atol=1e-6, rtol=1e-6)


def test_n_branches_1_block_parity_forward_and_backward():
    torch.manual_seed(0)
    n, t, c, n_head = 2, 5, 16, 4
    config = GPTConfig(n_layer=2, n_embd=c, n_head=n_head, n_kv_head=n_head, n_branches=1)
    cos_sin = _make_cos_sin(t, c // n_head, device="cpu", dtype=torch.float32)
    window_size = (t, 0)

    block = Block(config, layer_idx=0)
    parallel_block = ParallelBlock(config, layer_idx=0)

    with torch.no_grad():
        for param in block.parameters():
            torch.nn.init.normal_(param, mean=0.0, std=0.1)
        parallel_block.attn.c_q.weight[0].copy_(block.attn.c_q.weight)
        parallel_block.attn.c_k.weight[0].copy_(block.attn.c_k.weight)
        parallel_block.attn.c_v.weight[0].copy_(block.attn.c_v.weight)
        parallel_block.attn.c_proj.weight[0].copy_(block.attn.c_proj.weight)
        parallel_block.mlp.c_fc.weight[0].copy_(block.mlp.c_fc.weight)
        parallel_block.mlp.c_proj.weight[0].copy_(block.mlp.c_proj.weight)

    x = torch.randn(n, t, c, dtype=torch.float32, requires_grad=True)
    x_parallel = x.detach().clone().unsqueeze(2).requires_grad_(True)

    y = block(x, ve=None, cos_sin=cos_sin, window_size=window_size, kv_cache=None)
    y_parallel = parallel_block(x_parallel, ve=None, cos_sin=cos_sin, window_size=window_size, kv_cache=None).squeeze(2)

    torch.testing.assert_close(y_parallel, y, atol=1e-6, rtol=1e-6)

    loss = y.square().mean()
    loss_parallel = y_parallel.square().mean()
    loss.backward()
    loss_parallel.backward()

    assert x.grad is not None
    assert x_parallel.grad is not None
    assert parallel_block.attn.c_q.weight.grad is not None
    assert parallel_block.attn.c_k.weight.grad is not None
    assert parallel_block.attn.c_v.weight.grad is not None
    assert parallel_block.attn.c_proj.weight.grad is not None
    assert parallel_block.mlp.c_fc.weight.grad is not None
    assert parallel_block.mlp.c_proj.weight.grad is not None
    assert block.attn.c_q.weight.grad is not None
    assert block.attn.c_k.weight.grad is not None
    assert block.attn.c_v.weight.grad is not None
    assert block.attn.c_proj.weight.grad is not None
    assert block.mlp.c_fc.weight.grad is not None
    assert block.mlp.c_proj.weight.grad is not None
    torch.testing.assert_close(x_parallel.grad.squeeze(2), x.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parallel_block.attn.c_q.weight.grad[0], block.attn.c_q.weight.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parallel_block.attn.c_k.weight.grad[0], block.attn.c_k.weight.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parallel_block.attn.c_v.weight.grad[0], block.attn.c_v.weight.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parallel_block.attn.c_proj.weight.grad[0], block.attn.c_proj.weight.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parallel_block.mlp.c_fc.weight.grad[0], block.mlp.c_fc.weight.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parallel_block.mlp.c_proj.weight.grad[0], block.mlp.c_proj.weight.grad, atol=1e-6, rtol=1e-6)


def test_parallel_block_with_value_embeddings_matches_per_branch_reference():
    torch.manual_seed(0)
    n, t, r, c, n_head = 2, 5, 3, 64, 4
    config = GPTConfig(n_layer=2, n_embd=c, n_head=n_head, n_kv_head=n_head, n_branches=r)
    x = torch.randn(n, t, r, c, dtype=torch.float32)
    head_dim = c // n_head
    ve = torch.randn(n, t, r, n_head, head_dim, dtype=torch.float32)
    cos_sin = _make_cos_sin(t, head_dim, x.device, x.dtype)
    window_size = (t, 0)

    layer_idx = 1
    parallel_block = ParallelBlock(config, layer_idx=layer_idx)
    with torch.no_grad():
        for param in parallel_block.parameters():
            torch.nn.init.normal_(param, mean=0.0, std=0.1)

    y_parallel = parallel_block(x, ve=ve, cos_sin=cos_sin, window_size=window_size, kv_cache=None)

    refs = [Block(config, layer_idx=layer_idx) for _ in range(r)]
    with torch.no_grad():
        for branch_idx, ref in enumerate(refs):
            ref.attn.c_q.weight.copy_(parallel_block.attn.c_q.weight[branch_idx])
            ref.attn.c_k.weight.copy_(parallel_block.attn.c_k.weight[branch_idx])
            ref.attn.c_v.weight.copy_(parallel_block.attn.c_v.weight[branch_idx])
            ref.attn.c_proj.weight.copy_(parallel_block.attn.c_proj.weight[branch_idx])
            assert ref.attn.ve_gate is not None
            assert parallel_block.attn.ve_gate is not None
            ref.attn.ve_gate.weight.copy_(parallel_block.attn.ve_gate[branch_idx])
            ref.mlp.c_fc.weight.copy_(parallel_block.mlp.c_fc.weight[branch_idx])
            ref.mlp.c_proj.weight.copy_(parallel_block.mlp.c_proj.weight[branch_idx])

    y_ref = torch.stack(
        [
            ref(
                x[:, :, branch_idx, :],
                ve=ve[:, :, branch_idx, :, :],
                cos_sin=cos_sin,
                window_size=window_size,
                kv_cache=None,
            )
            for branch_idx, ref in enumerate(refs)
        ],
        dim=2,
    )

    assert y_parallel.shape == (n, t, r, c)
    torch.testing.assert_close(y_parallel, y_ref, atol=2e-6, rtol=2e-6)
