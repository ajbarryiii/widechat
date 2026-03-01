import torch

from nanochat.gpt import CausalSelfAttention, GPTConfig, ParallelCausalSelfAttention


def _make_cos_sin(seq_len, head_dim, device, dtype):
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000 ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype=dtype)[None, :, None, :]
    sin = freqs.sin().to(dtype=dtype)[None, :, None, :]
    return cos, sin


def test_parallel_attention_flatten_unflatten_roundtrip():
    torch.manual_seed(0)
    n, t, r, h, d = 2, 5, 3, 4, 8
    x = torch.randn(n, t, r, h, d, dtype=torch.float32)
    config = GPTConfig(n_embd=h * d, n_head=h, n_kv_head=h, n_branches=r)
    attn = ParallelCausalSelfAttention(config, layer_idx=0)

    flat = attn._flatten_attention_tensor(x)
    unflat = attn._unflatten_attention_tensor(flat, n, t)

    assert flat.shape == (n * r, t, h, d)
    assert unflat.shape == (n, t, r, h, d)
    torch.testing.assert_close(unflat, x, atol=0.0, rtol=0.0)


def test_parallel_attention_matches_per_branch_reference():
    torch.manual_seed(0)
    n, t, r, c, n_head = 2, 6, 3, 16, 4
    config = GPTConfig(n_layer=2, n_embd=c, n_head=n_head, n_kv_head=n_head, n_branches=r)
    x = torch.randn(n, t, r, c, dtype=torch.float32)
    cos_sin = _make_cos_sin(t, c // n_head, x.device, x.dtype)
    window_size = (t, 0)

    parallel_attn = ParallelCausalSelfAttention(config, layer_idx=0)
    with torch.no_grad():
        torch.nn.init.normal_(parallel_attn.c_q.weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(parallel_attn.c_k.weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(parallel_attn.c_v.weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(parallel_attn.c_proj.weight, mean=0.0, std=0.1)

    y_parallel = parallel_attn(x, ve=None, cos_sin=cos_sin, window_size=window_size, kv_cache=None)

    refs = [CausalSelfAttention(config, layer_idx=0) for _ in range(r)]
    with torch.no_grad():
        for branch_idx, ref in enumerate(refs):
            ref.c_q.weight.copy_(parallel_attn.c_q.weight[branch_idx])
            ref.c_k.weight.copy_(parallel_attn.c_k.weight[branch_idx])
            ref.c_v.weight.copy_(parallel_attn.c_v.weight[branch_idx])
            ref.c_proj.weight.copy_(parallel_attn.c_proj.weight[branch_idx])

    y_ref = torch.stack(
        [ref(x[:, :, branch_idx, :], ve=None, cos_sin=cos_sin, window_size=window_size, kv_cache=None) for branch_idx, ref in enumerate(refs)],
        dim=2,
    )

    assert y_parallel.shape == (n, t, r, c)
    torch.testing.assert_close(y_parallel, y_ref, atol=1e-6, rtol=1e-6)
