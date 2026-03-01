import torch

from nanochat.gpt import GPTConfig, ParallelMLP


def test_parallel_mlp_matches_per_branch_reference():
    torch.manual_seed(0)
    n, t, r, c = 2, 3, 4, 8
    x = torch.randn(n, t, r, c, dtype=torch.float32)

    config = GPTConfig(n_embd=c, n_branches=r)
    parallel_mlp = ParallelMLP(config)
    with torch.no_grad():
        torch.nn.init.normal_(parallel_mlp.c_fc.weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(parallel_mlp.c_proj.weight, mean=0.0, std=0.1)

    y_parallel = parallel_mlp(x)

    y_reference = []
    for branch_idx in range(r):
        fc = torch.nn.Linear(c, 4 * c, bias=False)
        proj = torch.nn.Linear(4 * c, c, bias=False)
        with torch.no_grad():
            fc.weight.copy_(parallel_mlp.c_fc.weight[branch_idx])
            proj.weight.copy_(parallel_mlp.c_proj.weight[branch_idx])
        y_branch = proj(torch.relu(fc(x[:, :, branch_idx, :])).square())
        y_reference.append(y_branch)
    y_reference = torch.stack(y_reference, dim=2)

    assert y_parallel.shape == (n, t, r, c)
    torch.testing.assert_close(y_parallel, y_reference, atol=1e-6, rtol=1e-6)
