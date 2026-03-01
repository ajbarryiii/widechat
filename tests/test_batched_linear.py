import torch

from nanochat.gpt import BatchedLinear


def test_batched_linear_matches_per_branch_linear():
    torch.manual_seed(0)
    n, t, r, i, o = 2, 3, 4, 5, 6
    x = torch.randn(n, t, r, i, dtype=torch.float32)

    layer = BatchedLinear(n_branches=r, in_features=i, out_features=o)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(r, o, i, dtype=torch.float32))

    per_branch = [torch.nn.Linear(i, o, bias=False) for _ in range(r)]
    with torch.no_grad():
        for branch_idx, linear in enumerate(per_branch):
            linear.weight.copy_(layer.weight[branch_idx])

    y_batched = layer(x)
    y_reference = torch.stack(
        [linear(x[:, :, branch_idx, :]) for branch_idx, linear in enumerate(per_branch)],
        dim=2,
    )

    assert y_batched.shape == (n, t, r, o)
    torch.testing.assert_close(y_batched, y_reference, atol=1e-6, rtol=1e-6)


def test_batched_linear_backward_gradients_are_finite_and_shaped():
    torch.manual_seed(0)
    n, t, r, i, o = 2, 4, 3, 7, 5
    x = torch.randn(n, t, r, i, dtype=torch.float32, requires_grad=True)
    layer = BatchedLinear(n_branches=r, in_features=i, out_features=o)
    torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)

    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert y.shape == (n, t, r, o)
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert x.grad.shape == x.shape
    assert layer.weight.grad.shape == layer.weight.shape
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()
