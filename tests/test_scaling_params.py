import torch

from nanochat.gpt import GPT, GPTConfig, has_ve


def _expected_transformer_matrices(n_layer, n_branches, n_embd, n_kv_head):
    block_params_per_branch = 12 * n_embd * n_embd
    branch_split_collect = 2 * n_branches * n_embd * n_embd
    ve_layers = sum(1 for layer_idx in range(n_layer) if has_ve(layer_idx, n_layer))
    ve_gate_params = ve_layers * n_branches * n_kv_head * 32
    return n_layer * n_branches * block_params_per_branch + branch_split_collect + ve_gate_params


def test_transformer_matrix_param_count_for_1x10_and_2x6_configs():
    c = 768
    n_head = 6
    n_kv_head = 6
    configs = [(1, 10), (2, 6)]

    for n_layer, n_branches in configs:
        with torch.device("meta"):
            model = GPT(GPTConfig(n_layer=n_layer, n_embd=c, n_head=n_head, n_kv_head=n_kv_head, n_branches=n_branches))

        counts = model.num_scaling_params()
        expected = _expected_transformer_matrices(n_layer, n_branches, c, n_kv_head)
        assert counts["transformer_matrices"] == expected
