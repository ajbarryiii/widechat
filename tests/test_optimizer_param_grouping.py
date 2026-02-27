from nanochat.gpt import GPT, GPTConfig


def _param_ids(groups, kind):
    return {id(p) for group in groups if group["kind"] == kind for p in group["params"]}


def test_batched_parallel_weights_use_adamw_not_muon():
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        n_branches=2,
    )
    model = GPT(config)
    model.init_weights()
    optimizer = model.setup_optimizer()

    batched_param_ids = {id(p) for p in model.transformer.h.parameters() if p.ndim != 2}
    assert batched_param_ids

    adamw_param_ids = _param_ids(optimizer.param_groups, "adamw")
    muon_param_ids = _param_ids(optimizer.param_groups, "muon")

    assert batched_param_ids.issubset(adamw_param_ids)
    assert batched_param_ids.isdisjoint(muon_param_ids)


def test_baseline_matrix_weights_still_use_muon():
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        n_branches=1,
    )
    model = GPT(config)
    model.init_weights()
    optimizer = model.setup_optimizer()

    matrix_param_ids = {id(p) for p in model.transformer.h.parameters() if p.ndim == 2}
    muon_param_ids = _param_ids(optimizer.param_groups, "muon")

    assert matrix_param_ids
    assert matrix_param_ids.issubset(muon_param_ids)
