import math

import pytest

from nanochat.common import compute_post_warmup_tok_per_sec, compute_training_perf_metrics


def test_compute_training_perf_metrics_returns_expected_values():
    metrics = compute_training_perf_metrics(
        total_batch_size=2048,
        dt=0.5,
        num_flops_per_token=1000,
        gpu_peak_flops=1e12,
        ddp_world_size=2,
        peak_memory_bytes=314572800,  # 300 MiB
    )

    assert metrics["train/tok_per_sec"] == 4096
    assert math.isclose(metrics["train/mfu"], 0.0002048)
    assert math.isclose(metrics["train/peak_memory_mib"], 300.0)


@pytest.mark.parametrize("dt", [0.0, -1.0])
def test_compute_training_perf_metrics_rejects_non_positive_dt(dt):
    with pytest.raises(ValueError, match="dt must be > 0"):
        compute_training_perf_metrics(
            total_batch_size=256,
            dt=dt,
            num_flops_per_token=100,
            gpu_peak_flops=1e12,
            ddp_world_size=1,
            peak_memory_bytes=0,
        )


def test_compute_training_perf_metrics_rejects_non_positive_world_size():
    with pytest.raises(ValueError, match="ddp_world_size must be > 0"):
        compute_training_perf_metrics(
            total_batch_size=256,
            dt=1.0,
            num_flops_per_token=100,
            gpu_peak_flops=1e12,
            ddp_world_size=0,
            peak_memory_bytes=0,
        )


def test_compute_post_warmup_tok_per_sec_returns_expected_value():
    avg_tok_per_sec = compute_post_warmup_tok_per_sec(
        total_batch_size=2048,
        post_warmup_steps=50,
        total_training_time=10.0,
    )
    assert avg_tok_per_sec == 10240


@pytest.mark.parametrize(
    "post_warmup_steps,total_training_time",
    [
        (0, 10.0),
        (-1, 10.0),
        (10, 0.0),
        (10, -1.0),
    ],
)
def test_compute_post_warmup_tok_per_sec_returns_none_without_valid_window(post_warmup_steps, total_training_time):
    assert compute_post_warmup_tok_per_sec(
        total_batch_size=2048,
        post_warmup_steps=post_warmup_steps,
        total_training_time=total_training_time,
    ) is None
