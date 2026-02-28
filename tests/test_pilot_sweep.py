import pytest
from types import SimpleNamespace

from nanochat import pilot_sweep
from nanochat.pilot_sweep import (
    MAX_RECOMMENDED_EVAL_EVERY,
    MIN_RECOMMENDED_EVAL_EVERY,
    PilotTarget,
    apply_ranking_rule,
    build_pilot_command,
    extract_val_bpb_trace,
    format_finalists_summary,
    format_ranking_table,
    select_finalists,
    summarize_pilot_output,
)


def test_build_pilot_command_sets_expected_flags_and_iterations():
    target = PilotTarget(label="6x2", depth=6, n_branches=2, aspect_ratio=128)
    command, num_iterations = build_pilot_command(
        target=target,
        python_exe="python",
        max_seq_len=2048,
        total_batch_size=524288,
        device_batch_size=16,
        pilot_tokens=250_000_000,
        eval_every=75,
        eval_tokens=1_048_576,
        device_type="cuda",
        extra_args=["--head-dim", "128"],
    )

    assert num_iterations == 476
    assert command[command.index("--depth") + 1] == "6"
    assert command[command.index("--n-branches") + 1] == "2"
    assert command[command.index("--aspect-ratio") + 1] == "128"
    assert command[command.index("--num-iterations") + 1] == "476"
    assert command[command.index("--eval-every") + 1] == "75"
    assert command[command.index("--eval-tokens") + 1] == "1048576"
    assert command[-2:] == ["--head-dim", "128"]


def test_build_pilot_command_rejects_eval_cadence_outside_recommended_range():
    target = PilotTarget(label="12x1", depth=12, n_branches=1, aspect_ratio=64)

    with pytest.raises(ValueError, match="eval_every must be between"):
        build_pilot_command(
            target=target,
            python_exe="python",
            max_seq_len=2048,
            total_batch_size=524288,
            device_batch_size=16,
            pilot_tokens=250_000_000,
            eval_every=MIN_RECOMMENDED_EVAL_EVERY - 1,
            eval_tokens=1_048_576,
            device_type="cuda",
            extra_args=[],
        )

    with pytest.raises(ValueError, match="eval_every must be between"):
        build_pilot_command(
            target=target,
            python_exe="python",
            max_seq_len=2048,
            total_batch_size=524288,
            device_batch_size=16,
            pilot_tokens=250_000_000,
            eval_every=MAX_RECOMMENDED_EVAL_EVERY + 1,
            eval_tokens=1_048_576,
            device_type="cuda",
            extra_args=[],
        )


def test_build_pilot_command_requires_budget_for_at_least_one_eval_point():
    target = PilotTarget(label="12x1", depth=12, n_branches=1, aspect_ratio=64)

    with pytest.raises(ValueError, match="pilot_tokens budget is too small"):
        build_pilot_command(
            target=target,
            python_exe="python",
            max_seq_len=2048,
            total_batch_size=1_000,
            device_batch_size=16,
            pilot_tokens=90_000,
            eval_every=100,
            eval_tokens=1_048_576,
            device_type="cuda",
            extra_args=[],
        )


def test_extract_val_bpb_trace_parses_all_evals():
    output = """
Step 00000 | Validation bpb: 4.123456
Step 00075 | Validation bpb: 4.001234
Step 00150 | Validation bpb: 3.987654
"""
    assert extract_val_bpb_trace(output) == [4.123456, 4.001234, 3.987654]


def test_summarize_pilot_output_uses_throughput_and_val_metrics():
    output = """
step 00039/00040 (97.50%) | loss: 4.0 | lrm: 0.5 | dt: 100.00ms | tok/sec: 327,680 | bf16_mfu: 12.50 | peak_mem: 1234.56MiB | epoch: 0 | total time: 0.50m
Peak memory usage: 1234.56MiB
Average tok/sec (post-warmup): 300,000
Step 00000 | Validation bpb: 4.123456
Step 00075 | Validation bpb: 4.001234
"""
    summary = summarize_pilot_output(output)
    assert summary["selected_tok_per_sec"] == 300000
    assert summary["min_val_bpb"] == 4.001234
    assert summary["final_val_bpb"] == 4.001234
    assert summary["unstable"] is False


def test_summarize_pilot_output_marks_parse_failures_unstable():
    output = "Step 00075 | Validation bpb: 4.001234\n"
    summary = summarize_pilot_output(output)
    assert summary["selected_tok_per_sec"] == 0
    assert summary["min_val_bpb"] == 4.001234
    assert summary["unstable"] is True


def test_run_single_pilot_marks_nonzero_returncode_unstable(monkeypatch):
    def fake_run(_command, check, capture_output, text):
        assert check is False
        assert capture_output is True
        assert text is True
        return SimpleNamespace(returncode=1, stdout="", stderr="fatal: oops")

    monkeypatch.setattr(pilot_sweep.subprocess, "run", fake_run)
    _output, metrics = pilot_sweep.run_single_pilot(["python", "-m", "scripts.base_train"])
    assert metrics["command_failed"] is True
    assert metrics["failure_returncode"] == 1
    assert metrics["unstable"] is True


def test_apply_ranking_rule_disqualifies_slow_without_clear_gain():
    ranked = apply_ranking_rule(
        [
            {
                "config": "12x1",
                "selected_tok_per_sec": 1000,
                "min_val_bpb": 4.10,
                "token_budget": 250_000_000,
                "unstable": False,
            },
            {
                "config": "2x5",
                "selected_tok_per_sec": 930,
                "min_val_bpb": 4.09,
                "token_budget": 250_000_000,
                "unstable": False,
            },
            {
                "config": "1x10",
                "selected_tok_per_sec": 980,
                "min_val_bpb": 4.20,
                "token_budget": 250_000_000,
                "unstable": True,
            },
        ],
        slowdown_threshold_pct=5.0,
        clear_bpb_gain=0.02,
    )

    slow_row = next(row for row in ranked if row["config"] == "2x5")
    unstable_row = next(row for row in ranked if row["config"] == "1x10")
    assert slow_row["qualified"] is False
    assert slow_row["disqualify_reason"] == "slow>5.0%"
    assert unstable_row["qualified"] is False
    assert unstable_row["disqualify_reason"] == "unstable"


def test_apply_ranking_rule_keeps_slow_model_if_clearly_better():
    ranked = apply_ranking_rule(
        [
            {
                "config": "12x1",
                "selected_tok_per_sec": 1000,
                "min_val_bpb": 4.10,
                "token_budget": 250_000_000,
                "unstable": False,
            },
            {
                "config": "6x2",
                "selected_tok_per_sec": 940,
                "min_val_bpb": 4.05,
                "token_budget": 250_000_000,
                "unstable": False,
            },
        ],
        slowdown_threshold_pct=5.0,
        clear_bpb_gain=0.02,
    )

    row = next(row for row in ranked if row["config"] == "6x2")
    assert row["qualified"] is True
    assert row["rank"] == 1


def test_apply_ranking_rule_disqualifies_mismatched_token_budget():
    ranked = apply_ranking_rule(
        [
            {
                "config": "12x1",
                "selected_tok_per_sec": 1000,
                "min_val_bpb": 4.10,
                "token_budget": 250_000_000,
                "unstable": False,
            },
            {
                "config": "4x3",
                "selected_tok_per_sec": 1200,
                "min_val_bpb": 4.00,
                "token_budget": 200_000_000,
                "unstable": False,
            },
        ],
        slowdown_threshold_pct=5.0,
        clear_bpb_gain=0.02,
    )

    row = next(item for item in ranked if item["config"] == "4x3")
    assert row["qualified"] is False
    assert row["disqualify_reason"] == "token-budget-mismatch"


def test_apply_ranking_rule_requires_positive_baseline_throughput():
    with pytest.raises(ValueError, match="baseline selected_tok_per_sec must be > 0"):
        apply_ranking_rule(
            [
                {
                    "config": "12x1",
                    "selected_tok_per_sec": 0,
                    "min_val_bpb": 4.10,
                    "token_budget": 250_000_000,
                    "unstable": True,
                },
                {
                    "config": "6x2",
                    "selected_tok_per_sec": 900,
                    "min_val_bpb": 4.00,
                    "token_budget": 250_000_000,
                    "unstable": False,
                },
            ]
        )


def test_format_ranking_table_emits_markdown_rows():
    rows = [
        {
            "rank": 1,
            "config": "12x1",
            "selected_tok_per_sec": 1000,
            "min_val_bpb": 4.1,
            "token_budget": 250000000,
            "qualified": True,
            "disqualify_reason": None,
        },
        {
            "rank": None,
            "config": "2x5",
            "selected_tok_per_sec": 900,
            "min_val_bpb": 4.2,
            "token_budget": 250000000,
            "qualified": False,
            "disqualify_reason": "slow>5.0%",
        },
    ]
    table = format_ranking_table(rows)
    assert "| Rank | Config | tok/sec | vs 12x1 | min val bpb | token budget | Status |" in table
    assert "| 1 | 12x1 | 1,000 | +0.0% | 4.1000 | 250,000,000 | qualified |" in table
    assert "| - | 2x5 | 900 | -10.0% | 4.2000 | 250,000,000 | disqualified (slow>5.0%) |" in table


def test_select_finalists_returns_only_qualified_rows_up_to_limit():
    rows = [
        {
            "rank": 1,
            "config": "12x1",
            "selected_tok_per_sec": 1000,
            "min_val_bpb": 4.1,
            "token_budget": 250000000,
            "qualified": True,
            "disqualify_reason": None,
        },
        {
            "rank": None,
            "config": "2x5",
            "selected_tok_per_sec": 900,
            "min_val_bpb": 4.2,
            "token_budget": 250000000,
            "qualified": False,
            "disqualify_reason": "slow>5.0%",
        },
        {
            "rank": 2,
            "config": "6x2",
            "selected_tok_per_sec": 980,
            "min_val_bpb": 4.0,
            "token_budget": 250000000,
            "qualified": True,
            "disqualify_reason": None,
        },
    ]
    finalists = select_finalists(rows, max_finalists=1)
    assert [row["config"] for row in finalists] == ["12x1"]


def test_format_finalists_summary_reports_selected_configs():
    rows = [
        {
            "rank": 1,
            "config": "12x1",
            "selected_tok_per_sec": 1000,
            "min_val_bpb": 4.1,
            "token_budget": 250000000,
            "qualified": True,
            "disqualify_reason": None,
        },
        {
            "rank": 2,
            "config": "6x2",
            "selected_tok_per_sec": 980,
            "min_val_bpb": 4.0,
            "token_budget": 250000000,
            "qualified": True,
            "disqualify_reason": None,
        },
    ]
    summary = format_finalists_summary(rows)
    assert "Selected finalists:" in summary
    assert "- 12x1: rank=1, tok/sec=1,000 (+0.0% vs 12x1), min_val_bpb=4.1000" in summary
    assert "- 6x2: rank=2, tok/sec=980 (-2.0% vs 12x1), min_val_bpb=4.0000" in summary
