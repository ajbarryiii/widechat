from nanochat.pilot_sweep import (
    PilotTarget,
    apply_ranking_rule,
    build_pilot_command,
    extract_val_bpb_trace,
    format_ranking_table,
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


def test_apply_ranking_rule_disqualifies_slow_without_clear_gain():
    ranked = apply_ranking_rule(
        [
            {"config": "12x1", "selected_tok_per_sec": 1000, "min_val_bpb": 4.10, "unstable": False},
            {"config": "2x5", "selected_tok_per_sec": 930, "min_val_bpb": 4.09, "unstable": False},
            {"config": "1x10", "selected_tok_per_sec": 980, "min_val_bpb": 4.20, "unstable": True},
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
            {"config": "12x1", "selected_tok_per_sec": 1000, "min_val_bpb": 4.10, "unstable": False},
            {"config": "6x2", "selected_tok_per_sec": 940, "min_val_bpb": 4.05, "unstable": False},
        ],
        slowdown_threshold_pct=5.0,
        clear_bpb_gain=0.02,
    )

    row = next(row for row in ranked if row["config"] == "6x2")
    assert row["qualified"] is True
    assert row["rank"] == 1


def test_format_ranking_table_emits_markdown_rows():
    rows = [
        {
            "rank": 1,
            "config": "12x1",
            "selected_tok_per_sec": 1000,
            "min_val_bpb": 4.1,
            "qualified": True,
            "disqualify_reason": None,
        },
        {
            "rank": None,
            "config": "2x5",
            "selected_tok_per_sec": 900,
            "min_val_bpb": 4.2,
            "qualified": False,
            "disqualify_reason": "slow>5.0%",
        },
    ]
    table = format_ranking_table(rows)
    assert "| Rank | Config | tok/sec | vs 12x1 | min val bpb | Status |" in table
    assert "| 1 | 12x1 | 1,000 | +0.0% | 4.1000 | qualified |" in table
    assert "| - | 2x5 | 900 | -10.0% | 4.2000 | disqualified (slow>5.0%) |" in table
