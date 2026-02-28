from nanochat.throughput_benchmark import ThroughputTarget, build_train_command, format_markdown_table, parse_train_output


def test_build_train_command_contains_required_config_flags():
    target = ThroughputTarget(label="2x5", depth=2, n_branches=5, aspect_ratio=384)
    command = build_train_command(
        target=target,
        python_exe="python",
        max_seq_len=1024,
        total_batch_size=65536,
        device_batch_size=8,
        num_iterations=30,
        device_type="cuda",
        extra_args=["--head-dim", "128"],
    )

    assert "--depth" in command and command[command.index("--depth") + 1] == "2"
    assert "--n-branches" in command and command[command.index("--n-branches") + 1] == "5"
    assert "--aspect-ratio" in command and command[command.index("--aspect-ratio") + 1] == "384"
    assert "--target-param-data-ratio" in command and command[command.index("--target-param-data-ratio") + 1] == "-1"
    assert command[-2:] == ["--head-dim", "128"]


def test_parse_train_output_prefers_average_tok_per_sec():
    output = """
step 00039/00040 (97.50%) | loss: 4.0 | lrm: 0.5 | dt: 100.00ms | tok/sec: 327,680 | bf16_mfu: 12.50 | peak_mem: 1234.56MiB | epoch: 0 | total time: 0.50m
Peak memory usage: 1234.56MiB
Average tok/sec (post-warmup): 300,000
"""
    parsed = parse_train_output(output)
    assert parsed["avg_tok_per_sec"] == 300000
    assert parsed["final_tok_per_sec"] == 327680
    assert parsed["selected_tok_per_sec"] == 300000
    assert parsed["final_mfu"] == 12.5
    assert parsed["peak_memory_mib"] == 1234.56


def test_parse_train_output_falls_back_to_final_tok_per_sec():
    output = "step 00001/00002 (50.00%) | tok/sec: 12,345 | bf16_mfu: 1.23 | peak_mem: 10.00MiB"
    parsed = parse_train_output(output)
    assert parsed["avg_tok_per_sec"] is None
    assert parsed["final_tok_per_sec"] == 12345
    assert parsed["selected_tok_per_sec"] == 12345


def test_format_markdown_table_produces_pipe_table():
    table = format_markdown_table(
        [
            {
                "Config": "12x1",
                "tok/sec": "100",
                "vs 12x1": "+0.0%",
                "MFU": "1.00",
                "Peak mem (MiB)": "500.00",
            }
        ]
    )
    assert "| Config | tok/sec | vs 12x1 | MFU | Peak mem (MiB) |" in table
    assert "| 12x1 | 100 | +0.0% | 1.00 | 500.00 |" in table
