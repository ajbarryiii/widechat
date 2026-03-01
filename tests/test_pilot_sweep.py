import json
import pytest
from types import SimpleNamespace

from nanochat import pilot_sweep
from scripts import pilot_sweep as pilot_sweep_script
from scripts.pilot_sweep import (
    _artifact_paths,
    _load_existing_run_artifact,
    _resolve_selected_targets,
    _sanitize_label,
    _validate_resume_run_artifact,
    _write_run_artifacts,
)
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


def test_sanitize_label_replaces_unsafe_chars():
    assert _sanitize_label("2x5/gpu run") == "2x5-gpu-run"
    assert _sanitize_label("!!!") == "run"


def test_resolve_selected_targets_defaults_to_full_grid():
    resolved = _resolve_selected_targets([])
    assert [target.label for _index, target in resolved] == [target.label for target in pilot_sweep.DEFAULT_PILOT_TARGETS]
    assert [index for index, _target in resolved] == list(range(1, len(pilot_sweep.DEFAULT_PILOT_TARGETS) + 1))


def test_resolve_selected_targets_preserves_canonical_order_and_indices():
    resolved = _resolve_selected_targets(["4x3", "12x1", "1x10"])
    assert [(index, target.label) for index, target in resolved] == [
        (1, "12x1"),
        (3, "4x3"),
        (7, "1x10"),
    ]


def test_resolve_selected_targets_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown --target labels"):
        _resolve_selected_targets(["99x99"])


def test_resolve_selected_targets_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate --target labels"):
        _resolve_selected_targets(["12x1", "12x1"])


def test_write_run_artifacts_writes_log_and_json(tmp_path):
    run_result = {
        "config": "2x5",
        "selected_tok_per_sec": 1234,
        "unstable": False,
    }
    _write_run_artifacts(
        artifacts_dir=str(tmp_path),
        run_index=3,
        run_result=run_result,
        output_text="training output",
    )

    log_path = tmp_path / "03-2x5.log"
    json_path = tmp_path / "03-2x5.json"
    assert log_path.read_text(encoding="utf-8") == "training output"
    assert json.loads(json_path.read_text(encoding="utf-8")) == run_result


def test_load_existing_run_artifact_returns_none_when_missing(tmp_path):
    loaded = _load_existing_run_artifact(str(tmp_path), run_index=1, config_label="12x1")
    assert loaded is None


def test_load_existing_run_artifact_rejects_config_mismatch(tmp_path):
    _log_path, metrics_path = _artifact_paths(str(tmp_path), run_index=1, config_label="12x1")
    tmp_path.mkdir(exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"config": "6x2", "selected_tok_per_sec": 1000, "unstable": False}, f)

    with pytest.raises(ValueError, match="artifact config mismatch"):
        _load_existing_run_artifact(str(tmp_path), run_index=1, config_label="12x1")


def test_validate_resume_run_artifact_requires_log_file(tmp_path):
    _log_path, metrics_path = _artifact_paths(str(tmp_path), run_index=1, config_label="12x1")
    tmp_path.mkdir(exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": "12x1",
                "selected_tok_per_sec": 1000,
                "unstable": False,
                "token_budget": 100000,
            },
            f,
        )

    loaded = _load_existing_run_artifact(str(tmp_path), run_index=1, config_label="12x1")
    assert loaded is not None

    with pytest.raises(ValueError, match="expected log file"):
        _validate_resume_run_artifact(
            loaded,
            artifacts_dir=str(tmp_path),
            run_index=1,
            config_label="12x1",
            expected_token_budget=100000,
        )


def test_validate_resume_run_artifact_requires_expected_token_budget(tmp_path):
    run_result = {
        "config": "12x1",
        "selected_tok_per_sec": 1000,
        "unstable": False,
        "token_budget": 90000,
    }
    _write_run_artifacts(
        artifacts_dir=str(tmp_path),
        run_index=1,
        run_result=run_result,
        output_text="pilot output",
    )
    loaded = _load_existing_run_artifact(str(tmp_path), run_index=1, config_label="12x1")
    assert loaded is not None

    with pytest.raises(ValueError, match="token_budget mismatch"):
        _validate_resume_run_artifact(
            loaded,
            artifacts_dir=str(tmp_path),
            run_index=1,
            config_label="12x1",
            expected_token_budget=100000,
        )


def test_main_resume_from_artifacts_reuses_saved_runs(tmp_path, monkeypatch):
    total_batch_size = 1000
    num_iterations = 100
    token_budget = total_batch_size * num_iterations

    for index, target in enumerate(pilot_sweep.DEFAULT_PILOT_TARGETS, start=1):
        _write_run_artifacts(
            artifacts_dir=str(tmp_path),
            run_index=index,
            run_result={
                "config": target.label,
                "selected_tok_per_sec": 1000 - index,
                "min_val_bpb": 4.0 + index * 0.001,
                "unstable": False,
                "token_budget": token_budget,
            },
            output_text=f"{target.label} output",
        )

    def fail_if_run_single_pilot(_command):
        raise AssertionError("run_single_pilot should not be called when all artifacts exist")

    monkeypatch.setattr(pilot_sweep_script, "run_single_pilot", fail_if_run_single_pilot)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            str(total_batch_size),
            "--device-batch-size",
            "1",
            "--pilot-tokens",
            str(token_budget),
            "--eval-every",
            "50",
            "--artifacts-dir",
            str(tmp_path),
            "--resume-from-artifacts",
        ],
    )

    pilot_sweep_script.main()


def test_main_resume_from_artifacts_requires_artifacts_dir(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            "1000",
            "--device-batch-size",
            "1",
            "--pilot-tokens",
            "100000",
            "--eval-every",
            "50",
            "--resume-from-artifacts",
        ],
    )

    with pytest.raises(ValueError, match="requires --artifacts-dir"):
        pilot_sweep_script.main()


def test_main_output_runbook_requires_artifacts_dir(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            "1000",
            "--device-batch-size",
            "1",
            "--pilot-tokens",
            "100000",
            "--eval-every",
            "50",
            "--output-runbook-md",
            "runbook.md",
            "--dry-run",
        ],
    )

    with pytest.raises(ValueError, match="--output-runbook-md requires --artifacts-dir"):
        pilot_sweep_script.main()


def test_main_output_preflight_json_requires_preflight(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            "1000",
            "--device-batch-size",
            "1",
            "--pilot-tokens",
            "100000",
            "--eval-every",
            "50",
            "--output-preflight-json",
            "preflight.json",
        ],
    )

    with pytest.raises(ValueError, match="--output-preflight-json requires --preflight"):
        pilot_sweep_script.main()


def test_main_preflight_writes_receipt_and_skips_execution(tmp_path, monkeypatch):
    preflight_json = tmp_path / "pilot_preflight.json"

    def fail_if_run_single_pilot(_command):
        raise AssertionError("run_single_pilot should not be called in preflight mode")

    monkeypatch.setattr(pilot_sweep_script, "run_single_pilot", fail_if_run_single_pilot)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            "1000",
            "--device-batch-size",
            "1",
            "--pilot-tokens",
            "100000",
            "--eval-every",
            "50",
            "--target",
            "12x1",
            "--preflight",
            "--output-preflight-json",
            str(preflight_json),
        ],
    )

    pilot_sweep_script.main()

    receipt = json.loads(preflight_json.read_text(encoding="utf-8"))
    assert receipt["ok"] is True
    assert receipt["is_full_grid"] is False
    assert receipt["errors"] == []
    assert [row["config"] for row in receipt["targets"]] == ["12x1"]


def test_main_preflight_resume_artifact_failure_writes_receipt(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    preflight_json = tmp_path / "preflight.json"
    _log_path, metrics_path = _artifact_paths(str(artifacts_dir), run_index=1, config_label="12x1")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": "12x1",
                "selected_tok_per_sec": 1000,
                "unstable": False,
                "token_budget": 100000,
            },
            f,
        )

    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            "1000",
            "--device-batch-size",
            "1",
            "--pilot-tokens",
            "100000",
            "--eval-every",
            "50",
            "--artifacts-dir",
            str(artifacts_dir),
            "--resume-from-artifacts",
            "--target",
            "12x1",
            "--preflight",
            "--output-preflight-json",
            str(preflight_json),
        ],
    )

    with pytest.raises(ValueError, match="pilot sweep preflight failed"):
        pilot_sweep_script.main()

    receipt = json.loads(preflight_json.read_text(encoding="utf-8"))
    assert receipt["ok"] is False
    assert any("expected log file" in err for err in receipt["errors"])


def test_main_preflight_failure_writes_blocked_markdown(tmp_path, monkeypatch):
    blocked_md = tmp_path / "pilot_sweep_blocked.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            "1000",
            "--device-batch-size",
            "1",
            "--pilot-tokens",
            "100000",
            "--eval-every",
            "5",
            "--target",
            "12x1",
            "--preflight",
            "--output-blocked-md",
            str(blocked_md),
        ],
    )

    with pytest.raises(ValueError, match="pilot sweep preflight failed"):
        pilot_sweep_script.main()

    blocked_text = blocked_md.read_text(encoding="utf-8")
    assert "# Pilot Sweep Blocked" in blocked_text
    assert "- mode: `preflight`" in blocked_text
    assert "pilot sweep preflight failed" in blocked_text


def test_main_runtime_failure_writes_blocked_markdown(tmp_path, monkeypatch):
    blocked_md = tmp_path / "pilot_sweep_runtime_blocked.md"

    def raise_runtime_error(_command):
        raise RuntimeError("target GPU is unavailable")

    monkeypatch.setattr(pilot_sweep_script, "run_single_pilot", raise_runtime_error)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            "1000",
            "--device-batch-size",
            "1",
            "--pilot-tokens",
            "100000",
            "--eval-every",
            "50",
            "--target",
            "12x1",
            "--output-blocked-md",
            str(blocked_md),
        ],
    )

    with pytest.raises(RuntimeError, match="target GPU is unavailable"):
        pilot_sweep_script.main()

    blocked_text = blocked_md.read_text(encoding="utf-8")
    assert "# Pilot Sweep Blocked" in blocked_text
    assert "- mode: `run`" in blocked_text
    assert "target GPU is unavailable" in blocked_text


def test_main_writes_finalists_artifacts_from_ranked_runs(tmp_path, monkeypatch):
    total_batch_size = 1000
    num_iterations = 100
    token_budget = total_batch_size * num_iterations

    for index, target in enumerate(pilot_sweep.DEFAULT_PILOT_TARGETS, start=1):
        selected_tok_per_sec = 1000 - index
        min_val_bpb = 4.20 + index * 0.01
        unstable = False
        if target.label == "12x1":
            selected_tok_per_sec = 1000
            min_val_bpb = 4.20
        elif target.label == "6x2":
            selected_tok_per_sec = 990
            min_val_bpb = 4.05
        elif target.label == "4x3":
            selected_tok_per_sec = 992
            min_val_bpb = 4.00
        elif target.label == "2x5":
            selected_tok_per_sec = 900
            min_val_bpb = 4.30

        _write_run_artifacts(
            artifacts_dir=str(tmp_path),
            run_index=index,
            run_result={
                "config": target.label,
                "selected_tok_per_sec": selected_tok_per_sec,
                "min_val_bpb": min_val_bpb,
                "unstable": unstable,
                "token_budget": token_budget,
            },
            output_text=f"{target.label} output",
        )

    finalists_json = tmp_path / "finalists.json"
    finalists_md = tmp_path / "finalists.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            str(total_batch_size),
            "--device-batch-size",
            "1",
            "--pilot-tokens",
            str(token_budget),
            "--eval-every",
            "50",
            "--artifacts-dir",
            str(tmp_path),
            "--resume-from-artifacts",
            "--max-finalists",
            "2",
            "--output-finalists-json",
            str(finalists_json),
            "--output-finalists-md",
            str(finalists_md),
        ],
    )

    pilot_sweep_script.main()

    finalists_payload = json.loads(finalists_json.read_text(encoding="utf-8"))
    assert finalists_payload["source"] == str(tmp_path / "pilot_ranked_runs.json")
    assert isinstance(finalists_payload["source_sha256"], str)
    assert len(finalists_payload["source_sha256"]) == 64
    assert finalists_payload["max_finalists"] == 2
    assert [row["config"] for row in finalists_payload["selected_finalists"]] == ["4x3", "6x2"]

    finalists_md_text = finalists_md.read_text(encoding="utf-8")
    assert "## Stage 2 Finalists" in finalists_md_text
    assert "`4x3`: `--depth 4 --n-branches 3 --aspect-ratio 192`" in finalists_md_text
    assert "`6x2`: `--depth 6 --n-branches 2 --aspect-ratio 128`" in finalists_md_text
    assert "`12x1`: `--depth 12 --n-branches 1 --aspect-ratio 64`" not in finalists_md_text


def test_main_dry_run_writes_pilot_runbook(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    runbook_path = tmp_path / "pilot_runbook.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            "1024",
            "--device-batch-size",
            "2",
            "--pilot-tokens",
            "102400",
            "--eval-every",
            "50",
            "--eval-tokens",
            "4096",
            "--artifacts-dir",
            str(artifacts_dir),
            "--output-runbook-md",
            str(runbook_path),
            "--extra-arg=--compile",
            "--dry-run",
        ],
    )

    pilot_sweep_script.main()

    runbook = runbook_path.read_text(encoding="utf-8")
    assert "## Pilot Sweep Runbook" in runbook
    assert "-m scripts.pilot_sweep" in runbook
    assert "--resume-from-artifacts" in runbook
    assert "-m scripts.run_pilot_check_in" in runbook
    assert f"`{artifacts_dir / 'pilot_ranked_runs.json'}`" in runbook
    assert f"`{artifacts_dir / 'stage2_finalists.json'}`" in runbook
    assert "--extra-arg --compile" in runbook


def test_main_dry_run_writes_launch_manifest_json(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    manifest_path = tmp_path / "receipts" / "pilot_launch_manifest.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            "1024",
            "--device-batch-size",
            "2",
            "--pilot-tokens",
            "102400",
            "--eval-every",
            "50",
            "--artifacts-dir",
            str(artifacts_dir),
            "--target",
            "4x3",
            "--target",
            "1x10",
            "--dry-run",
            "--output-launch-manifest-json",
            str(manifest_path),
        ],
    )

    pilot_sweep_script.main()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["is_full_grid"] is False
    assert payload["resume_from_artifacts"] is False
    assert payload["preflight"] is False
    assert payload["dry_run"] is True
    assert payload["artifacts_dir"] == str(artifacts_dir)
    assert isinstance(payload["generated_at_utc"], str)

    targets = payload["targets"]
    assert [row["index"] for row in targets] == [3, 7]
    assert [row["config"] for row in targets] == ["4x3", "1x10"]
    assert all(isinstance(row["command_shell"], str) and row["command_shell"] for row in targets)

    assert targets[0]["log_path"] == str(artifacts_dir / "03-4x3.log")
    assert targets[0]["metrics_path"] == str(artifacts_dir / "03-4x3.json")
    assert targets[1]["log_path"] == str(artifacts_dir / "07-1x10.log")
    assert targets[1]["metrics_path"] == str(artifacts_dir / "07-1x10.json")


def test_main_dry_run_creates_parent_dirs_for_runbook(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    runbook_path = tmp_path / "nested" / "runbooks" / "pilot_runbook.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            "1024",
            "--device-batch-size",
            "2",
            "--pilot-tokens",
            "102400",
            "--eval-every",
            "50",
            "--artifacts-dir",
            str(artifacts_dir),
            "--output-runbook-md",
            str(runbook_path),
            "--dry-run",
        ],
    )

    pilot_sweep_script.main()
    assert runbook_path.exists()


def test_main_resume_creates_parent_dirs_for_ranking_and_finalists_outputs(tmp_path, monkeypatch):
    total_batch_size = 1000
    num_iterations = 100
    token_budget = total_batch_size * num_iterations

    for index, target in enumerate(pilot_sweep.DEFAULT_PILOT_TARGETS, start=1):
        _write_run_artifacts(
            artifacts_dir=str(tmp_path),
            run_index=index,
            run_result={
                "config": target.label,
                "selected_tok_per_sec": 1000 - index,
                "min_val_bpb": 4.0 + index * 0.001,
                "unstable": False,
                "token_budget": token_budget,
            },
            output_text=f"{target.label} output",
        )

    output_root = tmp_path / "nested" / "reports"
    ranked_json = output_root / "pilot_ranked_runs.json"
    ranking_md = output_root / "pilot_ranking.md"
    finalists_json = output_root / "stage2_finalists.json"
    finalists_md = output_root / "stage2_finalists.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            str(total_batch_size),
            "--device-batch-size",
            "1",
            "--pilot-tokens",
            str(token_budget),
            "--eval-every",
            "50",
            "--artifacts-dir",
            str(tmp_path),
            "--resume-from-artifacts",
            "--output-json",
            str(ranked_json),
            "--output-md",
            str(ranking_md),
            "--output-finalists-json",
            str(finalists_json),
            "--output-finalists-md",
            str(finalists_md),
        ],
    )

    pilot_sweep_script.main()

    assert ranked_json.exists()
    assert ranking_md.exists()
    assert finalists_json.exists()
    assert finalists_md.exists()


def test_main_partial_targets_skip_ranking_and_preserve_global_artifact_indices(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            "1000",
            "--device-batch-size",
            "1",
            "--pilot-tokens",
            "100000",
            "--eval-every",
            "50",
            "--artifacts-dir",
            str(tmp_path),
            "--target",
            "4x3",
            "--target",
            "1x10",
        ],
    )

    def fake_run_single_pilot(_command):
        output = "pilot output"
        return output, {
            "selected_tok_per_sec": 1000,
            "min_val_bpb": 4.0,
            "unstable": False,
            "command_failed": False,
            "failure_returncode": None,
        }

    monkeypatch.setattr(pilot_sweep_script, "run_single_pilot", fake_run_single_pilot)

    pilot_sweep_script.main()

    assert (tmp_path / "03-4x3.json").exists()
    assert (tmp_path / "03-4x3.log").exists()
    assert (tmp_path / "07-1x10.json").exists()
    assert (tmp_path / "07-1x10.log").exists()
    assert not (tmp_path / "pilot_ranked_runs.json").exists()


def test_main_partial_targets_reject_ranking_artifact_outputs(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_sweep.py",
            "--total-batch-size",
            "1000",
            "--device-batch-size",
            "1",
            "--pilot-tokens",
            "100000",
            "--eval-every",
            "50",
            "--target",
            "6x2",
            "--output-json",
            "ranked.json",
        ],
    )

    with pytest.raises(ValueError, match="partial --target runs cannot emit ranking/finalist artifacts"):
        pilot_sweep_script.main()
