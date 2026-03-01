import json
from pathlib import Path

import pytest

from scripts import run_stage2_long_run_bundle


def _write_finalists(path: Path) -> None:
    payload = {
        "source": "artifacts/pilot/pilot_ranked_runs.json",
        "max_finalists": 2,
        "selected_finalists": [
            {"config": "6x2", "depth": 6, "n_branches": 2, "aspect_ratio": 128},
            {"config": "12x1", "depth": 12, "n_branches": 1, "aspect_ratio": 64},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_main_writes_plan_runbook_and_command(tmp_path, monkeypatch, capsys):
    finalists_json = tmp_path / "stage2_finalists.json"
    _write_finalists(finalists_json)

    output_dir = tmp_path / "bundle"
    command_sh = output_dir / "stage2_long_run_bundle_command.sh"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_long_run_bundle.py",
            "--finalists-json",
            str(finalists_json),
            "--output-dir",
            str(output_dir),
            "--token-budgets",
            "1000",
            "--total-batch-size",
            "500",
            "--output-bundle-command-sh",
            str(command_sh),
        ],
    )

    run_stage2_long_run_bundle.main()

    plan_json = output_dir / "stage2_long_runs_plan.json"
    runbook_md = output_dir / "stage2_long_runs_runbook.md"
    assert plan_json.is_file()
    assert runbook_md.is_file()
    assert command_sh.is_file()

    plan_payload = json.loads(plan_json.read_text(encoding="utf-8"))
    assert len(plan_payload["runs"]) == 2
    assert plan_payload["runs"][0]["num_iterations"] == 2

    stdout = capsys.readouterr().out
    assert "stage2_long_run_bundle_ok" in stdout
    assert "runs=2" in stdout


def test_preflight_writes_receipt_only(tmp_path, monkeypatch, capsys):
    finalists_json = tmp_path / "stage2_finalists.json"
    _write_finalists(finalists_json)

    output_dir = tmp_path / "bundle"
    preflight_json = output_dir / "stage2_long_runs_preflight.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_long_run_bundle.py",
            "--finalists-json",
            str(finalists_json),
            "--output-dir",
            str(output_dir),
            "--preflight",
            "--output-preflight-json",
            str(preflight_json),
        ],
    )

    run_stage2_long_run_bundle.main()

    assert preflight_json.is_file()
    assert not (output_dir / "stage2_long_runs_plan.json").exists()
    assert not (output_dir / "stage2_long_runs_runbook.md").exists()

    payload = json.loads(preflight_json.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["run_count"] == 4

    stdout = capsys.readouterr().out
    assert "stage2_long_run_bundle_preflight_ok" in stdout


def test_failure_writes_blocked_receipt(tmp_path, monkeypatch, capsys):
    missing_finalists = tmp_path / "missing_stage2_finalists.json"
    blocked_md = tmp_path / "stage2_long_run_bundle_blocked.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_long_run_bundle.py",
            "--finalists-json",
            str(missing_finalists),
            "--output-blocked-md",
            str(blocked_md),
        ],
    )

    with pytest.raises(FileNotFoundError):
        run_stage2_long_run_bundle.main()

    assert blocked_md.is_file()
    blocked_text = blocked_md.read_text(encoding="utf-8")
    assert "# Stage 2 Long-Run Bundle Blocked" in blocked_text
    assert "error_type: `FileNotFoundError`" in blocked_text

    stdout = capsys.readouterr().out
    assert "stage2_long_run_bundle_blocked" in stdout
