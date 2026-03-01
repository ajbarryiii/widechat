import json
from pathlib import Path

import pytest

from scripts import plan_stage2_long_runs


def _write_finalists(path: Path) -> None:
    payload = {
        "source": "artifacts/pilot/pilot_ranked_runs.json",
        "max_finalists": 2,
        "selected_finalists": [
            {"config": "4x3", "depth": 4, "n_branches": 3, "aspect_ratio": 192},
            {"config": "12x1", "depth": 12, "n_branches": 1, "aspect_ratio": 64},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_main_writes_plan_and_runbook(tmp_path, monkeypatch, capsys):
    finalists_json = tmp_path / "stage2_finalists.json"
    _write_finalists(finalists_json)

    plan_json = tmp_path / "stage2_long_runs_plan.json"
    runbook_md = tmp_path / "stage2_long_runs_runbook.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_stage2_long_runs.py",
            "--finalists-json",
            str(finalists_json),
            "--token-budgets",
            "1000,2000",
            "--total-batch-size",
            "512",
            "--output-plan-json",
            str(plan_json),
            "--output-runbook-md",
            str(runbook_md),
            "--extra-args",
            "--eval-every 100",
            "--run-prefix",
            "longrun",
        ],
    )

    plan_stage2_long_runs.main()

    payload = json.loads(plan_json.read_text(encoding="utf-8"))
    assert payload["source"] == str(finalists_json)
    assert payload["token_budgets"] == [1000, 2000]
    assert payload["total_batch_size"] == 512
    assert len(payload["runs"]) == 4
    first_run = payload["runs"][0]
    assert first_run["config"] == "4x3"
    assert first_run["num_iterations"] == 2
    assert first_run["model_tag"] == "longrun_4x3_tok1000"
    assert "--depth 4 --n-branches 3 --aspect-ratio 192" in first_run["command"]
    assert "--num-iterations 2" in first_run["command"]
    assert "--eval-every 100" in first_run["command"]

    runbook = runbook_md.read_text(encoding="utf-8")
    assert "# Stage 2 Long-Run Training Runbook" in runbook
    assert "python -m scripts.base_train --depth 4 --n-branches 3" in runbook

    stdout = capsys.readouterr().out
    assert "stage2_long_run_plan_ok" in stdout
    assert "finalists=2" in stdout
    assert "runs=4" in stdout


def test_main_dry_run_writes_no_artifacts(tmp_path, monkeypatch, capsys):
    finalists_json = tmp_path / "stage2_finalists.json"
    _write_finalists(finalists_json)
    plan_json = tmp_path / "stage2_long_runs_plan.json"
    runbook_md = tmp_path / "stage2_long_runs_runbook.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_stage2_long_runs.py",
            "--finalists-json",
            str(finalists_json),
            "--dry-run",
            "--output-plan-json",
            str(plan_json),
            "--output-runbook-md",
            str(runbook_md),
        ],
    )

    plan_stage2_long_runs.main()

    assert not plan_json.exists()
    assert not runbook_md.exists()
    stdout = capsys.readouterr().out
    assert "stage2_long_run_plan_dry_run_ok" in stdout


def test_main_rejects_invalid_token_budget(tmp_path, monkeypatch):
    finalists_json = tmp_path / "stage2_finalists.json"
    _write_finalists(finalists_json)

    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_stage2_long_runs.py",
            "--finalists-json",
            str(finalists_json),
            "--token-budgets",
            "1000,oops",
        ],
    )

    with pytest.raises(ValueError, match="invalid token budget 'oops'"):
        plan_stage2_long_runs.main()
