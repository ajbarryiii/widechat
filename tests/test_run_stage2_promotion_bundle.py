import json
from pathlib import Path

import pytest

from scripts import run_stage2_promotion_bundle as bundle
from scripts import pilot_promote


def test_main_writes_stage2_finalists_bundle(tmp_path, monkeypatch, capsys):
    input_json = tmp_path / "ranked_runs.json"
    output_dir = tmp_path / "artifacts"
    ranked_runs = {
        "ranked_runs": [
            {
                "config": "4x3",
                "depth": 4,
                "n_branches": 3,
                "aspect_ratio": 192,
                "selected_tok_per_sec": 572110.0,
                "min_val_bpb": 4.0123,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 1,
                "disqualify_reason": None,
            },
            {
                "config": "12x1",
                "depth": 12,
                "n_branches": 1,
                "aspect_ratio": 64,
                "selected_tok_per_sec": 565800.0,
                "min_val_bpb": 4.0310,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 2,
                "disqualify_reason": None,
            },
            {
                "config": "2x5",
                "depth": 2,
                "n_branches": 5,
                "aspect_ratio": 384,
                "selected_tok_per_sec": 525100.0,
                "min_val_bpb": 4.0388,
                "token_budget": 250000000,
                "qualified": False,
                "rank": None,
                "disqualify_reason": "slow>5.0%",
            },
        ]
    }
    input_json.write_text(json.dumps(ranked_runs), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(output_dir),
            "--max-finalists",
            "1",
            "--min-finalists",
            "1",
        ],
    )

    bundle.main()

    finalists_json = output_dir / "stage2_finalists.json"
    finalists_md = output_dir / "stage2_finalists.md"
    payload = json.loads(finalists_json.read_text(encoding="utf-8"))
    assert payload["source"] == str(input_json)
    assert payload["source_sha256"] == pilot_promote._stable_json_sha256(ranked_runs)
    assert payload["max_finalists"] == 1
    assert [row["config"] for row in payload["selected_finalists"]] == ["4x3"]

    body = finalists_md.read_text(encoding="utf-8")
    assert "## Stage 2 Finalists" in body
    assert "`4x3`: `--depth 4 --n-branches 3 --aspect-ratio 192`" in body
    assert "`12x1`: `--depth 12 --n-branches 1 --aspect-ratio 64`" not in body

    stdout = capsys.readouterr().out
    assert "bundle_ok finalists=1" in stdout


def test_main_rejects_when_no_qualified_finalists(tmp_path, monkeypatch):
    input_json = tmp_path / "ranked_runs.json"
    ranked_runs = {
        "ranked_runs": [
            {
                "config": "2x5",
                "depth": 2,
                "n_branches": 5,
                "aspect_ratio": 384,
                "selected_tok_per_sec": 525100.0,
                "min_val_bpb": 4.0388,
                "token_budget": 250000000,
                "qualified": False,
                "rank": None,
                "disqualify_reason": "slow>5.0%",
            }
        ]
    }
    input_json.write_text(json.dumps(ranked_runs), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ],
    )

    with pytest.raises(RuntimeError, match="expected at least 2 qualified finalists"):
        bundle.main()


def test_main_rejects_invalid_finalist_bounds(tmp_path, monkeypatch):
    input_json = tmp_path / "ranked_runs.json"
    ranked_runs = {
        "ranked_runs": [
            {
                "config": "4x3",
                "depth": 4,
                "n_branches": 3,
                "aspect_ratio": 192,
                "selected_tok_per_sec": 572110.0,
                "min_val_bpb": 4.0123,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 1,
                "disqualify_reason": None,
            }
        ]
    }
    input_json.write_text(json.dumps(ranked_runs), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--min-finalists",
            "3",
            "--max-finalists",
            "2",
        ],
    )

    with pytest.raises(ValueError, match="--min-finalists must be <= --max-finalists"):
        bundle.main()


def test_main_require_real_input_rejects_sample_fixture(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    ranked_runs_json = repo_root / "artifacts" / "pilot" / "sample_ranked_runs.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(ranked_runs_json),
            "--output-dir",
            str(repo_root / "artifacts" / "pilot"),
            "--min-finalists",
            "1",
            "--require-real-input",
        ],
    )

    with pytest.raises(ValueError, match="--require-real-input rejects sample/fixture"):
        bundle.main()


def test_main_writes_runbook_when_requested(tmp_path, monkeypatch, capsys):
    input_json = tmp_path / "ranked_runs.json"
    output_dir = tmp_path / "artifacts"
    runbook_md = tmp_path / "docs" / "stage2_runbook.md"
    ranked_runs = {
        "ranked_runs": [
            {
                "config": "4x3",
                "depth": 4,
                "n_branches": 3,
                "aspect_ratio": 192,
                "selected_tok_per_sec": 572110.0,
                "min_val_bpb": 4.0123,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 1,
                "disqualify_reason": None,
            },
            {
                "config": "12x1",
                "depth": 12,
                "n_branches": 1,
                "aspect_ratio": 64,
                "selected_tok_per_sec": 565800.0,
                "min_val_bpb": 4.0310,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 2,
                "disqualify_reason": None,
            },
        ]
    }
    input_json.write_text(json.dumps(ranked_runs), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(output_dir),
            "--min-finalists",
            "1",
            "--max-finalists",
            "2",
            "--require-real-input",
            "--output-runbook-md",
            str(runbook_md),
        ],
    )

    bundle.main()

    runbook = runbook_md.read_text(encoding="utf-8")
    assert "# Stage 2 Promotion Bundle Runbook" in runbook
    assert "python -m scripts.run_stage2_promotion_bundle" in runbook
    assert f"--input-json {input_json}" in runbook
    assert f"--output-dir {output_dir}" in runbook
    assert "--require-real-input" in runbook
    assert "python -m scripts.run_pilot_check_in" in runbook
    assert f"--ranked-json {input_json}" in runbook
    assert f"--finalists-json {output_dir / 'stage2_finalists.json'}" in runbook
    assert f"--finalists-md {output_dir / 'stage2_finalists.md'}" in runbook
    assert f"--output-check-json {output_dir / 'pilot_bundle_check.json'}" in runbook

    stdout = capsys.readouterr().out
    assert f"runbook_md={runbook_md}" in stdout


def test_main_runs_strict_check_in_when_requested(tmp_path, monkeypatch, capsys):
    input_json = tmp_path / "ranked_runs.json"
    output_dir = tmp_path / "artifacts"
    output_check_json = tmp_path / "receipts" / "stage2_check.json"
    ranked_runs = {
        "ranked_runs": [
            {
                "config": "4x3",
                "depth": 4,
                "n_branches": 3,
                "aspect_ratio": 192,
                "selected_tok_per_sec": 572110.0,
                "min_val_bpb": 4.0123,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 1,
                "disqualify_reason": None,
            },
            {
                "config": "12x1",
                "depth": 12,
                "n_branches": 1,
                "aspect_ratio": 64,
                "selected_tok_per_sec": 565800.0,
                "min_val_bpb": 4.0310,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 2,
                "disqualify_reason": None,
            },
        ]
    }
    input_json.write_text(json.dumps(ranked_runs), encoding="utf-8")

    check_call: dict[str, object] = {}

    def _fake_run_pilot_bundle_check(**kwargs):
        check_call.update(kwargs)
        return 2

    monkeypatch.setattr(bundle, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(output_dir),
            "--min-finalists",
            "1",
            "--max-finalists",
            "2",
            "--run-check-in",
            "--output-check-json",
            str(output_check_json),
        ],
    )

    bundle.main()

    assert check_call["ranked_json_path"] == Path(input_json)
    assert check_call["finalists_json_path"] == output_dir / "stage2_finalists.json"
    assert check_call["finalists_md_path"] == output_dir / "stage2_finalists.md"
    assert check_call["check_in"] is True
    assert check_call["output_check_json"] == str(output_check_json)

    stdout = capsys.readouterr().out
    assert f"check_json={output_check_json}" in stdout


def test_runbook_includes_check_in_flags_when_enabled(tmp_path):
    runbook_md = tmp_path / "stage2_runbook.md"
    check_json = tmp_path / "checks" / "bundle.json"

    bundle._write_runbook_md(
        path=runbook_md,
        input_json="artifacts/pilot/pilot_ranked_runs.json",
        output_dir="artifacts/pilot",
        finalists_json=Path("artifacts/pilot/stage2_finalists.json"),
        finalists_md=Path("artifacts/pilot/stage2_finalists.md"),
        min_finalists=2,
        max_finalists=3,
        require_real_input=True,
        run_check_in=True,
        output_check_json=str(check_json),
    )

    runbook = runbook_md.read_text(encoding="utf-8")
    assert "--require-real-input" in runbook
    assert "--run-check-in" in runbook
    assert f"--output-check-json {check_json}" in runbook
