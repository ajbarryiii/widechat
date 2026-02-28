import json
from pathlib import Path

import pytest

from scripts import run_stage2_promotion_bundle as bundle


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
