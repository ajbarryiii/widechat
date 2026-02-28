import json
from pathlib import Path

import pytest

from scripts import pilot_promote


def test_load_ranked_runs_requires_list(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"ranked_runs": "nope"}), encoding="utf-8")

    with pytest.raises(ValueError, match="ranked_runs list"):
        pilot_promote._load_ranked_runs(str(path))


def test_load_ranked_runs_rejects_empty_list(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"ranked_runs": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="must not be empty"):
        pilot_promote._load_ranked_runs(str(path))


def test_load_ranked_runs_rejects_non_object_row(tmp_path):
    path = tmp_path / "invalid-row.json"
    path.write_text(json.dumps({"ranked_runs": ["bad"]}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"ranked_runs\[0\] must be a JSON object"):
        pilot_promote._load_ranked_runs(str(path))


def test_load_ranked_runs_rejects_missing_required_fields(tmp_path):
    path = tmp_path / "missing-fields.json"
    path.write_text(
        json.dumps(
            {
                "ranked_runs": [
                    {
                        "config": "12x1",
                        "depth": 12,
                        "n_branches": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"ranked_runs\[0\] missing positive integer field: aspect_ratio"):
        pilot_promote._load_ranked_runs(str(path))


def test_load_ranked_runs_rejects_missing_qualified_field(tmp_path):
    path = tmp_path / "missing-qualified.json"
    path.write_text(
        json.dumps(
            {
                "ranked_runs": [
                    {
                        "config": "12x1",
                        "depth": 12,
                        "n_branches": 1,
                        "aspect_ratio": 64,
                        "selected_tok_per_sec": 1000,
                        "min_val_bpb": 4.1,
                        "token_budget": 250000000,
                        "rank": 1,
                        "disqualify_reason": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"ranked_runs\[0\] missing boolean field: qualified"):
        pilot_promote._load_ranked_runs(str(path))


def test_load_ranked_runs_rejects_qualified_row_without_rank(tmp_path):
    path = tmp_path / "qualified-without-rank.json"
    path.write_text(
        json.dumps(
            {
                "ranked_runs": [
                    {
                        "config": "12x1",
                        "depth": 12,
                        "n_branches": 1,
                        "aspect_ratio": 64,
                        "selected_tok_per_sec": 1000,
                        "min_val_bpb": 4.1,
                        "token_budget": 250000000,
                        "qualified": True,
                        "rank": None,
                        "disqualify_reason": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"qualified row must include a positive integer rank"):
        pilot_promote._load_ranked_runs(str(path))


def test_load_ranked_runs_rejects_disqualified_row_without_reason(tmp_path):
    path = tmp_path / "disqualified-without-reason.json"
    path.write_text(
        json.dumps(
            {
                "ranked_runs": [
                    {
                        "config": "2x5",
                        "depth": 2,
                        "n_branches": 5,
                        "aspect_ratio": 384,
                        "selected_tok_per_sec": 900,
                        "min_val_bpb": 4.3,
                        "token_budget": 250000000,
                        "qualified": False,
                        "rank": None,
                        "disqualify_reason": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"disqualified row must include non-empty disqualify_reason"):
        pilot_promote._load_ranked_runs(str(path))


def test_main_writes_selected_finalists_outputs(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "pilot.json"
    output_json = tmp_path / "finalists.json"
    output_md = tmp_path / "finalists.md"

    ranked_runs = [
        {
            "rank": 1,
            "config": "12x1",
            "depth": 12,
            "n_branches": 1,
            "aspect_ratio": 64,
            "selected_tok_per_sec": 1000,
            "min_val_bpb": 4.1,
            "token_budget": 250000000,
            "qualified": True,
            "disqualify_reason": None,
        },
        {
            "rank": 2,
            "config": "6x2",
            "depth": 6,
            "n_branches": 2,
            "aspect_ratio": 128,
            "selected_tok_per_sec": 980,
            "min_val_bpb": 4.0,
            "token_budget": 250000000,
            "qualified": True,
            "disqualify_reason": None,
        },
        {
            "rank": None,
            "config": "2x5",
            "depth": 2,
            "n_branches": 5,
            "aspect_ratio": 384,
            "selected_tok_per_sec": 900,
            "min_val_bpb": 4.2,
            "token_budget": 250000000,
            "qualified": False,
            "disqualify_reason": "slow>5.0%",
        },
    ]
    input_path.write_text(json.dumps({"ranked_runs": ranked_runs}), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_promote.py",
            "--input-json",
            str(input_path),
            "--max-finalists",
            "1",
            "--min-finalists",
            "1",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
    )

    pilot_promote.main()

    stdout = capsys.readouterr().out
    assert "Selected finalists:" in stdout
    assert "Stage 2 depth/branch flags:" in stdout
    assert "--depth 12 --n-branches 1 --aspect-ratio 64" in stdout
    assert "--depth 6 --n-branches 2 --aspect-ratio 128" not in stdout

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["max_finalists"] == 1
    assert [row["config"] for row in payload["selected_finalists"]] == ["12x1"]

    md = output_md.read_text(encoding="utf-8")
    assert "## Stage 2 Finalists" in md
    assert "`--depth 12 --n-branches 1 --aspect-ratio 64`" in md
    assert "`--depth 6 --n-branches 2 --aspect-ratio 128`" not in md


def test_sample_artifacts_stay_in_sync_with_pilot_promote(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    ranked_runs_json = repo_root / "artifacts" / "pilot" / "sample_ranked_runs.json"
    expected_finalists_json = repo_root / "artifacts" / "pilot" / "sample_stage2_finalists.json"
    expected_finalists_md = repo_root / "artifacts" / "pilot" / "sample_stage2_finalists.md"

    generated_json = tmp_path / "generated_finalists.json"
    generated_md = tmp_path / "generated_finalists.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_promote.py",
            "--input-json",
            str(ranked_runs_json.relative_to(repo_root)),
            "--max-finalists",
            "3",
            "--output-json",
            str(generated_json),
            "--output-md",
            str(generated_md),
        ],
    )
    pilot_promote.main()

    assert json.loads(generated_json.read_text(encoding="utf-8")) == json.loads(
        expected_finalists_json.read_text(encoding="utf-8")
    )
    assert generated_md.read_text(encoding="utf-8") == expected_finalists_md.read_text(encoding="utf-8")


def test_main_rejects_when_not_enough_finalists(tmp_path, monkeypatch):
    input_path = tmp_path / "pilot.json"
    ranked_runs = [
        {
            "rank": 1,
            "config": "12x1",
            "depth": 12,
            "n_branches": 1,
            "aspect_ratio": 64,
            "selected_tok_per_sec": 1000,
            "min_val_bpb": 4.1,
            "token_budget": 250000000,
            "qualified": True,
            "disqualify_reason": None,
        },
        {
            "rank": None,
            "config": "2x5",
            "depth": 2,
            "n_branches": 5,
            "aspect_ratio": 384,
            "selected_tok_per_sec": 900,
            "min_val_bpb": 4.2,
            "token_budget": 250000000,
            "qualified": False,
            "disqualify_reason": "slow>5.0%",
        },
    ]
    input_path.write_text(json.dumps({"ranked_runs": ranked_runs}), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_promote.py",
            "--input-json",
            str(input_path),
        ],
    )

    with pytest.raises(RuntimeError, match="expected at least 2 qualified finalists"):
        pilot_promote.main()


def test_main_rejects_invalid_finalist_bounds(tmp_path, monkeypatch):
    input_path = tmp_path / "pilot.json"
    ranked_runs = [
        {
            "rank": 1,
            "config": "12x1",
            "depth": 12,
            "n_branches": 1,
            "aspect_ratio": 64,
            "selected_tok_per_sec": 1000,
            "min_val_bpb": 4.1,
            "token_budget": 250000000,
            "qualified": True,
            "disqualify_reason": None,
        }
    ]
    input_path.write_text(json.dumps({"ranked_runs": ranked_runs}), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "pilot_promote.py",
            "--input-json",
            str(input_path),
            "--min-finalists",
            "3",
            "--max-finalists",
            "2",
        ],
    )

    with pytest.raises(ValueError, match="--min-finalists must be <= --max-finalists"):
        pilot_promote.main()
