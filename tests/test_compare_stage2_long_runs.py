import json

import pytest

from scripts import compare_stage2_long_runs


def _write_runs(path):
    payload = {
        "runs": [
            {
                "config": "12x1",
                "token_budget": 1_000_000_000,
                "final_val_bpb": 3.95,
                "min_val_bpb": 3.90,
                "selected_tok_per_sec": 560000.0,
                "unstable": False,
            },
            {
                "config": "4x3",
                "token_budget": 1_000_000_000,
                "final_val_bpb": 3.92,
                "min_val_bpb": 3.88,
                "selected_tok_per_sec": 575000.0,
                "unstable": False,
            },
            {
                "config": "6x2",
                "token_budget": 1_000_000_000,
                "final_val_bpb": 3.96,
                "min_val_bpb": 3.91,
                "selected_tok_per_sec": 570000.0,
                "unstable": False,
            },
            {
                "config": "12x1",
                "token_budget": 2_000_000_000,
                "final_val_bpb": 3.87,
                "min_val_bpb": 3.84,
                "selected_tok_per_sec": 558000.0,
                "unstable": False,
            },
            {
                "config": "4x3",
                "token_budget": 2_000_000_000,
                "final_val_bpb": 3.82,
                "min_val_bpb": 3.80,
                "selected_tok_per_sec": 572000.0,
                "unstable": False,
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_main_writes_json_and_markdown(tmp_path, monkeypatch, capsys):
    input_json = tmp_path / "stage2_runs.json"
    _write_runs(input_json)
    output_json = tmp_path / "compare.json"
    output_md = tmp_path / "compare.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_stage2_long_runs.py",
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
    )

    compare_stage2_long_runs.main()

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["source"] == str(input_json)
    assert report["baseline_config"] == "12x1"
    assert report["token_budgets"] == [1_000_000_000, 2_000_000_000]
    assert report["winners_by_budget"][0]["best_config"] == "4x3"
    assert report["winners_by_budget"][1]["best_config"] == "4x3"
    assert report["max_token_budget"] == 2_000_000_000
    assert report["top_at_max_budget"][0]["config"] == "4x3"
    assert report["top_at_max_budget"][1]["config"] == "12x1"

    markdown = output_md.read_text(encoding="utf-8")
    assert "# Stage 2 Baseline Comparison" in markdown
    assert "| token_budget | best_config |" in markdown
    assert "| 2000000000 | 4x3 |" in markdown
    assert "| config | token_budget | unstable |" in markdown

    stdout = capsys.readouterr().out
    assert "stage2_compare_ok" in stdout
    assert f"output_json={output_json}" in stdout


def test_main_rejects_missing_baseline_per_budget(tmp_path, monkeypatch):
    input_json = tmp_path / "stage2_runs_missing_baseline.json"
    payload = {
        "runs": [
            {
                "config": "12x1",
                "token_budget": 1_000_000_000,
                "final_val_bpb": 3.95,
                "min_val_bpb": 3.90,
                "selected_tok_per_sec": 560000.0,
            },
            {
                "config": "4x3",
                "token_budget": 2_000_000_000,
                "final_val_bpb": 3.82,
                "min_val_bpb": 3.80,
                "selected_tok_per_sec": 572000.0,
            },
        ]
    }
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_stage2_long_runs.py",
            "--input-json",
            str(input_json),
        ],
    )

    with pytest.raises(RuntimeError, match="must include exactly one baseline row"):
        compare_stage2_long_runs.main()


def test_main_preflight_writes_no_reports(tmp_path, monkeypatch, capsys):
    input_json = tmp_path / "stage2_runs.json"
    _write_runs(input_json)
    output_json = tmp_path / "compare.json"
    output_md = tmp_path / "compare.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_stage2_long_runs.py",
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--preflight",
        ],
    )

    compare_stage2_long_runs.main()

    assert not output_json.exists()
    assert not output_md.exists()
    stdout = capsys.readouterr().out
    assert "stage2_compare_preflight_ok" in stdout
