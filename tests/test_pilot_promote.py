import json

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

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["max_finalists"] == 1
    assert [row["config"] for row in payload["selected_finalists"]] == ["12x1"]

    md = output_md.read_text(encoding="utf-8")
    assert "## Stage 2 Finalists" in md
    assert "`--depth 12 --n-branches 1 --aspect-ratio 64`" in md
