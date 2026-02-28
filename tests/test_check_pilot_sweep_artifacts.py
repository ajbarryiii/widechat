import json
import subprocess

import pytest

from scripts import check_pilot_sweep_artifacts as checker
from scripts import pilot_promote


def _write_artifacts(base_dir):
    ranked_json = base_dir / "pilot_ranked_runs.json"
    finalists_json = base_dir / "stage2_finalists.json"
    finalists_md = base_dir / "stage2_finalists.md"

    ranked_runs = [
        {
            "rank": 1,
            "config": "4x3",
            "depth": 4,
            "n_branches": 3,
            "aspect_ratio": 192,
            "selected_tok_per_sec": 1000,
            "min_val_bpb": 4.0,
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
            "min_val_bpb": 4.01,
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
    ranked_payload = {"ranked_runs": ranked_runs}
    ranked_json.write_text(json.dumps(ranked_payload), encoding="utf-8")
    ranked_source_sha256 = pilot_promote._stable_json_sha256(ranked_payload)

    selected_finalists = ranked_runs[:2]
    finalists_json.write_text(
        json.dumps(
            {
                "source": str(ranked_json),
                "source_sha256": ranked_source_sha256,
                "max_finalists": 2,
                "selected_finalists": selected_finalists,
            }
        ),
        encoding="utf-8",
    )
    finalists_md.write_text(
        "\n".join(
            [
                "## Stage 2 Finalists",
                "",
                "Selected finalists:",
                "",
                "## Stage 2 depth/branch flags",
                "",
                "- `4x3`: `--depth 4 --n-branches 3 --aspect-ratio 192`",
                "- `6x2`: `--depth 6 --n-branches 2 --aspect-ratio 128`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return ranked_json, finalists_json, finalists_md


def test_main_accepts_valid_artifacts(tmp_path, monkeypatch, capsys):
    ranked_json, finalists_json, finalists_md = _write_artifacts(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pilot_sweep_artifacts.py",
            "--ranked-json",
            str(ranked_json),
            "--finalists-json",
            str(finalists_json),
            "--finalists-md",
            str(finalists_md),
        ],
    )

    checker.main()
    assert "pilot_bundle_check_ok finalists=2" in capsys.readouterr().out


def test_main_writes_machine_readable_receipt(tmp_path, monkeypatch, capsys):
    ranked_json, finalists_json, finalists_md = _write_artifacts(tmp_path)
    receipt_json = tmp_path / "pilot_bundle_check.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pilot_sweep_artifacts.py",
            "--ranked-json",
            str(ranked_json),
            "--finalists-json",
            str(finalists_json),
            "--finalists-md",
            str(finalists_md),
            "--output-check-json",
            str(receipt_json),
        ],
    )

    checker.main()
    stdout = capsys.readouterr().out
    assert f"check_json={receipt_json}" in stdout

    receipt = json.loads(receipt_json.read_text(encoding="utf-8"))
    assert receipt == {
        "status": "ok",
        "command": [
            "check_pilot_sweep_artifacts.py",
            "--ranked-json",
            str(ranked_json),
            "--finalists-json",
            str(finalists_json),
            "--finalists-md",
            str(finalists_md),
            "--output-check-json",
            str(receipt_json),
        ],
        "ranked_json": str(ranked_json),
        "finalists_json": str(finalists_json),
        "finalists_md": str(finalists_md),
        "finalists_count": 2,
        "require_real_input": False,
        "require_git_tracked": False,
        "check_in": False,
    }


def test_main_rejects_missing_source_sha256(tmp_path, monkeypatch):
    ranked_json, finalists_json, finalists_md = _write_artifacts(tmp_path)
    payload = json.loads(finalists_json.read_text(encoding="utf-8"))
    payload.pop("source_sha256")
    finalists_json.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pilot_sweep_artifacts.py",
            "--ranked-json",
            str(ranked_json),
            "--finalists-json",
            str(finalists_json),
            "--finalists-md",
            str(finalists_md),
        ],
    )

    with pytest.raises(RuntimeError, match="finalists JSON missing source_sha256 digest"):
        checker.main()


def test_main_rejects_source_sha256_mismatch(tmp_path, monkeypatch):
    ranked_json, finalists_json, finalists_md = _write_artifacts(tmp_path)
    payload = json.loads(finalists_json.read_text(encoding="utf-8"))
    payload["source_sha256"] = "0" * 64
    finalists_json.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pilot_sweep_artifacts.py",
            "--ranked-json",
            str(ranked_json),
            "--finalists-json",
            str(finalists_json),
            "--finalists-md",
            str(finalists_md),
        ],
    )

    with pytest.raises(RuntimeError, match="source_sha256 does not match --ranked-json contents"):
        checker.main()


def test_main_rejects_finalists_mismatch(tmp_path, monkeypatch):
    ranked_json, finalists_json, finalists_md = _write_artifacts(tmp_path)
    payload = json.loads(finalists_json.read_text(encoding="utf-8"))
    payload["selected_finalists"] = payload["selected_finalists"][::-1]
    finalists_json.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pilot_sweep_artifacts.py",
            "--ranked-json",
            str(ranked_json),
            "--finalists-json",
            str(finalists_json),
            "--finalists-md",
            str(finalists_md),
        ],
    )

    with pytest.raises(RuntimeError, match="selected_finalists does not match"):
        checker.main()


def test_main_rejects_finalists_source_mismatch(tmp_path, monkeypatch):
    ranked_json, finalists_json, finalists_md = _write_artifacts(tmp_path)
    payload = json.loads(finalists_json.read_text(encoding="utf-8"))
    payload["source"] = str(tmp_path / "other_ranked_runs.json")
    finalists_json.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pilot_sweep_artifacts.py",
            "--ranked-json",
            str(ranked_json),
            "--finalists-json",
            str(finalists_json),
            "--finalists-md",
            str(finalists_md),
        ],
    )

    with pytest.raises(RuntimeError, match="finalists JSON source does not match --ranked-json"):
        checker.main()


def test_main_rejects_markdown_missing_flag_line(tmp_path, monkeypatch):
    ranked_json, finalists_json, finalists_md = _write_artifacts(tmp_path)
    finalists_md.write_text("## Stage 2 Finalists\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pilot_sweep_artifacts.py",
            "--ranked-json",
            str(ranked_json),
            "--finalists-json",
            str(finalists_json),
            "--finalists-md",
            str(finalists_md),
        ],
    )

    with pytest.raises(RuntimeError, match="finalists markdown missing snippet"):
        checker.main()


def test_main_check_in_mode_enforces_real_input(tmp_path, monkeypatch):
    ranked_json, finalists_json, finalists_md = _write_artifacts(tmp_path)
    sample_ranked = tmp_path / "sample_ranked_runs.json"
    sample_ranked.write_text(ranked_json.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pilot_sweep_artifacts.py",
            "--ranked-json",
            str(sample_ranked),
            "--finalists-json",
            str(finalists_json),
            "--finalists-md",
            str(finalists_md),
            "--check-in",
        ],
    )

    with pytest.raises(ValueError, match="--require-real-input rejects sample/fixture"):
        checker.main()


def test_main_check_in_mode_rejects_sample_payload_flag(tmp_path, monkeypatch):
    ranked_json, finalists_json, finalists_md = _write_artifacts(tmp_path)
    payload = json.loads(ranked_json.read_text(encoding="utf-8"))
    payload["is_sample"] = True

    relabeled_ranked = tmp_path / "pilot_ranked_runs.json"
    relabeled_ranked.write_text(json.dumps(payload), encoding="utf-8")

    finalists_payload = json.loads(finalists_json.read_text(encoding="utf-8"))
    finalists_payload["source"] = str(relabeled_ranked)
    finalists_json.write_text(json.dumps(finalists_payload), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pilot_sweep_artifacts.py",
            "--ranked-json",
            str(relabeled_ranked),
            "--finalists-json",
            str(finalists_json),
            "--finalists-md",
            str(finalists_md),
            "--check-in",
        ],
    )

    with pytest.raises(ValueError, match="--require-real-input rejects sample/fixture"):
        checker.main()


def test_run_bundle_check_allows_sample_input_override_in_check_in_mode(tmp_path, monkeypatch):
    ranked_json, finalists_json, finalists_md = _write_artifacts(tmp_path)
    payload = json.loads(ranked_json.read_text(encoding="utf-8"))
    payload["is_sample"] = True
    ranked_json.write_text(json.dumps(payload), encoding="utf-8")

    finalists_payload = json.loads(finalists_json.read_text(encoding="utf-8"))
    finalists_payload["source"] = str(ranked_json)
    finalists_payload["source_sha256"] = pilot_promote._stable_json_sha256(payload)
    finalists_json.write_text(json.dumps(finalists_payload), encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    def _fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", _fake_run)

    finalists_count = checker.run_pilot_bundle_check(
        ranked_json_path=ranked_json,
        finalists_json_path=finalists_json,
        finalists_md_path=finalists_md,
        require_real_input=False,
        require_git_tracked=False,
        check_in=True,
        allow_sample_input_in_check_in=True,
    )

    assert finalists_count == 2


def test_main_require_git_tracked_rejects_untracked(tmp_path, monkeypatch):
    ranked_json, finalists_json, finalists_md = _write_artifacts(tmp_path)
    monkeypatch.chdir(tmp_path)

    def _fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fatal: pathspec did not match")

    monkeypatch.setattr(checker.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pilot_sweep_artifacts.py",
            "--ranked-json",
            str(ranked_json),
            "--finalists-json",
            str(finalists_json),
            "--finalists-md",
            str(finalists_md),
            "--require-git-tracked",
        ],
    )

    with pytest.raises(RuntimeError, match="artifact is not git-tracked"):
        checker.main()
