import json
import os
import subprocess
import hashlib

import pytest

from scripts import check_pilot_sweep_artifacts as checker
from scripts import pilot_promote


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifacts(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)
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


def _write_bundle_receipt(
    *,
    path,
    ranked_json,
    finalists_json,
    finalists_md,
    finalists_count,
    source_sha256,
    run_check_in=False,
    check_json=None,
):
    payload = {
        "status": "ok",
        "command": ["run_stage2_promotion_bundle.py"],
        "input_json": str(ranked_json),
        "source_sha256": source_sha256,
        "finalists_json": str(finalists_json),
        "finalists_md": str(finalists_md),
        "finalists_count": finalists_count,
        "run_check_in": run_check_in,
        "check_json": str(check_json) if check_json else None,
        "artifact_sha256": {
            "finalists_json": _sha256(finalists_json),
            "finalists_md": _sha256(finalists_md),
        },
    }
    if check_json:
        payload["artifact_sha256"]["check_json"] = _sha256(check_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def test_main_accepts_valid_bundle_receipt(tmp_path, monkeypatch, capsys):
    ranked_json, finalists_json, finalists_md = _write_artifacts(tmp_path)
    ranked_payload = json.loads(ranked_json.read_text(encoding="utf-8"))
    bundle_json = tmp_path / "stage2_bundle_receipt.json"
    _write_bundle_receipt(
        path=bundle_json,
        ranked_json=ranked_json,
        finalists_json=finalists_json,
        finalists_md=finalists_md,
        finalists_count=2,
        source_sha256=pilot_promote._stable_json_sha256(ranked_payload),
    )

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
            "--bundle-json",
            str(bundle_json),
        ],
    )

    checker.main()
    stdout = capsys.readouterr().out
    assert "pilot_bundle_check_ok finalists=2" in stdout
    assert f"bundle_json={bundle_json}" in stdout


def test_main_rejects_bundle_receipt_sha_mismatch(tmp_path, monkeypatch):
    ranked_json, finalists_json, finalists_md = _write_artifacts(tmp_path)
    ranked_payload = json.loads(ranked_json.read_text(encoding="utf-8"))
    bundle_json = tmp_path / "stage2_bundle_receipt.json"
    _write_bundle_receipt(
        path=bundle_json,
        ranked_json=ranked_json,
        finalists_json=finalists_json,
        finalists_md=finalists_md,
        finalists_count=2,
        source_sha256=pilot_promote._stable_json_sha256(ranked_payload),
    )

    payload = json.loads(bundle_json.read_text(encoding="utf-8"))
    payload["artifact_sha256"]["finalists_json"] = "0" * 64
    bundle_json.write_text(json.dumps(payload), encoding="utf-8")

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
            "--bundle-json",
            str(bundle_json),
        ],
    )

    with pytest.raises(RuntimeError, match="bundle receipt artifact_sha256.finalists_json does not match"):
        checker.main()


def test_main_dry_run_prints_resolved_paths_and_skips_validation(tmp_path, monkeypatch, capsys):
    ranked_json, finalists_json, finalists_md = _write_artifacts(tmp_path)

    def _fail_run_pilot_bundle_check(**kwargs):
        raise AssertionError("validation should not run in dry-run mode")

    monkeypatch.setattr(checker, "run_pilot_bundle_check", _fail_run_pilot_bundle_check)
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
            "--check-in",
            "--dry-run",
        ],
    )

    checker.main()
    stdout = capsys.readouterr().out
    assert "pilot_bundle_check_dry_run_ok" in stdout
    assert f"ranked_json={ranked_json}" in stdout
    assert f"finalists_json={finalists_json}" in stdout
    assert f"finalists_md={finalists_md}" in stdout
    assert "check_in=True" in stdout


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
        "artifact_sha256": {
            "ranked_json": _sha256(ranked_json),
            "finalists_json": _sha256(finalists_json),
            "finalists_md": _sha256(finalists_md),
        },
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


def test_main_auto_discovers_latest_real_artifacts_dir(tmp_path, monkeypatch, capsys):
    artifacts_root = tmp_path / "artifacts" / "pilot"
    older_dir = artifacts_root / "run_older"
    latest_dir = artifacts_root / "run_latest"
    sample_dir = artifacts_root / "sample_run"
    _write_artifacts(older_dir)
    _write_artifacts(latest_dir)
    _write_artifacts(sample_dir)

    os.utime(older_dir / "pilot_ranked_runs.json", (100, 100))
    os.utime(latest_dir / "pilot_ranked_runs.json", (200, 200))
    os.utime(sample_dir / "pilot_ranked_runs.json", (300, 300))

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pilot_sweep_artifacts.py",
            "--artifacts-dir",
            "auto",
            "--artifacts-root",
            str(artifacts_root),
        ],
    )

    checker.main()

    stdout = capsys.readouterr().out
    assert "pilot_bundle_check_ok finalists=2" in stdout
    assert f"ranked_json={latest_dir / 'pilot_ranked_runs.json'}" in stdout


def test_main_auto_discovery_lists_rejected_candidates(tmp_path, monkeypatch):
    artifacts_root = tmp_path / "artifacts" / "pilot"
    sample_dir = artifacts_root / "sample_run"
    incomplete_dir = artifacts_root / "run_incomplete"
    _write_artifacts(sample_dir)
    incomplete_dir.mkdir(parents=True, exist_ok=True)
    (incomplete_dir / "pilot_ranked_runs.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pilot_sweep_artifacts.py",
            "--artifacts-dir",
            "auto",
            "--artifacts-root",
            str(artifacts_root),
        ],
    )

    with pytest.raises(RuntimeError) as exc_info:
        checker.main()

    message = str(exc_info.value)
    assert "no real pilot artifact bundle found" in message
    assert "rejected 2 candidate bundle(s)" in message
    assert f"{sample_dir}: sample path segment" in message
    assert (
        f"{incomplete_dir}: missing files: stage2_finalists.json, stage2_finalists.md"
        in message
    )


def test_main_requires_explicit_paths_without_artifacts_dir(tmp_path, monkeypatch):
    ranked_json, finalists_json, _finalists_md = _write_artifacts(tmp_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_pilot_sweep_artifacts.py",
            "--ranked-json",
            str(ranked_json),
            "--finalists-json",
            str(finalists_json),
        ],
    )

    with pytest.raises(RuntimeError, match="missing required artifact paths"):
        checker.main()
