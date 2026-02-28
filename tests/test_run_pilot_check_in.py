import os
import json

import pytest

from scripts import run_pilot_check_in as runner


def _write_artifact_files(artifacts_dir):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "pilot_ranked_runs.json").write_text(
        json.dumps({"ranked_runs": []}) + "\n",
        encoding="utf-8",
    )
    (artifacts_dir / "stage2_finalists.json").write_text("{}\n", encoding="utf-8")
    (artifacts_dir / "stage2_finalists.md").write_text("# fixture\n", encoding="utf-8")


def test_main_runs_strict_check_in_with_default_receipt(tmp_path, monkeypatch, capsys):
    artifacts_dir = tmp_path / "pilot"
    calls = {}

    def _fake_run_pilot_bundle_check(**kwargs):
        calls.update(kwargs)
        return 2

    monkeypatch.setattr(runner, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            str(artifacts_dir),
        ],
    )

    runner.main()

    assert calls["ranked_json_path"] == artifacts_dir / "pilot_ranked_runs.json"
    assert calls["finalists_json_path"] == artifacts_dir / "stage2_finalists.json"
    assert calls["finalists_md_path"] == artifacts_dir / "stage2_finalists.md"
    assert calls["require_real_input"] is True
    assert calls["require_git_tracked"] is False
    assert calls["check_in"] is True
    assert calls["allow_sample_input_in_check_in"] is False
    assert calls["output_check_json"] == str(artifacts_dir / "pilot_bundle_check.json")

    stdout = capsys.readouterr().out
    assert "pilot_check_in_ok finalists=2" in stdout


def test_main_honors_custom_paths(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "pilot"
    receipt_path = tmp_path / "receipts" / "check.json"
    calls = {}

    def _fake_run_pilot_bundle_check(**kwargs):
        calls.update(kwargs)
        return 3

    monkeypatch.setattr(runner, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            str(artifacts_dir),
            "--ranked-json",
            "real_ranked.json",
            "--finalists-json",
            "real_finalists.json",
            "--finalists-md",
            "real_finalists.md",
            "--output-check-json",
            str(receipt_path),
            "--bundle-json",
            "stage2_promotion_bundle.json",
        ],
    )

    runner.main()

    assert calls["ranked_json_path"] == artifacts_dir / "real_ranked.json"
    assert calls["finalists_json_path"] == artifacts_dir / "real_finalists.json"
    assert calls["finalists_md_path"] == artifacts_dir / "real_finalists.md"
    assert calls["require_real_input"] is True
    assert calls["allow_sample_input_in_check_in"] is False
    assert calls["output_check_json"] == str(receipt_path)
    assert calls["bundle_json_path"] == artifacts_dir / "stage2_promotion_bundle.json"


def test_main_allow_sample_input_disables_real_input_guard(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "pilot"
    calls = {}

    def _fake_run_pilot_bundle_check(**kwargs):
        calls.update(kwargs)
        return 1

    monkeypatch.setattr(runner, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            str(artifacts_dir),
            "--allow-sample-input",
        ],
    )

    runner.main()

    assert calls["require_real_input"] is False
    assert calls["allow_sample_input_in_check_in"] is True


def test_main_dry_run_prints_paths_and_skips_checker(tmp_path, monkeypatch, capsys):
    artifacts_dir = tmp_path / "pilot"

    def _fake_run_pilot_bundle_check(**kwargs):
        raise AssertionError("checker should not run in dry-run mode")

    monkeypatch.setattr(runner, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            str(artifacts_dir),
            "--dry-run",
        ],
    )

    runner.main()

    stdout = capsys.readouterr().out
    assert "pilot_check_in_dry_run_ok" in stdout
    assert f"artifacts_dir={artifacts_dir}" in stdout
    assert f"check_json={artifacts_dir / 'pilot_bundle_check.json'}" in stdout


def test_main_dry_run_resolves_bundle_json_auto(tmp_path, monkeypatch, capsys):
    artifacts_dir = tmp_path / "pilot"

    def _fake_run_pilot_bundle_check(**kwargs):
        raise AssertionError("checker should not run in dry-run mode")

    monkeypatch.setattr(runner, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            str(artifacts_dir),
            "--bundle-json",
            "auto",
            "--dry-run",
        ],
    )

    runner.main()

    stdout = capsys.readouterr().out
    assert f"bundle_json={artifacts_dir / 'stage2_promotion_bundle.json'}" in stdout


def test_main_dry_run_prints_check_md_when_set(tmp_path, monkeypatch, capsys):
    artifacts_dir = tmp_path / "pilot"
    check_md = tmp_path / "receipts" / "pilot_check_in.md"

    def _fake_run_pilot_bundle_check(**kwargs):
        raise AssertionError("checker should not run in dry-run mode")

    monkeypatch.setattr(runner, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            str(artifacts_dir),
            "--output-check-md",
            str(check_md),
            "--dry-run",
        ],
    )

    runner.main()

    stdout = capsys.readouterr().out
    assert f"check_md={check_md}" in stdout


def test_main_writes_check_markdown_summary(tmp_path, monkeypatch, capsys):
    artifacts_dir = tmp_path / "pilot"
    check_md = tmp_path / "receipts" / "pilot_check_in.md"
    bundle_json = tmp_path / "receipts" / "promotion_bundle.json"

    def _fake_run_pilot_bundle_check(**kwargs):
        return 2

    monkeypatch.setattr(runner, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            str(artifacts_dir),
            "--output-check-md",
            str(check_md),
            "--bundle-json",
            str(bundle_json),
        ],
    )

    runner.main()

    stdout = capsys.readouterr().out
    assert f"check_md={check_md}" in stdout
    markdown = check_md.read_text(encoding="utf-8")
    assert "# Pilot Artifact Strict Check-In Evidence" in markdown
    assert "- finalists: `2`" in markdown
    assert f"- artifacts_dir: `{artifacts_dir}`" in markdown
    assert f"- ranked_json: `{artifacts_dir / 'pilot_ranked_runs.json'}`" in markdown
    assert f"- finalists_json: `{artifacts_dir / 'stage2_finalists.json'}`" in markdown
    assert f"- finalists_md: `{artifacts_dir / 'stage2_finalists.md'}`" in markdown
    assert f"- check_json: `{artifacts_dir / 'pilot_bundle_check.json'}`" in markdown
    assert "- require_real_input: `true`" in markdown
    assert "- check_in_mode: `true`" in markdown
    assert f"- bundle_json: `{bundle_json}`" in markdown


def test_main_auto_selects_latest_real_artifacts_dir(tmp_path, monkeypatch):
    artifacts_root = tmp_path / "artifacts" / "pilot"
    older_artifacts = artifacts_root / "run_older"
    latest_artifacts = artifacts_root / "run_latest"
    sample_artifacts = artifacts_root / "sample_run"
    _write_artifact_files(older_artifacts)
    _write_artifact_files(latest_artifacts)
    _write_artifact_files(sample_artifacts)

    os.utime(older_artifacts / "pilot_ranked_runs.json", (100, 100))
    os.utime(latest_artifacts / "pilot_ranked_runs.json", (200, 200))
    os.utime(sample_artifacts / "pilot_ranked_runs.json", (300, 300))

    calls = {}

    def _fake_run_pilot_bundle_check(**kwargs):
        calls.update(kwargs)
        return 2

    monkeypatch.setattr(runner, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            "auto",
            "--artifacts-root",
            str(artifacts_root),
        ],
    )

    runner.main()

    assert calls["ranked_json_path"] == latest_artifacts / "pilot_ranked_runs.json"
    assert calls["output_check_json"] == str(latest_artifacts / "pilot_bundle_check.json")


def test_main_auto_rejects_when_no_real_artifacts_dir_exists(tmp_path, monkeypatch):
    artifacts_root = tmp_path / "artifacts" / "pilot"
    sample_artifacts = artifacts_root / "sample_run"
    _write_artifact_files(sample_artifacts)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            "auto",
            "--artifacts-root",
            str(artifacts_root),
        ],
    )

    with pytest.raises(RuntimeError, match="no real pilot artifact bundle found"):
        runner.main()


def test_main_auto_rejection_error_lists_candidate_reasons(tmp_path, monkeypatch):
    artifacts_root = tmp_path / "artifacts" / "pilot"
    sample_artifacts = artifacts_root / "sample_run"
    incomplete_artifacts = artifacts_root / "run_incomplete"
    _write_artifact_files(sample_artifacts)
    incomplete_artifacts.mkdir(parents=True, exist_ok=True)
    (incomplete_artifacts / "pilot_ranked_runs.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            "auto",
            "--artifacts-root",
            str(artifacts_root),
        ],
    )

    with pytest.raises(RuntimeError) as exc_info:
        runner.main()

    message = str(exc_info.value)
    assert "rejected 2 candidate bundle(s)" in message
    assert f"{sample_artifacts}: sample path segment" in message
    assert (
        f"{incomplete_artifacts}: missing files: stage2_finalists.json, stage2_finalists.md"
        in message
    )


def test_main_auto_rejects_payload_marked_sample_candidates(tmp_path, monkeypatch):
    artifacts_root = tmp_path / "artifacts" / "pilot"
    payload_sample = artifacts_root / "run_payload_flagged"
    _write_artifact_files(payload_sample)

    ranked_payload = json.loads((payload_sample / "pilot_ranked_runs.json").read_text(encoding="utf-8"))
    ranked_payload["is_sample"] = True
    (payload_sample / "pilot_ranked_runs.json").write_text(json.dumps(ranked_payload), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            "auto",
            "--artifacts-root",
            str(artifacts_root),
        ],
    )

    with pytest.raises(RuntimeError) as exc_info:
        runner.main()

    message = str(exc_info.value)
    assert "rejected 1 candidate bundle(s)" in message
    assert f"{payload_sample}: ranked JSON payload marks is_sample=true" in message


def test_main_auto_rejects_malformed_ranked_json_candidates(tmp_path, monkeypatch):
    artifacts_root = tmp_path / "artifacts" / "pilot"
    malformed = artifacts_root / "run_malformed"
    _write_artifact_files(malformed)
    (malformed / "pilot_ranked_runs.json").write_text("{\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            "auto",
            "--artifacts-root",
            str(artifacts_root),
        ],
    )

    with pytest.raises(RuntimeError) as exc_info:
        runner.main()

    message = str(exc_info.value)
    assert "rejected 1 candidate bundle(s)" in message
    assert f"{malformed}: malformed ranked JSON" in message
