import json
import os
from pathlib import Path

import pytest

from scripts import run_blackwell_check_in as runner
from scripts import check_blackwell_evidence_bundle as checker


def _write_bundle_files(bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "flash_backend_smoke.json").write_text("{}\n", encoding="utf-8")
    for name in ("flash_backend_status.log", "blackwell_smoke_evidence.md", "blackwell_smoke_runbook.md"):
        (bundle_dir / name).write_text("fixture\n", encoding="utf-8")


def test_main_runs_strict_check_in_with_default_receipt(tmp_path, monkeypatch, capsys):
    bundle_dir = tmp_path / "blackwell"
    calls = {}

    def _fake_run_bundle_check(**kwargs):
        calls.update(kwargs)
        return "fa4"

    monkeypatch.setattr(runner, "run_bundle_check", _fake_run_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_check_in.py",
            "--bundle-dir",
            str(bundle_dir),
            "--expect-backend",
            "fa4",
        ],
    )

    runner.main()

    assert calls["bundle_dir"] == bundle_dir
    assert calls["expect_backend"] == "fa4"
    assert calls["check_in"] is True
    assert calls["require_blackwell"] is False
    assert calls["require_git_tracked"] is False
    assert calls["require_real_bundle"] is True
    assert calls["require_device_substring"] == "RTX 5090"
    assert calls["output_check_json"] == str(bundle_dir / "blackwell_bundle_check.json")

    stdout = capsys.readouterr().out
    assert "blackwell_check_in_ok selected=fa4" in stdout


def test_main_honors_custom_receipt_path(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    receipt_path = tmp_path / "receipts" / "check.json"
    calls = {}

    def _fake_run_bundle_check(**kwargs):
        calls.update(kwargs)
        return "fa4"

    monkeypatch.setattr(runner, "run_bundle_check", _fake_run_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_check_in.py",
            "--bundle-dir",
            str(bundle_dir),
            "--expect-backend",
            "fa4",
            "--output-check-json",
            str(receipt_path),
        ],
    )

    runner.main()

    assert calls["output_check_json"] == str(receipt_path)
    assert calls["require_real_bundle"] is True
    assert calls["require_device_substring"] == "RTX 5090"


def test_main_allows_sample_bundle_when_requested(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "artifacts" / "blackwell" / "sample_bundle"
    calls = {}

    def _fake_run_bundle_check(**kwargs):
        calls.update(kwargs)
        return "fa4"

    monkeypatch.setattr(runner, "run_bundle_check", _fake_run_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_check_in.py",
            "--bundle-dir",
            str(bundle_dir),
            "--allow-sample-bundle",
        ],
    )

    runner.main()

    assert calls["require_real_bundle"] is False
    assert calls["require_device_substring"] == "RTX 5090"


def test_main_dry_run_prints_paths_and_skips_checker(tmp_path, monkeypatch, capsys):
    bundle_dir = tmp_path / "blackwell"

    def _fake_run_bundle_check(**kwargs):
        raise AssertionError("checker should not run in dry-run mode")

    monkeypatch.setattr(runner, "run_bundle_check", _fake_run_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_check_in.py",
            "--bundle-dir",
            str(bundle_dir),
            "--expect-backend",
            "fa4",
            "--dry-run",
        ],
    )

    runner.main()

    stdout = capsys.readouterr().out
    assert "blackwell_check_in_dry_run_ok" in stdout
    assert f"bundle_dir={bundle_dir}" in stdout
    assert f"check_json={bundle_dir / 'blackwell_bundle_check.json'}" in stdout
    assert "require_device_substring=RTX 5090" in stdout


def test_sample_bundle_receipt_stays_in_sync(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)

    sample_bundle_dir = Path("artifacts/blackwell/sample_bundle")
    expected_receipt = repo_root / sample_bundle_dir / "blackwell_bundle_check.json"
    generated_receipt = tmp_path / "generated_check.json"

    def _fake_git_ls_files(cmd, capture_output, text, check):
        return checker.subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", _fake_git_ls_files)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_check_in.py",
            "--bundle-dir",
            str(sample_bundle_dir),
            "--expect-backend",
            "fa4",
            "--output-check-json",
            str(generated_receipt),
            "--allow-sample-bundle",
        ],
    )

    runner.main()

    assert json.loads(generated_receipt.read_text(encoding="utf-8")) == json.loads(
        expected_receipt.read_text(encoding="utf-8")
    )


def test_main_auto_selects_latest_real_bundle(tmp_path, monkeypatch):
    bundle_root = tmp_path / "artifacts" / "blackwell"
    older_bundle = bundle_root / "run_older"
    latest_bundle = bundle_root / "run_latest"
    sample_bundle = bundle_root / "sample_bundle"
    _write_bundle_files(older_bundle)
    _write_bundle_files(latest_bundle)
    _write_bundle_files(sample_bundle)

    os.utime(older_bundle / "flash_backend_smoke.json", (100, 100))
    os.utime(latest_bundle / "flash_backend_smoke.json", (200, 200))
    os.utime(sample_bundle / "flash_backend_smoke.json", (300, 300))

    calls = {}

    def _fake_run_bundle_check(**kwargs):
        calls.update(kwargs)
        return "fa4"

    monkeypatch.setattr(runner, "run_bundle_check", _fake_run_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_check_in.py",
            "--bundle-dir",
            "auto",
            "--bundle-root",
            str(bundle_root),
        ],
    )

    runner.main()

    assert calls["bundle_dir"] == latest_bundle
    assert calls["output_check_json"] == str(latest_bundle / "blackwell_bundle_check.json")


def test_main_auto_rejects_when_only_sample_bundle_exists(tmp_path, monkeypatch):
    bundle_root = tmp_path / "artifacts" / "blackwell"
    sample_bundle = bundle_root / "sample_bundle"
    _write_bundle_files(sample_bundle)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_check_in.py",
            "--bundle-dir",
            "auto",
            "--bundle-root",
            str(bundle_root),
        ],
    )

    with pytest.raises(RuntimeError, match="no real Blackwell bundle found"):
        runner.main()


def test_main_auto_rejects_payload_marked_sample_bundle(tmp_path, monkeypatch):
    bundle_root = tmp_path / "artifacts" / "blackwell"
    relabeled_sample_bundle = bundle_root / "run_relabeled"
    _write_bundle_files(relabeled_sample_bundle)
    (relabeled_sample_bundle / "flash_backend_smoke.json").write_text(
        json.dumps({"is_sample": True}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_check_in.py",
            "--bundle-dir",
            "auto",
            "--bundle-root",
            str(bundle_root),
        ],
    )

    with pytest.raises(RuntimeError, match="payload marked is_sample=true"):
        runner.main()
