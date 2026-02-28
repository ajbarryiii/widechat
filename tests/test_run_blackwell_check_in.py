import json
from pathlib import Path

from scripts import run_blackwell_check_in as runner
from scripts import check_blackwell_evidence_bundle as checker


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
        ],
    )

    runner.main()

    assert json.loads(generated_receipt.read_text(encoding="utf-8")) == json.loads(
        expected_receipt.read_text(encoding="utf-8")
    )
