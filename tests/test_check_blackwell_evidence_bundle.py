import json
import subprocess

import pytest

from scripts import check_blackwell_evidence_bundle as checker


def _write_valid_bundle(bundle_dir, *, cuda_capability=(10, 0)):
    bundle_dir.mkdir(parents=True, exist_ok=True)
    status_line = "Flash Attention backend selection: selected=fa4, mode=auto"
    payload = {
        "selected_backend": "fa4",
        "status_line": status_line,
        "cuda_available": True,
        "device_name": "RTX 5090",
        "cuda_capability": [cuda_capability[0], cuda_capability[1]],
    }
    (bundle_dir / "flash_backend_smoke.json").write_text(json.dumps(payload), encoding="utf-8")
    (bundle_dir / "flash_backend_status.log").write_text(status_line + "\n", encoding="utf-8")
    (bundle_dir / "blackwell_smoke_evidence.md").write_text(
        "\n".join(
            [
                "# Blackwell Flash Backend Smoke Evidence",
                "",
                "- selected_backend: `fa4`",
                "- cuda_available: `true`",
                "- cuda_capability: `sm100`",
                "- device_name: `RTX 5090`",
                "- status_line_ok: `true`",
                "- status_line: `Flash Attention backend selection: selected=fa4, mode=auto`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (bundle_dir / "blackwell_smoke_runbook.md").write_text(
        "\n".join(
            [
                "# Blackwell Smoke Bundle Runbook",
                "",
                "## Command",
                "```bash",
                "python -m scripts.run_blackwell_smoke_bundle \\",
                f"  --output-dir {bundle_dir} \\",
                "  --expect-backend fa4",
                "```",
                "",
                "## Expected outputs",
                f"- `{bundle_dir}/flash_backend_smoke.json`",
                f"- `{bundle_dir}/flash_backend_status.log`",
                f"- `{bundle_dir}/blackwell_smoke_evidence.md`",
                "",
                "## Check-in checklist",
                "- Ensure command prints `bundle_ok selected=fa4`.",
                "- Run `python -m scripts.check_blackwell_evidence_bundle --bundle-dir",
                f" {bundle_dir} --expect-backend fa4 --check-in`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_main_accepts_valid_bundle(tmp_path, monkeypatch, capsys):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--expect-backend",
            "fa4",
            "--require-blackwell",
        ],
    )

    checker.main()
    stdout = capsys.readouterr().out
    assert "bundle_check_ok selected=fa4" in stdout


def test_main_rejects_missing_required_file(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)
    (bundle_dir / "blackwell_smoke_runbook.md").unlink()

    monkeypatch.setattr("sys.argv", ["check_blackwell_evidence_bundle.py", "--bundle-dir", str(bundle_dir)])

    with pytest.raises(RuntimeError, match="missing runbook_md file"):
        checker.main()


def test_main_rejects_evidence_markdown_drift(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)
    (bundle_dir / "blackwell_smoke_evidence.md").write_text("# Blackwell Flash Backend Smoke Evidence\n", encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["check_blackwell_evidence_bundle.py", "--bundle-dir", str(bundle_dir)])

    with pytest.raises(RuntimeError, match="evidence markdown missing line"):
        checker.main()


def test_main_require_git_tracked_accepts_tracked_bundle(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)

    def _fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--require-git-tracked",
        ],
    )

    checker.main()


def test_main_require_git_tracked_rejects_untracked_bundle(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)

    def _fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fatal: pathspec did not match")

    monkeypatch.setattr(checker.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--require-git-tracked",
        ],
    )

    with pytest.raises(RuntimeError, match="bundle artifact is not git-tracked"):
        checker.main()


def test_main_check_in_mode_enforces_requirements(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)

    def _fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--check-in",
        ],
    )

    checker.main()


def test_main_check_in_mode_rejects_non_blackwell_bundle(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir, cuda_capability=(9, 0))

    def _fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--check-in",
        ],
    )

    with pytest.raises(RuntimeError, match="artifact is not from Blackwell"):
        checker.main()
