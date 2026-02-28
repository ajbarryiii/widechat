import json
import os
import shlex
import subprocess
import hashlib

import pytest

from scripts import check_blackwell_evidence_bundle as checker


def _write_valid_bundle(bundle_dir, *, cuda_capability=(10, 0), is_sample=False):
    bundle_dir.mkdir(parents=True, exist_ok=True)
    status_line = "Flash Attention backend selection: selected=fa4, mode=auto"
    payload = {
        "selected_backend": "fa4",
        "status_line": status_line,
        "is_sample": is_sample,
        "cuda_available": True,
        "device_name": "RTX 5090",
        "cuda_capability": [cuda_capability[0], cuda_capability[1]],
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
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
                "- generated_at_utc: `2026-02-27T00:00:00Z`",
                "- git_commit: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`",
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
                "  --expect-backend fa4 \\",
                "  --require-device-substring 'RTX 5090'",
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
                f" {bundle_dir} --expect-backend fa4 --check-in --output-check-json",
                f" {bundle_dir}/blackwell_bundle_check.json --require-device-substring 'RTX 5090'`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_main_rejects_evidence_generated_at_mismatch(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)
    (bundle_dir / "blackwell_smoke_evidence.md").write_text(
        (bundle_dir / "blackwell_smoke_evidence.md")
        .read_text(encoding="utf-8")
        .replace("2026-02-27T00:00:00Z", "2026-02-27T01:00:00Z"),
        encoding="utf-8",
    )

    monkeypatch.setattr("sys.argv", ["check_blackwell_evidence_bundle.py", "--bundle-dir", str(bundle_dir)])

    with pytest.raises(RuntimeError, match="evidence markdown generated_at_utc mismatch"):
        checker.main()


def test_main_rejects_evidence_git_commit_mismatch(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)
    (bundle_dir / "blackwell_smoke_evidence.md").write_text(
        (bundle_dir / "blackwell_smoke_evidence.md")
        .read_text(encoding="utf-8")
        .replace(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            1,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("sys.argv", ["check_blackwell_evidence_bundle.py", "--bundle-dir", str(bundle_dir)])

    with pytest.raises(RuntimeError, match="evidence markdown git_commit mismatch"):
        checker.main()


def test_main_rejects_evidence_status_line_not_true(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)
    (bundle_dir / "blackwell_smoke_evidence.md").write_text(
        (bundle_dir / "blackwell_smoke_evidence.md")
        .read_text(encoding="utf-8")
        .replace("- status_line_ok: `true`", "- status_line_ok: `false`"),
        encoding="utf-8",
    )

    monkeypatch.setattr("sys.argv", ["check_blackwell_evidence_bundle.py", "--bundle-dir", str(bundle_dir)])

    with pytest.raises(RuntimeError, match="evidence markdown status_line_ok mismatch"):
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


def test_main_accepts_runbook_with_check_in_helper(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)
    (bundle_dir / "blackwell_smoke_runbook.md").write_text(
        "\n".join(
            [
                "# Blackwell Smoke Bundle Runbook",
                "",
                "## Command",
                "```bash",
                "python -m scripts.run_blackwell_smoke_bundle \\",
                f"  --output-dir {bundle_dir} \\",
                "  --expect-backend fa4 \\",
                "  --require-device-substring 'RTX 5090'",
                "```",
                "",
                "## Expected outputs",
                f"- `{bundle_dir}/flash_backend_smoke.json`",
                f"- `{bundle_dir}/flash_backend_status.log`",
                f"- `{bundle_dir}/blackwell_smoke_evidence.md`",
                "",
                "## Check-in checklist",
                "- Ensure command prints `bundle_ok selected=fa4`.",
                "- Run `python -m scripts.run_blackwell_check_in --bundle-dir",
                f" {bundle_dir} --expect-backend fa4 --output-check-json",
                f" {bundle_dir}/blackwell_bundle_check.json --require-device-substring 'RTX 5090'`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

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


def test_main_accepts_shell_quoted_runbook_paths(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell artifacts"
    _write_valid_bundle(bundle_dir)
    quoted_bundle_dir = shlex.quote(str(bundle_dir))
    quoted_check_json = shlex.quote(str(bundle_dir / "blackwell_bundle_check.json"))
    (bundle_dir / "blackwell_smoke_runbook.md").write_text(
        "\n".join(
            [
                "# Blackwell Smoke Bundle Runbook",
                "",
                "## Command",
                "```bash",
                "python -m scripts.run_blackwell_smoke_bundle \\",
                f"  --output-dir {quoted_bundle_dir} \\",
                "  --expect-backend fa4 \\",
                "  --require-device-substring 'RTX 5090'",
                "```",
                "",
                "## Expected outputs",
                f"- `{bundle_dir}/flash_backend_smoke.json`",
                f"- `{bundle_dir}/flash_backend_status.log`",
                f"- `{bundle_dir}/blackwell_smoke_evidence.md`",
                "",
                "## Check-in checklist",
                "- Ensure command prints `bundle_ok selected=fa4`.",
                "- Run `python -m scripts.run_blackwell_check_in --bundle-dir",
                f" {quoted_bundle_dir} --expect-backend fa4 --output-check-json",
                f" {quoted_check_json} --require-device-substring 'RTX 5090'`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

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


def test_main_accepts_runbook_with_custom_check_receipt_path(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)
    custom_check_json = tmp_path / "receipts" / "blackwell check.json"
    quoted_bundle_dir = shlex.quote(str(bundle_dir))
    quoted_custom_check_json = shlex.quote(str(custom_check_json))
    (bundle_dir / "blackwell_smoke_runbook.md").write_text(
        "\n".join(
            [
                "# Blackwell Smoke Bundle Runbook",
                "",
                "## Command",
                "```bash",
                "python -m scripts.run_blackwell_smoke_bundle \\",
                f"  --output-dir {quoted_bundle_dir} \\",
                "  --expect-backend fa4 \\",
                "  --require-device-substring 'RTX 5090'",
                "```",
                "",
                "## Expected outputs",
                f"- `{bundle_dir}/flash_backend_smoke.json`",
                f"- `{bundle_dir}/flash_backend_status.log`",
                f"- `{bundle_dir}/blackwell_smoke_evidence.md`",
                "",
                "## Check-in checklist",
                "- Ensure command prints `bundle_ok selected=fa4`.",
                "- Run `python -m scripts.run_blackwell_check_in --bundle-dir",
                f" {quoted_bundle_dir} --expect-backend fa4 --output-check-json",
                f" {quoted_custom_check_json} --require-device-substring 'RTX 5090'`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

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


def test_main_check_in_mode_rejects_non_5090_device_name(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)
    payload = json.loads((bundle_dir / "flash_backend_smoke.json").read_text(encoding="utf-8"))
    payload["device_name"] = "NVIDIA H100"
    (bundle_dir / "flash_backend_smoke.json").write_text(json.dumps(payload), encoding="utf-8")

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

    with pytest.raises(RuntimeError, match="device_name does not include required substring"):
        checker.main()


def test_main_require_real_bundle_rejects_sample_fixture_path(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "artifacts" / "blackwell" / "sample_bundle"
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
            "--require-real-bundle",
        ],
    )

    with pytest.raises(RuntimeError, match="sample fixture artifacts"):
        checker.main()


def test_main_require_real_bundle_rejects_sample_payload_metadata(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "artifacts" / "blackwell" / "renamed_real_path"
    _write_valid_bundle(bundle_dir, is_sample=True)

    def _fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--require-real-bundle",
        ],
    )

    with pytest.raises(RuntimeError, match="payload is marked as sample fixture"):
        checker.main()


def test_main_writes_machine_readable_check_report(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)
    report_path = bundle_dir / "check_report.json"

    def _fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--expect-backend",
            "fa4",
            "--check-in",
            "--output-check-json",
            str(report_path),
        ],
    )

    checker.main()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["bundle_dir"] == str(bundle_dir)
    assert payload["expect_backend"] == "fa4"
    assert payload["selected_backend"] == "fa4"
    assert payload["check_in"] is True
    assert payload["require_blackwell"] is True
    assert payload["require_git_tracked"] is True
    assert payload["require_real_bundle"] is False
    assert payload["require_device_substring"] == "RTX 5090"
    assert payload["artifact_sha256"] == {
        "artifact_json": _sha256(bundle_dir / "flash_backend_smoke.json"),
        "status_line": _sha256(bundle_dir / "flash_backend_status.log"),
        "evidence_md": _sha256(bundle_dir / "blackwell_smoke_evidence.md"),
        "runbook_md": _sha256(bundle_dir / "blackwell_smoke_runbook.md"),
    }


def test_main_check_report_includes_require_real_bundle(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)
    report_path = bundle_dir / "check_report_real_bundle.json"

    def _fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--expect-backend",
            "fa4",
            "--check-in",
            "--require-real-bundle",
            "--output-check-json",
            str(report_path),
        ],
    )

    checker.main()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["require_real_bundle"] is True
    assert payload["require_device_substring"] == "RTX 5090"


def test_main_auto_selects_latest_real_bundle(tmp_path, monkeypatch, capsys):
    bundle_root = tmp_path / "artifacts" / "blackwell"
    older_bundle = bundle_root / "run_older"
    latest_bundle = bundle_root / "run_latest"
    sample_bundle = bundle_root / "sample_bundle"
    relabeled_sample_bundle = bundle_root / "run_relabeled_sample"
    _write_valid_bundle(older_bundle)
    _write_valid_bundle(latest_bundle)
    _write_valid_bundle(sample_bundle)
    _write_valid_bundle(relabeled_sample_bundle, is_sample=True)

    os.utime(older_bundle / "flash_backend_smoke.json", (100, 100))
    os.utime(latest_bundle / "flash_backend_smoke.json", (200, 200))
    os.utime(sample_bundle / "flash_backend_smoke.json", (300, 300))
    os.utime(relabeled_sample_bundle / "flash_backend_smoke.json", (400, 400))

    def _fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            "auto",
            "--bundle-root",
            str(bundle_root),
            "--check-in",
        ],
    )

    checker.main()
    stdout = capsys.readouterr().out
    assert f"bundle_dir={latest_bundle}" in stdout


def test_main_auto_rejects_when_only_sample_bundle_exists(tmp_path, monkeypatch):
    bundle_root = tmp_path / "artifacts" / "blackwell"
    sample_bundle = bundle_root / "sample_bundle"
    _write_valid_bundle(sample_bundle)

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            "auto",
            "--bundle-root",
            str(bundle_root),
        ],
    )

    with pytest.raises(RuntimeError, match="no real Blackwell bundle found"):
        checker.main()


def test_main_auto_rejection_error_lists_candidate_reasons(tmp_path, monkeypatch):
    bundle_root = tmp_path / "artifacts" / "blackwell"
    sample_bundle = bundle_root / "sample_bundle"
    incomplete_bundle = bundle_root / "run_incomplete"
    malformed_bundle = bundle_root / "run_malformed"
    runbook_only_bundle = bundle_root / "runbook_only"
    _write_valid_bundle(sample_bundle)

    incomplete_bundle.mkdir(parents=True, exist_ok=True)
    (incomplete_bundle / "flash_backend_smoke.json").write_text("{}", encoding="utf-8")

    malformed_bundle.mkdir(parents=True, exist_ok=True)
    (malformed_bundle / "flash_backend_smoke.json").write_text("{not-json", encoding="utf-8")
    (malformed_bundle / "flash_backend_status.log").write_text("status\n", encoding="utf-8")
    (malformed_bundle / "blackwell_smoke_evidence.md").write_text("# evidence\n", encoding="utf-8")
    (malformed_bundle / "blackwell_smoke_runbook.md").write_text("# runbook\n", encoding="utf-8")

    runbook_only_bundle.mkdir(parents=True, exist_ok=True)
    (runbook_only_bundle / "blackwell_smoke_runbook.md").write_text("# runbook\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            "auto",
            "--bundle-root",
            str(bundle_root),
        ],
    )

    with pytest.raises(RuntimeError) as exc_info:
        checker.main()

    message = str(exc_info.value)
    assert "discovery searched for required Blackwell bundle files" in message
    assert "rejected 4 candidate bundle(s)" in message
    assert f"{sample_bundle}: sample path segment" in message
    assert (
        f"{incomplete_bundle}: missing files: flash_backend_status.log, blackwell_smoke_evidence.md, "
        "blackwell_smoke_runbook.md"
    ) in message
    assert f"{malformed_bundle}: invalid flash_backend_smoke.json:" in message
    assert (
        f"{runbook_only_bundle}: missing files: flash_backend_smoke.json, flash_backend_status.log, "
        "blackwell_smoke_evidence.md"
    ) in message


def test_main_auto_rejects_missing_bundle_root(tmp_path, monkeypatch):
    missing_root = tmp_path / "does_not_exist"

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            "auto",
            "--bundle-root",
            str(missing_root),
        ],
    )

    with pytest.raises(RuntimeError, match="bundle_root does not exist"):
        checker.main()


def test_main_dry_run_skips_bundle_validation(tmp_path, monkeypatch, capsys):
    bundle_dir = tmp_path / "blackwell"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    report_path = bundle_dir / "dry_run_report.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--check-in",
            "--require-real-bundle",
            "--output-check-json",
            str(report_path),
            "--dry-run",
        ],
    )

    checker.main()
    stdout = capsys.readouterr().out
    assert "bundle_check_dry_run_ok" in stdout
    assert f"bundle_dir={bundle_dir}" in stdout
    assert "check_in=True" in stdout
    assert "require_blackwell=True" in stdout
    assert "require_git_tracked=True" in stdout
    assert "require_real_bundle=True" in stdout
    assert "require_device_substring=RTX 5090" in stdout
    assert f"output_check_json={report_path}" in stdout
    assert not report_path.exists()


def test_main_dry_run_uses_none_for_empty_report_path(tmp_path, monkeypatch, capsys):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--dry-run",
        ],
    )

    checker.main()
    stdout = capsys.readouterr().out
    assert "bundle_check_dry_run_ok" in stdout
    assert "output_check_json=<none>" in stdout


def test_main_preflight_accepts_complete_bundle(tmp_path, monkeypatch, capsys):
    bundle_dir = tmp_path / "blackwell"
    _write_valid_bundle(bundle_dir)

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--preflight",
        ],
    )

    checker.main()
    stdout = capsys.readouterr().out
    assert "bundle_preflight_ok" in stdout
    assert f"bundle_dir={bundle_dir}" in stdout


def test_main_preflight_rejects_missing_files_with_actionable_error(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "blackwell_smoke_runbook.md").write_text("# runbook\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--preflight",
        ],
    )

    with pytest.raises(RuntimeError) as exc_info:
        checker.main()

    message = str(exc_info.value)
    assert "bundle preflight failed" in message
    assert "flash_backend_smoke.json" in message
    assert "flash_backend_status.log" in message
    assert "blackwell_smoke_evidence.md" in message
    assert "scripts.run_blackwell_smoke_bundle --output-dir" in message


def test_main_preflight_mutually_exclusive_with_dry_run(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "blackwell"

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_blackwell_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--preflight",
            "--dry-run",
        ],
    )

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        checker.main()
