import json
import shlex

import pytest

from scripts import run_blackwell_smoke_bundle as bundle


def test_main_writes_validated_artifact_bundle(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "blackwell"
    status = "Flash Attention backend selection: selected=fa4, mode=auto"

    monkeypatch.setattr(bundle, "_validate_environment", lambda require_cuda, require_blackwell, require_device_substring: None)
    monkeypatch.setattr(bundle, "backend_status_message", lambda: status)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda: "RTX 5090")
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_smoke_bundle.py",
            "--output-dir",
            str(output_dir),
        ],
    )

    bundle.main()

    artifact_json = output_dir / "flash_backend_smoke.json"
    status_line_path = output_dir / "flash_backend_status.log"
    evidence_md = output_dir / "blackwell_smoke_evidence.md"
    runbook_md = output_dir / "blackwell_smoke_runbook.md"
    payload = json.loads(artifact_json.read_text(encoding="utf-8"))

    assert payload["selected_backend"] == "fa4"
    assert status_line_path.read_text(encoding="utf-8") == f"{status}\n"
    assert "selected_backend: `fa4`" in evidence_md.read_text(encoding="utf-8")
    runbook_content = runbook_md.read_text(encoding="utf-8")
    assert "python -m scripts.run_blackwell_smoke_bundle" in runbook_content
    assert f"--output-dir {output_dir}" in runbook_content
    assert "--require-device-substring" in runbook_content
    assert "python -m scripts.run_blackwell_check_in --bundle-dir" in runbook_content
    assert "python -m scripts.check_blackwell_evidence_bundle --bundle-dir" in runbook_content
    assert "--check-in" in runbook_content
    assert "--output-check-json" in runbook_content
    assert f"{output_dir}/blackwell_bundle_check.json" in runbook_content
    assert "bundle_ok selected=fa4" in runbook_content

    stdout = capsys.readouterr().out
    assert "bundle_ok selected=fa4" in stdout
    assert f"runbook_md={runbook_md}" in stdout
    assert "check_json=<none>" in stdout


def test_main_optionally_runs_bundle_checker_and_writes_receipt(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "blackwell"
    status = "Flash Attention backend selection: selected=fa4, mode=auto"
    calls = {}

    def _fake_run_bundle_check(**kwargs):
        calls.update(kwargs)
        return "fa4"

    monkeypatch.setattr(bundle, "run_bundle_check", _fake_run_bundle_check)
    monkeypatch.setattr(bundle, "_validate_environment", lambda require_cuda, require_blackwell, require_device_substring: None)
    monkeypatch.setattr(bundle, "backend_status_message", lambda: status)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda: "RTX 5090")
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_smoke_bundle.py",
            "--output-dir",
            str(output_dir),
            "--run-bundle-check",
        ],
    )

    bundle.main()

    expected_receipt = output_dir / "blackwell_bundle_check.json"
    assert calls == {
        "bundle_dir": output_dir,
        "expect_backend": "fa4",
        "check_in": False,
        "require_blackwell": True,
        "require_git_tracked": False,
        "require_real_bundle": False,
        "require_device_substring": "RTX 5090",
        "output_check_json": str(expected_receipt),
    }

    stdout = capsys.readouterr().out
    assert f"check_json={expected_receipt}" in stdout


def test_main_honors_custom_check_receipt_path_in_runbook_and_checker_call(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "blackwell"
    custom_receipt = tmp_path / "receipts" / "bundle check.json"
    status = "Flash Attention backend selection: selected=fa4, mode=auto"
    calls = {}

    def _fake_run_bundle_check(**kwargs):
        calls.update(kwargs)
        return "fa4"

    monkeypatch.setattr(bundle, "run_bundle_check", _fake_run_bundle_check)
    monkeypatch.setattr(bundle, "_validate_environment", lambda require_cuda, require_blackwell, require_device_substring: None)
    monkeypatch.setattr(bundle, "backend_status_message", lambda: status)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda: "RTX 5090")
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_smoke_bundle.py",
            "--output-dir",
            str(output_dir),
            "--run-bundle-check",
            "--output-check-json",
            str(custom_receipt),
        ],
    )

    bundle.main()

    runbook_md = output_dir / "blackwell_smoke_runbook.md"
    runbook_content = runbook_md.read_text(encoding="utf-8")
    quoted_custom_receipt = shlex.quote(str(custom_receipt))

    assert calls["output_check_json"] == str(custom_receipt)
    assert quoted_custom_receipt in runbook_content

    stdout = capsys.readouterr().out
    assert f"check_json={custom_receipt}" in stdout


def test_main_rejects_unexpected_backend(tmp_path, monkeypatch):
    status = "Flash Attention backend selection: selected=sdpa, mode=auto"
    monkeypatch.setattr(bundle, "_validate_environment", lambda require_cuda, require_blackwell, require_device_substring: None)
    monkeypatch.setattr(bundle, "backend_status_message", lambda: status)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_smoke_bundle.py",
            "--output-dir",
            str(tmp_path / "blackwell"),
            "--expect-backend",
            "fa4",
        ],
    )

    with pytest.raises(RuntimeError, match="expected backend fa4, got sdpa"):
        bundle.main()


def test_main_passes_required_device_substring_to_environment_validation(tmp_path, monkeypatch):
    output_dir = tmp_path / "blackwell"
    status = "Flash Attention backend selection: selected=fa4, mode=auto"
    captured = {}

    def _fake_validate_environment(require_cuda, require_blackwell, require_device_substring):
        captured["require_cuda"] = require_cuda
        captured["require_blackwell"] = require_blackwell
        captured["require_device_substring"] = require_device_substring

    monkeypatch.setattr(bundle, "_validate_environment", _fake_validate_environment)
    monkeypatch.setattr(bundle, "backend_status_message", lambda: status)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda: "NVIDIA GeForce RTX 5090")
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_smoke_bundle.py",
            "--output-dir",
            str(output_dir),
            "--require-device-substring",
            "NVIDIA GeForce RTX 5090",
        ],
    )

    bundle.main()

    assert captured == {
        "require_cuda": True,
        "require_blackwell": True,
        "require_device_substring": "NVIDIA GeForce RTX 5090",
    }


def test_main_honors_custom_runbook_path(tmp_path, monkeypatch):
    output_dir = tmp_path / "blackwell"
    runbook_path = tmp_path / "custom" / "runbook.md"
    status = "Flash Attention backend selection: selected=fa4, mode=auto"

    monkeypatch.setattr(bundle, "_validate_environment", lambda require_cuda, require_blackwell, require_device_substring: None)
    monkeypatch.setattr(bundle, "backend_status_message", lambda: status)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda: "RTX 5090")
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_smoke_bundle.py",
            "--output-dir",
            str(output_dir),
            "--output-runbook-md",
            str(runbook_path),
        ],
    )

    bundle.main()

    assert runbook_path.exists()
    assert "# Blackwell Smoke Bundle Runbook" in runbook_path.read_text(encoding="utf-8")


def test_main_shell_quotes_runbook_commands_for_spaced_paths(tmp_path, monkeypatch):
    output_dir = tmp_path / "blackwell artifacts"
    status = "Flash Attention backend selection: selected=fa4, mode=auto"

    monkeypatch.setattr(bundle, "_validate_environment", lambda require_cuda, require_blackwell, require_device_substring: None)
    monkeypatch.setattr(bundle, "backend_status_message", lambda: status)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda: "RTX 5090")
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_smoke_bundle.py",
            "--output-dir",
            str(output_dir),
        ],
    )

    bundle.main()

    runbook_md = output_dir / "blackwell_smoke_runbook.md"
    runbook_content = runbook_md.read_text(encoding="utf-8")
    quoted_output_dir = shlex.quote(str(output_dir))
    quoted_check_json = shlex.quote(str(output_dir / "blackwell_bundle_check.json"))

    assert f"--output-dir {quoted_output_dir}" in runbook_content
    assert quoted_check_json in runbook_content
    assert "python -m scripts.check_blackwell_evidence_bundle --bundle-dir" in runbook_content


def test_main_dry_run_writes_runbook_without_cuda_probe(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "blackwell"
    calls = {"validated": False, "status": False}

    def _fake_validate_environment(require_cuda, require_blackwell, require_device_substring):
        calls["validated"] = True

    def _fake_backend_status_message():
        calls["status"] = True
        return "Flash Attention backend selection: selected=fa4, mode=auto"

    monkeypatch.setattr(bundle, "_validate_environment", _fake_validate_environment)
    monkeypatch.setattr(bundle, "backend_status_message", _fake_backend_status_message)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_smoke_bundle.py",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
    )

    bundle.main()

    assert calls["validated"] is False
    assert calls["status"] is False
    assert (output_dir / "blackwell_smoke_runbook.md").is_file()
    assert not (output_dir / "flash_backend_smoke.json").exists()
    assert not (output_dir / "flash_backend_status.log").exists()
    assert not (output_dir / "blackwell_smoke_evidence.md").exists()

    stdout = capsys.readouterr().out
    assert "bundle_dry_run_ok" in stdout
    assert f"artifact_json={output_dir / 'flash_backend_smoke.json'}" in stdout
    assert "run_bundle_check=False" in stdout
    assert "require_device_substring=RTX 5090" in stdout
    assert "check_json=<none>" in stdout


def test_main_dry_run_reports_checker_receipt_when_enabled(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "blackwell"
    calls = {"validated": False, "status": False}

    def _fake_validate_environment(require_cuda, require_blackwell, require_device_substring):
        calls["validated"] = True

    def _fake_backend_status_message():
        calls["status"] = True
        return "Flash Attention backend selection: selected=fa4, mode=auto"

    monkeypatch.setattr(bundle, "_validate_environment", _fake_validate_environment)
    monkeypatch.setattr(bundle, "backend_status_message", _fake_backend_status_message)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_smoke_bundle.py",
            "--output-dir",
            str(output_dir),
            "--run-bundle-check",
            "--dry-run",
        ],
    )

    bundle.main()

    assert calls["validated"] is False
    assert calls["status"] is False

    stdout = capsys.readouterr().out
    assert "run_bundle_check=True" in stdout
    assert f"check_json={output_dir / 'blackwell_bundle_check.json'}" in stdout


def test_main_preflight_writes_ready_receipt_without_backend_probe(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "blackwell"
    calls = {"validated": False, "status": False}

    def _fake_validate_environment(require_cuda, require_blackwell, require_device_substring):
        calls["validated"] = True

    def _fake_backend_status_message():
        calls["status"] = True
        return "Flash Attention backend selection: selected=fa4, mode=auto"

    monkeypatch.setattr(bundle, "_validate_environment", _fake_validate_environment)
    monkeypatch.setattr(bundle, "backend_status_message", _fake_backend_status_message)
    monkeypatch.setattr(
        bundle,
        "_device_metadata",
        lambda: {
            "cuda_available": True,
            "device_name": "NVIDIA GeForce RTX 5090",
            "cuda_capability": [10, 0],
        },
    )
    monkeypatch.setattr(
        bundle,
        "_query_nvidia_smi_gpus",
        lambda: (True, ["NVIDIA GeForce RTX 5090, 575.64"], None),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_smoke_bundle.py",
            "--output-dir",
            str(output_dir),
            "--preflight",
        ],
    )

    bundle.main()

    assert calls["validated"] is True
    assert calls["status"] is False
    assert (output_dir / "blackwell_smoke_runbook.md").is_file()
    assert not (output_dir / "flash_backend_smoke.json").exists()
    assert not (output_dir / "flash_backend_status.log").exists()
    assert not (output_dir / "blackwell_smoke_evidence.md").exists()

    receipt_path = output_dir / "blackwell_smoke_preflight.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["mode"] == "preflight"
    assert receipt["ready"] is True
    assert receipt["error"] == ""
    assert receipt["check_json"] is None
    assert receipt["cuda_available"] is True
    assert receipt["device_name"] == "NVIDIA GeForce RTX 5090"
    assert receipt["cuda_capability"] == [10, 0]
    assert receipt["nvidia_smi_ok"] is True
    assert receipt["nvidia_smi"] == ["NVIDIA GeForce RTX 5090, 575.64"]
    assert receipt["nvidia_smi_error"] is None

    stdout = capsys.readouterr().out
    assert "bundle_preflight_ok" in stdout
    assert f"preflight_json={receipt_path}" in stdout


def test_main_preflight_writes_blocked_receipt_and_raises(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "blackwell"
    reason = "CUDA is required but not available"
    calls = {"status": False}

    def _fake_validate_environment(require_cuda, require_blackwell, require_device_substring):
        raise RuntimeError(reason)

    def _fake_backend_status_message():
        calls["status"] = True
        return "Flash Attention backend selection: selected=fa4, mode=auto"

    monkeypatch.setattr(bundle, "_validate_environment", _fake_validate_environment)
    monkeypatch.setattr(bundle, "backend_status_message", _fake_backend_status_message)
    monkeypatch.setattr(
        bundle,
        "_device_metadata",
        lambda: {
            "cuda_available": False,
            "device_name": None,
            "cuda_capability": None,
        },
    )
    monkeypatch.setattr(
        bundle,
        "_query_nvidia_smi_gpus",
        lambda: (False, [], "nvidia-smi: command not found"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_smoke_bundle.py",
            "--output-dir",
            str(output_dir),
            "--preflight",
            "--run-bundle-check",
        ],
    )

    with pytest.raises(RuntimeError, match="Blackwell smoke preflight failed"):
        bundle.main()

    assert calls["status"] is False

    receipt_path = output_dir / "blackwell_smoke_preflight.json"
    blocked_md = output_dir / "blackwell_smoke_blocked.md"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["mode"] == "preflight"
    assert receipt["ready"] is False
    assert reason in receipt["error"]
    assert receipt["check_json"] == str(output_dir / "blackwell_bundle_check.json")
    assert receipt["cuda_available"] is False
    assert receipt["device_name"] is None
    assert receipt["cuda_capability"] is None
    assert receipt["nvidia_smi_ok"] is False
    assert receipt["nvidia_smi"] == []
    assert "nvidia-smi" in receipt["nvidia_smi_error"]
    blocked_content = blocked_md.read_text(encoding="utf-8")
    assert "# Blackwell Smoke Preflight Blocker" in blocked_content
    assert f"- error: `{reason}`" in blocked_content
    assert "--require-device-substring 'RTX 5090'" in blocked_content
    assert "This machine is not ready to run the Blackwell FA4 smoke bundle." in blocked_content

    stdout = capsys.readouterr().out
    assert "bundle_preflight_blocked" in stdout
    assert f"preflight_json={receipt_path}" in stdout
    assert f"blocked_md={blocked_md}" in stdout


def test_main_preflight_honors_custom_blocked_markdown_path(tmp_path, monkeypatch):
    output_dir = tmp_path / "blackwell"
    blocked_md = tmp_path / "receipts" / "custom blocker.md"

    def _fake_validate_environment(require_cuda, require_blackwell, require_device_substring):
        raise RuntimeError("CUDA is required but not available")

    monkeypatch.setattr(bundle, "_validate_environment", _fake_validate_environment)
    monkeypatch.setattr(
        bundle,
        "_device_metadata",
        lambda: {
            "cuda_available": False,
            "device_name": None,
            "cuda_capability": None,
        },
    )
    monkeypatch.setattr(
        bundle,
        "_query_nvidia_smi_gpus",
        lambda: (False, [], "nvidia-smi: command not found"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_smoke_bundle.py",
            "--output-dir",
            str(output_dir),
            "--preflight",
            "--output-blocked-md",
            str(blocked_md),
        ],
    )

    with pytest.raises(RuntimeError, match="Blackwell smoke preflight failed"):
        bundle.main()

    assert blocked_md.is_file()
    content = blocked_md.read_text(encoding="utf-8")
    assert "# Blackwell Smoke Preflight Blocker" in content


def test_main_rejects_preflight_with_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_blackwell_smoke_bundle.py",
            "--output-dir",
            str(tmp_path / "blackwell"),
            "--preflight",
            "--dry-run",
        ],
    )

    with pytest.raises(ValueError, match="--dry-run cannot be combined with --preflight"):
        bundle.main()
