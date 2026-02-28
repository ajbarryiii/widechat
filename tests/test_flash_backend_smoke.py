import json
from datetime import datetime

import pytest

from scripts import flash_backend_smoke
from scripts.flash_backend_smoke import (
    _cuda_unavailable_diagnostics,
    _device_metadata,
    _extract_selected_backend,
    _resolve_output_paths,
    _validate_environment,
    _write_smoke_artifact,
    _write_status_line,
)


def test_extract_selected_backend_parses_known_backends():
    assert _extract_selected_backend("Flash Attention backend selection: selected=fa4, mode=auto") == "fa4"
    assert _extract_selected_backend("Flash Attention backend selection: selected=fa3, mode=auto") == "fa3"
    assert _extract_selected_backend("Flash Attention backend selection: selected=sdpa, mode=auto") == "sdpa"


def test_extract_selected_backend_rejects_malformed_status_line():
    with pytest.raises(ValueError, match="cannot parse backend"):
        _extract_selected_backend("Flash Attention backend selection: mode=auto")


def test_validate_environment_fails_when_cuda_required_but_missing(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        _validate_environment(require_cuda=True, require_blackwell=False)


def test_validate_environment_includes_nvidia_smi_diagnostics_when_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    class _Result:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 5090, 570.86.16\n"

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: _Result())

    with pytest.raises(RuntimeError, match=r"nvidia-smi reports GPU\(s\)"):
        _validate_environment(require_cuda=True, require_blackwell=False)


def test_cuda_unavailable_diagnostics_returns_empty_when_nvidia_smi_missing(monkeypatch):
    def _raise_missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr("subprocess.run", _raise_missing)

    assert _cuda_unavailable_diagnostics() == ""


def test_validate_environment_fails_blackwell_requirement_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Blackwell check requires CUDA"):
        _validate_environment(require_cuda=False, require_blackwell=True)


def test_validate_environment_fails_blackwell_requirement_pre_sm100(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (9, 0))
    with pytest.raises(RuntimeError, match="must be sm100"):
        _validate_environment(require_cuda=False, require_blackwell=True)


def test_validate_environment_accepts_sm100(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (10, 0))
    _validate_environment(require_cuda=True, require_blackwell=True)


def test_validate_environment_rejects_missing_required_device_substring(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda: "NVIDIA H100")
    with pytest.raises(RuntimeError, match="does not include required substring"):
        _validate_environment(require_cuda=False, require_blackwell=False, require_device_substring="RTX 5090")


def test_validate_environment_accepts_required_device_substring_case_insensitive(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda: "NVIDIA GeForce RTX 5090")
    _validate_environment(require_cuda=False, require_blackwell=False, require_device_substring="rtx 5090")


def test_validate_environment_rejects_device_substring_check_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="device-name check requires CUDA"):
        _validate_environment(require_cuda=False, require_blackwell=False, require_device_substring="RTX 5090")


def test_write_smoke_artifact_writes_expected_payload(tmp_path):
    output = tmp_path / "artifacts" / "flash_backend.json"
    status = "Flash Attention backend selection: selected=fa4, mode=auto"
    diagnostics = {
        "mode": "auto",
        "has_fa4": True,
        "has_fa3": False,
        "fa4_probe": "available",
        "fa3_probe": "unsupported_cc_sm100",
    }
    _write_smoke_artifact(str(output), status, "fa4", diagnostics=diagnostics)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_backend"] == "fa4"
    assert payload["status_line"] == status
    assert payload["is_sample"] is False
    assert isinstance(payload["cuda_available"], bool)
    assert "device_name" in payload
    assert "cuda_capability" in payload
    assert payload["selection_mode"] == "auto"
    assert payload["has_fa4"] is True
    assert payload["has_fa3"] is False
    assert payload["fa4_probe"] == "available"
    assert payload["fa3_probe"] == "unsupported_cc_sm100"
    assert isinstance(payload["nvidia_smi_ok"], bool)
    assert isinstance(payload["nvidia_smi"], list)
    assert "nvidia_smi_error" in payload
    assert payload["generated_at_utc"].endswith("Z")
    datetime.strptime(payload["generated_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
    assert "git_commit" in payload


def test_device_metadata_reports_no_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    metadata = _device_metadata()
    assert metadata == {
        "cuda_available": False,
        "device_name": None,
        "cuda_capability": None,
    }


def test_device_metadata_reports_capability_when_cuda_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda: "RTX 5090")
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (10, 0))

    metadata = _device_metadata()
    assert metadata == {
        "cuda_available": True,
        "device_name": "RTX 5090",
        "cuda_capability": [10, 0],
    }


def test_write_status_line_writes_canonical_line(tmp_path):
    output = tmp_path / "logs" / "backend_status.log"
    status = "Flash Attention backend selection: selected=fa4, mode=auto\n"

    _write_status_line(str(output), status)

    assert output.read_text(encoding="utf-8") == "Flash Attention backend selection: selected=fa4, mode=auto\n"


def test_resolve_output_paths_expands_output_dir_only():
    output_json, output_status = _resolve_output_paths("", "", "artifacts/blackwell")

    assert output_json == "artifacts/blackwell/flash_backend_smoke.json"
    assert output_status == "artifacts/blackwell/flash_backend_status.log"


def test_resolve_output_paths_rejects_mixed_output_flags():
    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_output_paths("artifact.json", "", "artifacts/blackwell")


def test_main_output_dir_writes_both_artifacts(tmp_path, monkeypatch):
    output_dir = tmp_path / "smoke"
    status = "Flash Attention backend selection: selected=fa4, mode=auto"
    diagnostics = {
        "mode": "auto",
        "has_fa4": True,
        "has_fa3": False,
        "fa4_probe": "available",
        "fa3_probe": "unsupported_cc_sm100",
    }
    monkeypatch.setattr(flash_backend_smoke, "backend_status_message", lambda: status)
    monkeypatch.setattr(flash_backend_smoke, "backend_diagnostics", lambda: diagnostics)
    monkeypatch.setattr(
        "sys.argv",
        [
            "flash_backend_smoke.py",
            "--expect-backend",
            "fa4",
            "--output-dir",
            str(output_dir),
        ],
    )

    flash_backend_smoke.main()

    artifact_json = output_dir / "flash_backend_smoke.json"
    artifact_status = output_dir / "flash_backend_status.log"
    assert artifact_json.exists()
    assert artifact_status.exists()
    payload = json.loads(artifact_json.read_text(encoding="utf-8"))
    assert payload["status_line"] == status
    assert payload["selected_backend"] == "fa4"
    assert payload["selection_mode"] == "auto"
    assert payload["fa4_probe"] == "available"
    assert payload["generated_at_utc"].endswith("Z")
    assert "git_commit" in payload
    assert artifact_status.read_text(encoding="utf-8") == f"{status}\n"
