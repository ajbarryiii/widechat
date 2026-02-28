import json
from datetime import datetime

import pytest

from scripts import flash_backend_smoke
from scripts.flash_backend_smoke import (
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


def test_write_smoke_artifact_writes_expected_payload(tmp_path):
    output = tmp_path / "artifacts" / "flash_backend.json"
    status = "Flash Attention backend selection: selected=fa4, mode=auto"
    _write_smoke_artifact(str(output), status, "fa4")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_backend"] == "fa4"
    assert payload["status_line"] == status
    assert payload["is_sample"] is False
    assert isinstance(payload["cuda_available"], bool)
    assert "device_name" in payload
    assert "cuda_capability" in payload
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
    monkeypatch.setattr(flash_backend_smoke, "backend_status_message", lambda: status)
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
    assert payload["generated_at_utc"].endswith("Z")
    assert "git_commit" in payload
    assert artifact_status.read_text(encoding="utf-8") == f"{status}\n"
