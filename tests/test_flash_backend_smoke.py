import json

import pytest

from scripts.flash_backend_smoke import _extract_selected_backend, _validate_environment, _write_smoke_artifact


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
    assert payload == {
        "selected_backend": "fa4",
        "status_line": status,
    }
