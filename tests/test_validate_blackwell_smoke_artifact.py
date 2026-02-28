import json

import pytest

from scripts import validate_blackwell_smoke_artifact as validator


def test_load_artifact_rejects_non_object(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        validator._load_artifact(str(path))


def test_validate_artifact_accepts_blackwell_fa4_payload():
    payload = {
        "selected_backend": "fa4",
        "status_line": "Flash Attention backend selection: selected=fa4, mode=auto",
        "cuda_available": True,
        "device_name": "RTX 5090",
        "cuda_capability": [10, 0],
    }

    selected, capability = validator._validate_artifact(payload, expect_backend="fa4", require_blackwell=True)
    assert selected == "fa4"
    assert capability == (10, 0)


def test_validate_artifact_rejects_wrong_backend():
    payload = {
        "selected_backend": "sdpa",
        "cuda_available": True,
        "cuda_capability": [10, 0],
    }

    with pytest.raises(RuntimeError, match="expected backend fa4"):
        validator._validate_artifact(payload, expect_backend="fa4", require_blackwell=True)


def test_validate_artifact_rejects_non_blackwell_capability():
    payload = {
        "selected_backend": "fa4",
        "cuda_available": True,
        "cuda_capability": [9, 0],
    }

    with pytest.raises(RuntimeError, match="not from Blackwell"):
        validator._validate_artifact(payload, expect_backend="fa4", require_blackwell=True)


def test_validate_artifact_rejects_malformed_capability_list():
    payload = {
        "selected_backend": "fa4",
        "cuda_available": True,
        "cuda_capability": [10],
    }

    with pytest.raises(ValueError, match="cuda_capability"):
        validator._validate_artifact(payload, expect_backend="fa4", require_blackwell=False)
