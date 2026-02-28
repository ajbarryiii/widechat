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
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    }

    selected, capability = validator._validate_artifact(payload, expect_backend="fa4", require_blackwell=True)
    assert selected == "fa4"
    assert capability == (10, 0)


def test_validate_artifact_rejects_wrong_backend():
    payload = {
        "selected_backend": "sdpa",
        "cuda_available": True,
        "cuda_capability": [10, 0],
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    }

    with pytest.raises(RuntimeError, match="expected backend fa4"):
        validator._validate_artifact(payload, expect_backend="fa4", require_blackwell=True)


def test_validate_artifact_rejects_non_blackwell_capability():
    payload = {
        "selected_backend": "fa4",
        "cuda_available": True,
        "cuda_capability": [9, 0],
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    }

    with pytest.raises(RuntimeError, match="not from Blackwell"):
        validator._validate_artifact(payload, expect_backend="fa4", require_blackwell=True)


def test_validate_artifact_rejects_malformed_capability_list():
    payload = {
        "selected_backend": "fa4",
        "cuda_available": True,
        "cuda_capability": [10],
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    }

    with pytest.raises(ValueError, match="cuda_capability"):
        validator._validate_artifact(payload, expect_backend="fa4", require_blackwell=False)


def test_load_status_line_rejects_empty_file(tmp_path):
    path = tmp_path / "flash_backend_status.log"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty line"):
        validator._load_status_line(str(path))


def test_validate_status_line_consistency_accepts_matching_payload():
    payload = {
        "selected_backend": "fa4",
        "status_line": "Flash Attention backend selection: selected=fa4, mode=auto",
        "cuda_available": True,
        "cuda_capability": [10, 0],
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    }

    parsed = validator._validate_status_line_consistency(payload, payload["status_line"])
    assert parsed == "fa4"


def test_validate_status_line_consistency_rejects_mismatch():
    payload = {
        "selected_backend": "fa4",
        "status_line": "Flash Attention backend selection: selected=fa4, mode=auto",
        "cuda_available": True,
        "cuda_capability": [10, 0],
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    }

    with pytest.raises(RuntimeError, match="does not match artifact status_line"):
        validator._validate_status_line_consistency(payload, "Flash Attention backend selection: selected=sdpa, mode=auto")


def test_main_with_status_line_file_reports_ok(tmp_path, monkeypatch, capsys):
    artifact_path = tmp_path / "flash_backend_smoke.json"
    status_path = tmp_path / "flash_backend_status.log"
    status = "Flash Attention backend selection: selected=fa4, mode=auto"
    artifact_payload = {
        "selected_backend": "fa4",
        "status_line": status,
        "cuda_available": True,
        "device_name": "RTX 5090",
        "cuda_capability": [10, 0],
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    }
    artifact_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
    status_path.write_text(status + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_blackwell_smoke_artifact.py",
            "--artifact-json",
            str(artifact_path),
            "--status-line-file",
            str(status_path),
            "--expect-backend",
            "fa4",
            "--require-blackwell",
        ],
    )

    validator.main()

    stdout = capsys.readouterr().out
    assert "artifact_ok selected=fa4 capability=sm100 status_line_ok=True" in stdout


def test_write_evidence_markdown_writes_selected_line(tmp_path):
    output_path = tmp_path / "evidence" / "blackwell_smoke.md"
    payload = {
        "selected_backend": "fa4",
        "status_line": "Flash Attention backend selection: selected=fa4, mode=auto",
        "cuda_available": True,
        "device_name": "RTX 5090",
        "cuda_capability": [10, 0],
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    }

    validator._write_evidence_markdown(
        path=str(output_path),
        payload=payload,
        selected_backend="fa4",
        capability=(10, 0),
        status_line_ok=True,
    )

    body = output_path.read_text(encoding="utf-8")
    assert "selected_backend: `fa4`" in body
    assert "generated_at_utc: `2026-02-27T00:00:00Z`" in body
    assert "git_commit: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`" in body
    assert "status_line: `Flash Attention backend selection: selected=fa4, mode=auto`" in body


def test_main_writes_evidence_markdown_when_requested(tmp_path, monkeypatch):
    artifact_path = tmp_path / "flash_backend_smoke.json"
    status_path = tmp_path / "flash_backend_status.log"
    evidence_path = tmp_path / "blackwell_smoke.md"
    status = "Flash Attention backend selection: selected=fa4, mode=auto"
    artifact_payload = {
        "selected_backend": "fa4",
        "status_line": status,
        "cuda_available": True,
        "device_name": "RTX 5090",
        "cuda_capability": [10, 0],
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    }
    artifact_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
    status_path.write_text(status + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_blackwell_smoke_artifact.py",
            "--artifact-json",
            str(artifact_path),
            "--status-line-file",
            str(status_path),
            "--expect-backend",
            "fa4",
            "--require-blackwell",
            "--output-evidence-md",
            str(evidence_path),
        ],
    )

    validator.main()

    body = evidence_path.read_text(encoding="utf-8")
    assert "selected_backend: `fa4`" in body
    assert "cuda_capability: `sm100`" in body
    assert "generated_at_utc: `2026-02-27T00:00:00Z`" in body
    assert "status_line_ok: `true`" in body


def test_validate_artifact_rejects_bad_generated_at_utc():
    payload = {
        "selected_backend": "fa4",
        "cuda_available": True,
        "cuda_capability": [10, 0],
        "generated_at_utc": "2026/02/27 00:00:00",
        "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    }

    with pytest.raises(ValueError, match="generated_at_utc"):
        validator._validate_artifact(payload, expect_backend="fa4", require_blackwell=False)


def test_validate_artifact_rejects_bad_git_commit():
    payload = {
        "selected_backend": "fa4",
        "cuda_available": True,
        "cuda_capability": [10, 0],
        "generated_at_utc": "2026-02-27T00:00:00Z",
        "git_commit": "deadbeef",
    }

    with pytest.raises(ValueError, match="git_commit"):
        validator._validate_artifact(payload, expect_backend="fa4", require_blackwell=False)
