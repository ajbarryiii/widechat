import json

import pytest

from scripts import run_blackwell_smoke_bundle as bundle


def test_main_writes_validated_artifact_bundle(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "blackwell"
    status = "Flash Attention backend selection: selected=fa4, mode=auto"

    monkeypatch.setattr(bundle, "_validate_environment", lambda require_cuda, require_blackwell: None)
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
    payload = json.loads(artifact_json.read_text(encoding="utf-8"))

    assert payload["selected_backend"] == "fa4"
    assert status_line_path.read_text(encoding="utf-8") == f"{status}\n"
    assert "selected_backend: `fa4`" in evidence_md.read_text(encoding="utf-8")

    stdout = capsys.readouterr().out
    assert "bundle_ok selected=fa4" in stdout


def test_main_rejects_unexpected_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "_validate_environment", lambda require_cuda, require_blackwell: None)
    monkeypatch.setattr(bundle, "backend_status_message", lambda: "Flash Attention backend selection: selected=sdpa, mode=auto")
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
