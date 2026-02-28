import json
import os
import subprocess
import sys

import pytest


def test_blackwell_on_device_smoke_selects_fa4(tmp_path):
    if os.environ.get("WIDECHAT_RUN_BLACKWELL_ON_DEVICE_SMOKE") != "1":
        pytest.skip("set WIDECHAT_RUN_BLACKWELL_ON_DEVICE_SMOKE=1 to run RTX 5090 smoke")

    artifact_dir = tmp_path / "blackwell_smoke"
    command = [
        sys.executable,
        "-m",
        "scripts.flash_backend_smoke",
        "--expect-backend",
        "fa4",
        "--require-cuda",
        "--require-blackwell",
        "--require-device-substring",
        "RTX 5090",
        "--output-dir",
        str(artifact_dir),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            "Blackwell smoke command failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    assert "selected=fa4" in result.stdout
    payload = json.loads((artifact_dir / "flash_backend_smoke.json").read_text(encoding="utf-8"))
    assert payload["selected_backend"] == "fa4"
    assert payload["device_name"] is not None
    assert "RTX 5090" in payload["device_name"]
    assert payload["cuda_capability"][0] >= 10
