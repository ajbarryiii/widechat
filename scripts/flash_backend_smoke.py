"""Smoke-check Flash Attention backend selection for a single environment.

Example:
python -m scripts.flash_backend_smoke --expect-backend fa4 --require-cuda --require-blackwell
python -m scripts.flash_backend_smoke --output-json artifacts/flash_backend_smoke.json
python -m scripts.flash_backend_smoke --output-status-line artifacts/flash_backend_status.log
python -m scripts.flash_backend_smoke --output-dir artifacts/blackwell_smoke
"""

import argparse
from datetime import datetime, timezone
import json
import re
import subprocess
from pathlib import Path

import torch

from nanochat.flash_attention import backend_diagnostics, backend_status_message


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check Flash Attention backend selection")
    parser.add_argument(
        "--expect-backend",
        choices=["fa4", "fa3", "sdpa"],
        default="",
        help="fail if selected backend does not match",
    )
    parser.add_argument("--output-json", default="", help="optional path to write parsed backend artifact")
    parser.add_argument("--output-status-line", default="", help="optional path to write canonical backend status line")
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "optional artifact directory that writes both flash_backend_smoke.json and "
            "flash_backend_status.log"
        ),
    )
    parser.add_argument("--require-cuda", action="store_true", help="fail if CUDA is unavailable")
    parser.add_argument("--require-blackwell", action="store_true", help="fail unless CUDA capability is sm100+")
    return parser.parse_args()


def _extract_selected_backend(status_line: str) -> str:
    match = re.search(r"selected=(fa4|fa3|sdpa)", status_line)
    if match is None:
        raise ValueError(f"cannot parse backend from status line: {status_line!r}")
    return match.group(1)


def _validate_environment(require_cuda: bool, require_blackwell: bool) -> None:
    if require_cuda and not torch.cuda.is_available():
        details = _cuda_unavailable_diagnostics()
        if details:
            raise RuntimeError(f"CUDA is required but not available. {details}")
        raise RuntimeError("CUDA is required but not available")
    if require_blackwell:
        if not torch.cuda.is_available():
            raise RuntimeError("Blackwell check requires CUDA")
        major, _minor = torch.cuda.get_device_capability()
        if major < 10:
            raise RuntimeError("Blackwell check failed: CUDA capability must be sm100+")


def _cuda_unavailable_diagnostics() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return ""

    if result.returncode != 0:
        return ""

    gpu_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not gpu_lines:
        return ""

    gpu_summary = "; ".join(gpu_lines)
    return (
        f"nvidia-smi reports GPU(s): {gpu_summary}. "
        "This usually means the active PyTorch build lacks CUDA support or is mismatched with the system CUDA driver/runtime."
    )


def _device_metadata() -> dict[str, bool | str | list[int] | None]:
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        return {
            "cuda_available": False,
            "device_name": None,
            "cuda_capability": None,
        }

    major, minor = torch.cuda.get_device_capability()
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(),
        "cuda_capability": [major, minor],
    }


def _generated_at_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    commit = result.stdout.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        return None
    return commit


def _write_smoke_artifact(path: str, status_line: str, selected_backend: str, diagnostics: dict | None = None) -> None:
    metadata = _device_metadata()
    diagnostics = diagnostics or {}
    payload = {
        "status_line": status_line,
        "selected_backend": selected_backend,
        "is_sample": False,
        "generated_at_utc": _generated_at_utc(),
        "git_commit": _git_commit(),
        "selection_mode": diagnostics.get("mode"),
        "has_fa4": diagnostics.get("has_fa4"),
        "has_fa3": diagnostics.get("has_fa3"),
        "fa4_probe": diagnostics.get("fa4_probe"),
        "fa3_probe": diagnostics.get("fa3_probe"),
        **metadata,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_status_line(path: str, status_line: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(status_line.rstrip("\n") + "\n", encoding="utf-8")


def _resolve_output_paths(output_json: str, output_status_line: str, output_dir: str) -> tuple[str, str]:
    if output_dir:
        if output_json or output_status_line:
            raise ValueError("--output-dir cannot be combined with --output-json or --output-status-line")
        artifact_dir = Path(output_dir)
        return str(artifact_dir / "flash_backend_smoke.json"), str(artifact_dir / "flash_backend_status.log")
    return output_json, output_status_line


def main() -> None:
    args = _parse_args()
    _validate_environment(args.require_cuda, args.require_blackwell)
    output_json, output_status_line = _resolve_output_paths(args.output_json, args.output_status_line, args.output_dir)

    diagnostics = backend_diagnostics()
    status = backend_status_message()
    print(status)
    selected = _extract_selected_backend(status)
    if args.expect_backend and selected != args.expect_backend:
        raise RuntimeError(f"expected backend {args.expect_backend}, got {selected}")
    if output_json:
        _write_smoke_artifact(output_json, status, selected, diagnostics=diagnostics)
    if output_status_line:
        _write_status_line(output_status_line, status)


if __name__ == "__main__":
    main()
