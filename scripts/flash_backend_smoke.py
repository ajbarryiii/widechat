"""Smoke-check Flash Attention backend selection for a single environment.

Example:
python -m scripts.flash_backend_smoke --expect-backend fa4 --require-cuda --require-blackwell
python -m scripts.flash_backend_smoke --output-json artifacts/flash_backend_smoke.json
"""

import argparse
import json
import re
from pathlib import Path

import torch

from nanochat.flash_attention import backend_status_message


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check Flash Attention backend selection")
    parser.add_argument(
        "--expect-backend",
        choices=["fa4", "fa3", "sdpa"],
        default="",
        help="fail if selected backend does not match",
    )
    parser.add_argument("--output-json", default="", help="optional path to write parsed backend artifact")
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
        raise RuntimeError("CUDA is required but not available")
    if require_blackwell:
        if not torch.cuda.is_available():
            raise RuntimeError("Blackwell check requires CUDA")
        major, _minor = torch.cuda.get_device_capability()
        if major < 10:
            raise RuntimeError("Blackwell check failed: CUDA capability must be sm100+")


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


def _write_smoke_artifact(path: str, status_line: str, selected_backend: str) -> None:
    metadata = _device_metadata()
    payload = {
        "status_line": status_line,
        "selected_backend": selected_backend,
        **metadata,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    _validate_environment(args.require_cuda, args.require_blackwell)

    status = backend_status_message()
    print(status)
    selected = _extract_selected_backend(status)
    if args.expect_backend and selected != args.expect_backend:
        raise RuntimeError(f"expected backend {args.expect_backend}, got {selected}")
    if args.output_json:
        _write_smoke_artifact(args.output_json, status, selected)


if __name__ == "__main__":
    main()
