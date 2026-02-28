"""Smoke-check Flash Attention backend selection for a single environment.

Example:
python -m scripts.flash_backend_smoke --expect-backend fa4 --require-cuda --require-blackwell
"""

import argparse
import re

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


def main() -> None:
    args = _parse_args()
    _validate_environment(args.require_cuda, args.require_blackwell)

    status = backend_status_message()
    print(status)
    selected = _extract_selected_backend(status)
    if args.expect_backend and selected != args.expect_backend:
        raise RuntimeError(f"expected backend {args.expect_backend}, got {selected}")


if __name__ == "__main__":
    main()
