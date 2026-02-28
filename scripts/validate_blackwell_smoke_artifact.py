"""Validate a recorded flash backend smoke artifact.

Example:
python -m scripts.validate_blackwell_smoke_artifact --artifact-json artifacts/flash_backend_smoke.json --require-blackwell --expect-backend fa4
"""

import argparse
from datetime import datetime
import json
import re
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate flash backend smoke artifact")
    parser.add_argument("--artifact-json", required=True, help="artifact produced by scripts.flash_backend_smoke --output-json")
    parser.add_argument(
        "--status-line-file",
        default="",
        help="optional flash_backend_status.log produced by scripts.flash_backend_smoke --output-status-line/--output-dir",
    )
    parser.add_argument("--expect-backend", choices=["fa4", "fa3", "sdpa"], default="fa4", help="required selected backend")
    parser.add_argument("--require-blackwell", action="store_true", help="require sm100+ CUDA capability in artifact")
    parser.add_argument(
        "--output-evidence-md",
        default="",
        help="optional path to write a markdown evidence summary for check-in/review",
    )
    return parser.parse_args()


def _load_artifact(path: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("artifact must be a JSON object")
    return payload


def _load_status_line(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        status_line = f.read().strip()
    if not status_line:
        raise ValueError("status-line file must include a non-empty line")
    return status_line


def _parse_capability(raw: object) -> tuple[int, int] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != 2 or not all(isinstance(v, int) for v in raw):
        raise ValueError("cuda_capability must be [major, minor] or null")
    return raw[0], raw[1]


def _parse_generated_at_utc(raw: object) -> str:
    if not isinstance(raw, str):
        raise ValueError("artifact missing generated_at_utc string")
    try:
        datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("generated_at_utc must be UTC ISO-8601 format YYYY-MM-DDTHH:MM:SSZ") from exc
    return raw


def _parse_git_commit(raw: object) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or re.fullmatch(r"[0-9a-f]{40}", raw) is None:
        raise ValueError("git_commit must be a 40-character lowercase hex SHA or null")
    return raw


def _extract_selected_backend(status_line: str) -> str:
    match = re.search(r"selected=(fa4|fa3|sdpa)", status_line)
    if match is None:
        raise ValueError(f"cannot parse backend from status line: {status_line!r}")
    return match.group(1)


def _validate_artifact(payload: dict[str, object], expect_backend: str, require_blackwell: bool) -> tuple[str, tuple[int, int] | None]:
    selected_backend = payload.get("selected_backend")
    if not isinstance(selected_backend, str):
        raise ValueError("artifact missing selected_backend string")
    if selected_backend != expect_backend:
        raise RuntimeError(f"expected backend {expect_backend}, got {selected_backend}")

    cuda_available = payload.get("cuda_available")
    if not isinstance(cuda_available, bool):
        raise ValueError("artifact missing cuda_available bool")

    capability = _parse_capability(payload.get("cuda_capability"))
    _parse_generated_at_utc(payload.get("generated_at_utc"))
    _parse_git_commit(payload.get("git_commit"))
    if require_blackwell:
        if not cuda_available:
            raise RuntimeError("artifact does not report CUDA availability")
        if capability is None:
            raise RuntimeError("artifact missing cuda_capability")
        major, _minor = capability
        if major < 10:
            raise RuntimeError("artifact is not from Blackwell (requires sm100+)")

    return selected_backend, capability


def _validate_status_line_consistency(payload: dict[str, object], status_line: str) -> str:
    payload_status_line = payload.get("status_line")
    if not isinstance(payload_status_line, str) or not payload_status_line:
        raise ValueError("artifact missing status_line string")
    if status_line != payload_status_line:
        raise RuntimeError("status-line file does not match artifact status_line")

    selected_backend = payload.get("selected_backend")
    if not isinstance(selected_backend, str):
        raise ValueError("artifact missing selected_backend string")

    parsed_backend = _extract_selected_backend(status_line)
    if parsed_backend != selected_backend:
        raise RuntimeError(
            f"status-line backend {parsed_backend} does not match artifact selected_backend {selected_backend}"
        )

    return parsed_backend


def _capability_str(capability: tuple[int, int] | None) -> str:
    if capability is None:
        return "none"
    return f"sm{capability[0]}{capability[1]}"


def _write_evidence_markdown(
    path: str,
    payload: dict[str, object],
    selected_backend: str,
    capability: tuple[int, int] | None,
    status_line_ok: bool,
) -> None:
    status_line = payload.get("status_line")
    if not isinstance(status_line, str) or not status_line:
        raise ValueError("artifact missing status_line string")

    cuda_available = payload.get("cuda_available")
    if not isinstance(cuda_available, bool):
        raise ValueError("artifact missing cuda_available bool")

    device_name = payload.get("device_name")
    if device_name is not None and not isinstance(device_name, str):
        raise ValueError("artifact device_name must be string or null")

    generated_at_utc = _parse_generated_at_utc(payload.get("generated_at_utc"))
    git_commit = _parse_git_commit(payload.get("git_commit"))

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Blackwell Flash Backend Smoke Evidence",
        "",
        f"- selected_backend: `{selected_backend}`",
        f"- cuda_available: `{str(cuda_available).lower()}`",
        f"- cuda_capability: `{_capability_str(capability)}`",
        f"- device_name: `{device_name if device_name is not None else 'none'}`",
        f"- generated_at_utc: `{generated_at_utc}`",
        f"- git_commit: `{git_commit if git_commit is not None else 'none'}`",
        f"- status_line_ok: `{str(status_line_ok).lower()}`",
        f"- status_line: `{status_line}`",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    payload = _load_artifact(args.artifact_json)
    selected_backend, capability = _validate_artifact(payload, args.expect_backend, args.require_blackwell)
    status_line_ok = False
    if args.status_line_file:
        status_line = _load_status_line(args.status_line_file)
        status_line_backend = _validate_status_line_consistency(payload, status_line)
        if status_line_backend != args.expect_backend:
            raise RuntimeError(f"expected backend {args.expect_backend}, got {status_line_backend} in status-line file")
        status_line_ok = True

    capability_str = _capability_str(capability)
    if args.output_evidence_md:
        _write_evidence_markdown(args.output_evidence_md, payload, selected_backend, capability, status_line_ok)
    print(f"artifact_ok selected={selected_backend} capability={capability_str} status_line_ok={status_line_ok}")


if __name__ == "__main__":
    main()
