"""Offline validation for a recorded Blackwell smoke artifact bundle.

Example:
python -m scripts.check_blackwell_evidence_bundle --bundle-dir artifacts/blackwell_smoke --expect-backend fa4
"""

import argparse
from pathlib import Path

from scripts.validate_blackwell_smoke_artifact import (
    _load_artifact,
    _load_status_line,
    _validate_artifact,
    _validate_status_line_consistency,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate recorded Blackwell smoke bundle directory")
    parser.add_argument("--bundle-dir", required=True, help="directory emitted by scripts.run_blackwell_smoke_bundle")
    parser.add_argument("--expect-backend", choices=["fa4", "fa3", "sdpa"], default="fa4")
    parser.add_argument(
        "--require-blackwell",
        action="store_true",
        help="require artifact CUDA capability to be sm100+",
    )
    return parser.parse_args()


def _required_paths(bundle_dir: Path) -> dict[str, Path]:
    return {
        "artifact_json": bundle_dir / "flash_backend_smoke.json",
        "status_line": bundle_dir / "flash_backend_status.log",
        "evidence_md": bundle_dir / "blackwell_smoke_evidence.md",
        "runbook_md": bundle_dir / "blackwell_smoke_runbook.md",
    }


def _assert_files_exist(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing {label} file: {path}")


def _assert_evidence_content(evidence_text: str, selected_backend: str) -> None:
    required_lines = [
        "# Blackwell Flash Backend Smoke Evidence",
        f"- selected_backend: `{selected_backend}`",
        "- status_line_ok: `true`",
    ]
    for line in required_lines:
        if line not in evidence_text:
            raise RuntimeError(f"evidence markdown missing line: {line}")


def _assert_runbook_content(runbook_text: str, bundle_dir: Path, expect_backend: str) -> None:
    expected_path = str(bundle_dir)
    expected_evidence = str(bundle_dir / "blackwell_smoke_evidence.md")
    expected_snippets = [
        "# Blackwell Smoke Bundle Runbook",
        "python -m scripts.run_blackwell_smoke_bundle",
        f"--output-dir {expected_path}",
        f"--expect-backend {expect_backend}",
        f"- `{expected_evidence}`",
        f"- Ensure command prints `bundle_ok selected={expect_backend}`.",
    ]
    for snippet in expected_snippets:
        if snippet not in runbook_text:
            raise RuntimeError(f"runbook markdown missing snippet: {snippet}")


def main() -> None:
    args = _parse_args()
    bundle_dir = Path(args.bundle_dir)
    paths = _required_paths(bundle_dir)
    _assert_files_exist(paths)

    payload = _load_artifact(str(paths["artifact_json"]))
    selected_backend, _capability = _validate_artifact(payload, args.expect_backend, args.require_blackwell)

    status_line = _load_status_line(str(paths["status_line"]))
    status_backend = _validate_status_line_consistency(payload, status_line)
    if status_backend != args.expect_backend:
        raise RuntimeError(f"expected backend {args.expect_backend}, got {status_backend} in status-line file")

    evidence_text = paths["evidence_md"].read_text(encoding="utf-8")
    _assert_evidence_content(evidence_text, selected_backend)

    runbook_text = paths["runbook_md"].read_text(encoding="utf-8")
    _assert_runbook_content(runbook_text, bundle_dir, args.expect_backend)

    print(f"bundle_check_ok selected={selected_backend} bundle_dir={bundle_dir}")


if __name__ == "__main__":
    main()
