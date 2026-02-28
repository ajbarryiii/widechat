"""Run Blackwell smoke + artifact validation in one command.

Example:
python -m scripts.run_blackwell_smoke_bundle --output-dir artifacts/blackwell_smoke
"""

import argparse
from pathlib import Path

from scripts.flash_backend_smoke import (
    _extract_selected_backend,
    _resolve_output_paths,
    _validate_environment,
    _write_smoke_artifact,
    _write_status_line,
    backend_status_message,
)
from scripts.validate_blackwell_smoke_artifact import (
    _load_artifact,
    _load_status_line,
    _validate_artifact,
    _validate_status_line_consistency,
    _write_evidence_markdown,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Blackwell smoke artifact bundle")
    parser.add_argument("--output-dir", required=True, help="artifact directory for smoke JSON/status/evidence")
    parser.add_argument("--expect-backend", choices=["fa4", "fa3", "sdpa"], default="fa4")
    parser.add_argument(
        "--output-evidence-md",
        default="",
        help="optional markdown path (defaults to <output-dir>/blackwell_smoke_evidence.md)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _validate_environment(require_cuda=True, require_blackwell=True)

    artifact_json, status_line_path = _resolve_output_paths("", "", args.output_dir)
    evidence_md = args.output_evidence_md or str(Path(args.output_dir) / "blackwell_smoke_evidence.md")

    status_line = backend_status_message()
    print(status_line)
    selected_backend = _extract_selected_backend(status_line)
    if selected_backend != args.expect_backend:
        raise RuntimeError(f"expected backend {args.expect_backend}, got {selected_backend}")

    _write_smoke_artifact(artifact_json, status_line, selected_backend)
    _write_status_line(status_line_path, status_line)

    payload = _load_artifact(artifact_json)
    selected_backend, capability = _validate_artifact(
        payload,
        expect_backend=args.expect_backend,
        require_blackwell=True,
    )

    recorded_status_line = _load_status_line(status_line_path)
    status_line_backend = _validate_status_line_consistency(payload, recorded_status_line)
    if status_line_backend != args.expect_backend:
        raise RuntimeError(f"expected backend {args.expect_backend}, got {status_line_backend} in status-line file")

    _write_evidence_markdown(
        path=evidence_md,
        payload=payload,
        selected_backend=selected_backend,
        capability=capability,
        status_line_ok=True,
    )

    print(
        "bundle_ok "
        f"selected={selected_backend} "
        f"artifact_json={artifact_json} "
        f"status_line={status_line_path} "
        f"evidence_md={evidence_md}"
    )


if __name__ == "__main__":
    main()
