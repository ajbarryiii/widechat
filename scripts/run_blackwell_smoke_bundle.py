"""Run Blackwell smoke + artifact validation in one command.

Example:
python -m scripts.run_blackwell_smoke_bundle --output-dir artifacts/blackwell_smoke
"""

import argparse
import shlex
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
    parser.add_argument(
        "--output-dir",
        required=True,
        help="artifact directory for smoke JSON/status/evidence",
    )
    parser.add_argument("--expect-backend", choices=["fa4", "fa3", "sdpa"], default="fa4")
    parser.add_argument(
        "--output-evidence-md",
        default="",
        help="optional markdown path (defaults to <output-dir>/blackwell_smoke_evidence.md)",
    )
    parser.add_argument(
        "--output-runbook-md",
        default="",
        help="optional markdown path for a check-in runbook (defaults to <output-dir>/blackwell_smoke_runbook.md)",
    )
    return parser.parse_args()


def _write_runbook_markdown(path: str, output_dir: str, expect_backend: str, evidence_md: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quoted_output_dir = shlex.quote(output_dir)
    quoted_expect_backend = shlex.quote(expect_backend)
    quoted_check_json = shlex.quote(f"{output_dir}/blackwell_bundle_check.json")
    lines = [
        "# Blackwell Smoke Bundle Runbook",
        "",
        "## Command",
        "```bash",
        "python -m scripts.run_blackwell_smoke_bundle \\",
        f"  --output-dir {quoted_output_dir} \\",
        f"  --expect-backend {quoted_expect_backend}",
        "```",
        "",
        "## Expected outputs",
        f"- `{output_dir}/flash_backend_smoke.json`",
        f"- `{output_dir}/flash_backend_status.log`",
        f"- `{evidence_md}`",
        "",
        "## Check-in checklist",
        f"- Ensure command prints `bundle_ok selected={expect_backend}`.",
        "- Run `python -m scripts.run_blackwell_check_in --bundle-dir"
        f" {quoted_output_dir} --expect-backend {quoted_expect_backend} --output-check-json"
        f" {quoted_check_json}`.",
        "- Verify evidence markdown includes device metadata and `status_line_ok: true`.",
        f"- Confirm `{output_dir}/blackwell_bundle_check.json` records `selected_backend: {expect_backend}`.",
        "- Commit the emitted evidence artifacts from this run.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    _validate_environment(require_cuda=True, require_blackwell=True)

    artifact_json, status_line_path = _resolve_output_paths("", "", args.output_dir)
    evidence_md = args.output_evidence_md or str(Path(args.output_dir) / "blackwell_smoke_evidence.md")
    runbook_md = args.output_runbook_md or str(Path(args.output_dir) / "blackwell_smoke_runbook.md")

    _write_runbook_markdown(
        path=runbook_md,
        output_dir=args.output_dir,
        expect_backend=args.expect_backend,
        evidence_md=evidence_md,
    )

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
        f"evidence_md={evidence_md} "
        f"runbook_md={runbook_md}"
    )


if __name__ == "__main__":
    main()
