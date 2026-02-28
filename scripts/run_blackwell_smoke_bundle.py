"""Run Blackwell smoke + artifact validation in one command.

Example:
python -m scripts.run_blackwell_smoke_bundle --output-dir artifacts/blackwell_smoke
"""

import argparse
from datetime import datetime, timezone
import json
import shlex
from pathlib import Path

from scripts.flash_backend_smoke import (
    _device_metadata,
    _extract_selected_backend,
    _query_nvidia_smi_gpus,
    _resolve_output_paths,
    _validate_environment,
    _write_smoke_artifact,
    _write_status_line,
    backend_status_message,
)
from scripts.check_blackwell_evidence_bundle import run_bundle_check
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
        "--require-device-substring",
        default="RTX 5090",
        help="case-insensitive device_name substring required in artifact metadata",
    )
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only emit runbook/planned paths without probing CUDA or writing smoke artifacts",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate CUDA/Blackwell/device prerequisites and emit a preflight receipt without running backend smoke",
    )
    parser.add_argument(
        "--output-preflight-json",
        default="",
        help="optional preflight receipt path (defaults to <output-dir>/blackwell_smoke_preflight.json when --preflight)",
    )
    parser.add_argument(
        "--output-blocked-md",
        default="",
        help="optional markdown blocker receipt path (defaults to <output-dir>/blackwell_smoke_blocked.md on failed --preflight)",
    )
    parser.add_argument(
        "--run-bundle-check",
        action="store_true",
        help="run offline bundle checker after smoke capture and write a checker receipt",
    )
    parser.add_argument(
        "--output-check-json",
        default="",
        help="optional checker receipt path (defaults to <output-dir>/blackwell_bundle_check.json when --run-bundle-check)",
    )
    parser.add_argument(
        "--run-strict-check-in",
        action="store_true",
        help="run strict --check-in bundle validation after smoke capture and write a strict checker receipt",
    )
    parser.add_argument(
        "--output-strict-check-json",
        default="",
        help=(
            "optional strict checker receipt path "
            "(defaults to <output-dir>/blackwell_bundle_check_strict.json when --run-strict-check-in)"
        ),
    )
    return parser.parse_args()


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_preflight_receipt(path: str, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_runbook_markdown(
    path: str,
    output_dir: str,
    expect_backend: str,
    evidence_md: str,
    check_json_path: str,
    strict_check_json_path: str,
    require_device_substring: str,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quoted_output_dir = shlex.quote(output_dir)
    quoted_expect_backend = shlex.quote(expect_backend)
    quoted_require_device_substring = shlex.quote(require_device_substring)
    quoted_check_json = shlex.quote(check_json_path)
    quoted_strict_check_json = shlex.quote(strict_check_json_path)
    lines = [
        "# Blackwell Smoke Bundle Runbook",
        "",
        "## Command",
        "```bash",
        "python -m scripts.run_blackwell_smoke_bundle \\",
        f"  --output-dir {quoted_output_dir} \\",
        f"  --expect-backend {quoted_expect_backend} \\",
        f"  --require-device-substring {quoted_require_device_substring}",
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
        f" {quoted_check_json} --require-device-substring {quoted_require_device_substring}`.",
        "- Equivalent strict checker command: `python -m scripts.check_blackwell_evidence_bundle"
        f" --bundle-dir {quoted_output_dir} --expect-backend {quoted_expect_backend} --check-in"
        f" --output-check-json {quoted_check_json} --require-device-substring"
        f" {quoted_require_device_substring}`.",
        "- Optional one-command strict check at smoke time: `python -m scripts.run_blackwell_smoke_bundle"
        f" --output-dir {quoted_output_dir} --expect-backend {quoted_expect_backend}"
        f" --require-device-substring {quoted_require_device_substring} --run-strict-check-in"
        f" --output-strict-check-json {quoted_strict_check_json}`.",
        "- Verify evidence markdown includes device metadata and `status_line_ok: true`.",
        f"- Confirm `{check_json_path}` records `selected_backend: {expect_backend}`.",
        "- Commit the emitted evidence artifacts from this run.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_blocked_markdown(path: str, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = str(payload.get("mode") or "preflight")
    is_preflight = mode == "preflight"
    title = "# Blackwell Smoke Preflight Blocker" if is_preflight else "# Blackwell Smoke Runtime Blocker"
    intro = (
        "This machine is not ready to run the Blackwell FA4 smoke bundle."
        if is_preflight
        else "Blackwell FA4 smoke execution failed before a complete evidence bundle was emitted."
    )
    require_device_substring = payload.get("require_device_substring") or ""
    quoted_output_dir = shlex.quote(str(payload.get("output_dir") or ""))
    quoted_expect_backend = shlex.quote(str(payload.get("expect_backend") or "fa4"))
    quoted_require_device_substring = shlex.quote(require_device_substring)
    selected_backend = payload.get("selected_backend")
    status_line = payload.get("status_line")
    selected_backend_line = [] if selected_backend is None else [f"- selected_backend: `{selected_backend}`"]
    status_line_line = [] if status_line is None else [f"- status_line: `{status_line}`"]
    lines = [
        title,
        "",
        intro,
        "",
        "## Receipt",
        f"- mode: `{mode}`",
        f"- generated_at_utc: `{payload.get('generated_at_utc') or ''}`",
        f"- ready: `{str(payload.get('ready')).lower()}`",
        f"- error: `{payload.get('error') or ''}`",
        f"- expect_backend: `{payload.get('expect_backend') or ''}`",
        *selected_backend_line,
        *status_line_line,
        f"- require_device_substring: `{require_device_substring}`",
        f"- cuda_available: `{str(payload.get('cuda_available')).lower()}`",
        f"- device_name: `{payload.get('device_name')}`",
        f"- cuda_capability: `{payload.get('cuda_capability')}`",
        f"- nvidia_smi_ok: `{str(payload.get('nvidia_smi_ok')).lower()}`",
        f"- nvidia_smi_error: `{payload.get('nvidia_smi_error')}`",
        "",
        "## Next command on RTX 5090",
        "```bash",
        "python -m scripts.run_blackwell_smoke_bundle \\",
        f"  --output-dir {quoted_output_dir} \\",
        f"  --expect-backend {quoted_expect_backend} \\",
        f"  --require-device-substring {quoted_require_device_substring}",
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _blocked_payload(
    *,
    mode: str,
    output_dir: str,
    expect_backend: str,
    require_device_substring: str,
    run_bundle_check: bool,
    run_strict_check_in: bool,
    artifact_json: str,
    status_line_path: str,
    evidence_md: str,
    runbook_md: str,
    output_check_json: str,
    output_strict_check_json: str,
    ready: bool,
    error: str,
    status_line: str | None = None,
    selected_backend: str | None = None,
) -> dict:
    payload = {
        "generated_at_utc": _now_utc(),
        "mode": mode,
        "ready": ready,
        "error": error,
        "expect_backend": expect_backend,
        "selected_backend": selected_backend,
        "status_line": status_line,
        "require_device_substring": require_device_substring,
        "output_dir": output_dir,
        "artifact_json": artifact_json,
        "status_line_path": status_line_path,
        "evidence_md": evidence_md,
        "runbook_md": runbook_md,
        "run_bundle_check": run_bundle_check,
        "run_strict_check_in": run_strict_check_in,
        "check_json": output_check_json if run_bundle_check else None,
        "strict_check_json": output_strict_check_json if run_strict_check_in else None,
    }
    payload.update(_device_metadata())
    nvidia_smi_ok, nvidia_smi_lines, nvidia_smi_error = _query_nvidia_smi_gpus()
    payload["nvidia_smi_ok"] = nvidia_smi_ok
    payload["nvidia_smi"] = nvidia_smi_lines
    payload["nvidia_smi_error"] = nvidia_smi_error
    return payload


def main() -> None:
    args = _parse_args()
    if args.dry_run and args.preflight:
        raise ValueError("--dry-run cannot be combined with --preflight")

    artifact_json, status_line_path = _resolve_output_paths("", "", args.output_dir)
    evidence_md = args.output_evidence_md or str(Path(args.output_dir) / "blackwell_smoke_evidence.md")
    runbook_md = args.output_runbook_md or str(Path(args.output_dir) / "blackwell_smoke_runbook.md")
    output_check_json = args.output_check_json or str(Path(args.output_dir) / "blackwell_bundle_check.json")
    output_strict_check_json = args.output_strict_check_json or str(
        Path(args.output_dir) / "blackwell_bundle_check_strict.json"
    )
    output_preflight_json = args.output_preflight_json or str(Path(args.output_dir) / "blackwell_smoke_preflight.json")
    output_blocked_md = args.output_blocked_md or str(Path(args.output_dir) / "blackwell_smoke_blocked.md")

    _write_runbook_markdown(
        path=runbook_md,
        output_dir=args.output_dir,
        expect_backend=args.expect_backend,
        evidence_md=evidence_md,
        check_json_path=output_check_json,
        strict_check_json_path=output_strict_check_json,
        require_device_substring=args.require_device_substring,
    )

    if args.dry_run:
        dry_run_check_json = output_check_json if args.run_bundle_check else "<none>"
        dry_run_strict_check_json = output_strict_check_json if args.run_strict_check_in else "<none>"
        print(
            "bundle_dry_run_ok "
            f"expect_backend={args.expect_backend} "
            f"artifact_json={artifact_json} "
            f"status_line={status_line_path} "
            f"evidence_md={evidence_md} "
            f"runbook_md={runbook_md} "
            f"run_bundle_check={args.run_bundle_check} "
            f"run_strict_check_in={args.run_strict_check_in} "
            f"require_device_substring={args.require_device_substring or '<none>'} "
            f"check_json={dry_run_check_json} "
            f"strict_check_json={dry_run_strict_check_json}"
        )
        return

    if args.preflight:
        preflight_error = ""
        preflight_ready = True
        try:
            _validate_environment(
                require_cuda=True,
                require_blackwell=True,
                require_device_substring=args.require_device_substring,
            )
        except RuntimeError as exc:
            preflight_ready = False
            preflight_error = str(exc)

        payload = _blocked_payload(
            mode="preflight",
            output_dir=args.output_dir,
            expect_backend=args.expect_backend,
            require_device_substring=args.require_device_substring,
            run_bundle_check=args.run_bundle_check,
            run_strict_check_in=args.run_strict_check_in,
            artifact_json=artifact_json,
            status_line_path=status_line_path,
            evidence_md=evidence_md,
            runbook_md=runbook_md,
            output_check_json=output_check_json,
            output_strict_check_json=output_strict_check_json,
            ready=preflight_ready,
            error=preflight_error,
        )
        _write_preflight_receipt(output_preflight_json, payload)

        if not preflight_ready:
            _write_blocked_markdown(output_blocked_md, payload)
            print(
                "bundle_preflight_blocked "
                f"preflight_json={output_preflight_json} "
                f"blocked_md={output_blocked_md} "
                f"reason={preflight_error}"
            )
            raise RuntimeError(f"Blackwell smoke preflight failed: {preflight_error}")

        print(
            "bundle_preflight_ok "
            f"preflight_json={output_preflight_json} "
            f"expect_backend={args.expect_backend} "
            f"require_device_substring={args.require_device_substring or '<none>'}"
        )
        return

    status_line = None
    selected_backend = None
    try:
        _validate_environment(
            require_cuda=True,
            require_blackwell=True,
            require_device_substring=args.require_device_substring,
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
            require_device_substring=args.require_device_substring,
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

        checker_receipt_path = "<none>"
        strict_checker_receipt_path = "<none>"
        if args.run_bundle_check:
            run_bundle_check(
                bundle_dir=Path(args.output_dir),
                expect_backend=args.expect_backend,
                check_in=False,
                require_blackwell=True,
                require_git_tracked=False,
                require_real_bundle=False,
                require_device_substring=args.require_device_substring,
                output_check_json=output_check_json,
            )
            checker_receipt_path = output_check_json

        if args.run_strict_check_in:
            run_bundle_check(
                bundle_dir=Path(args.output_dir),
                expect_backend=args.expect_backend,
                check_in=True,
                require_blackwell=True,
                require_git_tracked=False,
                require_real_bundle=True,
                require_device_substring=args.require_device_substring,
                output_check_json=output_strict_check_json,
            )
            strict_checker_receipt_path = output_strict_check_json

        print(
            "bundle_ok "
            f"selected={selected_backend} "
            f"artifact_json={artifact_json} "
            f"status_line={status_line_path} "
            f"evidence_md={evidence_md} "
            f"runbook_md={runbook_md} "
            f"check_json={checker_receipt_path} "
            f"strict_check_json={strict_checker_receipt_path}"
        )
    except RuntimeError as exc:
        blocked_payload = _blocked_payload(
            mode="smoke",
            output_dir=args.output_dir,
            expect_backend=args.expect_backend,
            require_device_substring=args.require_device_substring,
            run_bundle_check=args.run_bundle_check,
            run_strict_check_in=args.run_strict_check_in,
            artifact_json=artifact_json,
            status_line_path=status_line_path,
            evidence_md=evidence_md,
            runbook_md=runbook_md,
            output_check_json=output_check_json,
            output_strict_check_json=output_strict_check_json,
            ready=False,
            error=str(exc),
            status_line=status_line,
            selected_backend=selected_backend,
        )
        _write_blocked_markdown(output_blocked_md, blocked_payload)
        print(f"bundle_smoke_blocked blocked_md={output_blocked_md} reason={exc}")
        raise


if __name__ == "__main__":
    main()
