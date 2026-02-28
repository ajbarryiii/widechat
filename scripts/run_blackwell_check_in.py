"""Run strict Blackwell evidence check-in validation in one command.

Example:
python -m scripts.run_blackwell_check_in --bundle-dir artifacts/blackwell_smoke
"""

import argparse
import json
import shlex
import sys
from pathlib import Path

from scripts.check_blackwell_evidence_bundle import _resolve_bundle_dir as _resolve_bundle_dir_from_checker
from scripts.check_blackwell_evidence_bundle import run_bundle_check
from scripts.check_blackwell_evidence_bundle import run_bundle_preflight


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict check-in validation for Blackwell smoke bundle")
    parser.add_argument(
        "--bundle-dir",
        default="auto",
        help="directory emitted by scripts.run_blackwell_smoke_bundle, or 'auto' to discover latest real bundle",
    )
    parser.add_argument(
        "--bundle-root",
        default="artifacts/blackwell",
        help="bundle search root used when --bundle-dir=auto",
    )
    parser.add_argument("--expect-backend", choices=["fa4", "fa3", "sdpa"], default="fa4")
    parser.add_argument(
        "--require-device-substring",
        default="RTX 5090",
        help="case-insensitive device_name substring required in artifact metadata",
    )
    parser.add_argument(
        "--output-check-json",
        default="",
        help="optional path for checker receipt (defaults to <bundle-dir>/blackwell_bundle_check.json)",
    )
    parser.add_argument(
        "--output-check-md",
        default="",
        help="optional path for markdown check-in evidence summary",
    )
    parser.add_argument(
        "--output-check-command-sh",
        default="",
        help="optional path to write the resolved strict checker command",
    )
    parser.add_argument(
        "--allow-sample-bundle",
        action="store_true",
        help="allow sample fixture bundles (for local regression checks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only resolve paths and print planned checker invocation",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate required bundle files before strict check-in validation",
    )
    parser.add_argument(
        "--output-preflight-json",
        default="",
        help="optional path to write machine-readable preflight receipt JSON (preflight mode only)",
    )
    parser.add_argument(
        "--output-blocked-md",
        default="",
        help="optional path to write markdown blocker evidence when strict check-in fails",
    )
    return parser.parse_args()


def _resolve_bundle_dir(bundle_dir_arg: str, bundle_root_arg: str) -> Path:
    return _resolve_bundle_dir_from_checker(bundle_dir_arg, bundle_root_arg)


def _write_check_markdown(
    *,
    output_path: Path,
    bundle_dir: Path,
    expect_backend: str,
    selected_backend: str,
    check_json: str,
    require_device_substring: str,
    allow_sample_bundle: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Blackwell Strict Check-In Evidence",
        "",
        f"- selected_backend: `{selected_backend}`",
        f"- expect_backend: `{expect_backend}`",
        f"- bundle_dir: `{bundle_dir}`",
        f"- check_json: `{check_json}`",
        f"- require_real_bundle: `{str(not allow_sample_bundle).lower()}`",
        f"- require_device_substring: `{require_device_substring or '<none>'}`",
        "- check_in_mode: `true`",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_preflight_receipt(
    *,
    output_path: Path,
    bundle_dir: Path | None,
    expect_backend: str,
    output_check_json: str,
    require_device_substring: str,
    allow_sample_bundle: bool,
    ok: bool,
    error: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok" if ok else "blocked",
        "bundle_dir": str(bundle_dir) if bundle_dir is not None else "",
        "expect_backend": expect_backend,
        "check_json": output_check_json,
        "require_real_bundle": not allow_sample_bundle,
        "require_device_substring": require_device_substring,
        "ok": ok,
        "error": error,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_blocked_markdown(
    *,
    output_path: Path,
    command: list[str],
    reason: str,
    bundle_dir_arg: str,
    bundle_root_arg: str,
    expect_backend: str,
    require_device_substring: str,
    preflight_mode: bool,
    dry_run_mode: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Blackwell Strict Check-In Blocked",
        "",
        "## Context",
        f"- command: `{json.dumps(command)}`",
        f"- bundle_dir_arg: `{bundle_dir_arg}`",
        f"- bundle_root_arg: `{bundle_root_arg}`",
        f"- expect_backend: `{expect_backend}`",
        f"- require_device_substring: `{require_device_substring or '<none>'}`",
        f"- preflight_mode: `{str(preflight_mode).lower()}`",
        f"- dry_run_mode: `{str(dry_run_mode).lower()}`",
        "",
        "## Blocker",
        "```text",
        reason,
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _resolved_checker_command(
    *,
    bundle_dir: Path,
    expect_backend: str,
    output_check_json: str,
    require_device_substring: str,
    allow_sample_bundle: bool,
) -> str:
    command = [
        "python",
        "-m",
        "scripts.check_blackwell_evidence_bundle",
        "--bundle-dir",
        str(bundle_dir),
        "--expect-backend",
        expect_backend,
        "--check-in",
        "--output-check-json",
        output_check_json,
    ]
    if require_device_substring:
        command.extend(["--require-device-substring", require_device_substring])
    if not allow_sample_bundle:
        command.append("--require-real-bundle")
    return " ".join(shlex.quote(part) for part in command)


def _write_checker_command(path: str, command: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(command + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    try:
        if args.preflight and args.dry_run:
            raise RuntimeError("--preflight and --dry-run are mutually exclusive")
        if args.output_preflight_json and not args.preflight:
            raise RuntimeError("--output-preflight-json requires --preflight")

        bundle_dir: Path | None = None
        output_check_json = args.output_check_json

        try:
            bundle_dir = _resolve_bundle_dir(args.bundle_dir, args.bundle_root)
            if not output_check_json:
                output_check_json = str(bundle_dir / "blackwell_bundle_check.json")
        except RuntimeError as exc:
            if args.preflight and args.output_preflight_json:
                _write_preflight_receipt(
                    output_path=Path(args.output_preflight_json),
                    bundle_dir=None,
                    expect_backend=args.expect_backend,
                    output_check_json=output_check_json,
                    require_device_substring=args.require_device_substring,
                    allow_sample_bundle=args.allow_sample_bundle,
                    ok=False,
                    error=str(exc),
                )
            raise

        assert bundle_dir is not None
        output_check_json = output_check_json or str(bundle_dir / "blackwell_bundle_check.json")
        output_check_md = args.output_check_md
        resolved_command = _resolved_checker_command(
            bundle_dir=bundle_dir,
            expect_backend=args.expect_backend,
            output_check_json=output_check_json,
            require_device_substring=args.require_device_substring,
            allow_sample_bundle=args.allow_sample_bundle,
        )

        if args.output_check_command_sh:
            _write_checker_command(args.output_check_command_sh, resolved_command)

        if args.preflight:
            preflight_ok = True
            preflight_error = ""
            try:
                run_bundle_preflight(
                    bundle_dir=bundle_dir,
                    require_real_bundle=not args.allow_sample_bundle,
                )
            except RuntimeError as exc:
                preflight_ok = False
                preflight_error = str(exc)

            if args.output_preflight_json:
                _write_preflight_receipt(
                    output_path=Path(args.output_preflight_json),
                    bundle_dir=bundle_dir,
                    expect_backend=args.expect_backend,
                    output_check_json=output_check_json,
                    require_device_substring=args.require_device_substring,
                    allow_sample_bundle=args.allow_sample_bundle,
                    ok=preflight_ok,
                    error=preflight_error,
                )

            if not preflight_ok:
                raise RuntimeError(preflight_error)

            print(
                "blackwell_check_in_preflight_ok "
                f"bundle_dir={bundle_dir} "
                f"require_real_bundle={not args.allow_sample_bundle}"
                + (f" preflight_json={args.output_preflight_json}" if args.output_preflight_json else "")
            )
            return

        if args.dry_run:
            check_md_suffix = f" check_md={output_check_md}" if output_check_md else ""
            check_command_suffix = (
                f" check_command_sh={args.output_check_command_sh}" if args.output_check_command_sh else ""
            )
            print(
                "blackwell_check_in_dry_run_ok "
                f"bundle_dir={bundle_dir} "
                f"expect_backend={args.expect_backend} "
                f"check_json={output_check_json} "
                f"{check_md_suffix} "
                f"{check_command_suffix} "
                f"require_device_substring={args.require_device_substring or '<none>'} "
                f"allow_sample_bundle={args.allow_sample_bundle}"
            )
            return

        selected_backend = run_bundle_check(
            bundle_dir=bundle_dir,
            expect_backend=args.expect_backend,
            check_in=True,
            require_blackwell=False,
            require_git_tracked=False,
            require_real_bundle=not args.allow_sample_bundle,
            require_device_substring=args.require_device_substring,
            output_check_json=output_check_json,
        )

        if output_check_md:
            _write_check_markdown(
                output_path=Path(output_check_md),
                bundle_dir=bundle_dir,
                expect_backend=args.expect_backend,
                selected_backend=selected_backend,
                check_json=output_check_json,
                require_device_substring=args.require_device_substring,
                allow_sample_bundle=args.allow_sample_bundle,
            )

        print(
            "blackwell_check_in_ok "
            f"selected={selected_backend} "
            f"bundle_dir={bundle_dir} "
            f"check_json={output_check_json}"
            + (f" check_md={output_check_md}" if output_check_md else "")
            + (f" check_command_sh={args.output_check_command_sh}" if args.output_check_command_sh else "")
        )
    except RuntimeError:
        if args.output_blocked_md:
            _write_blocked_markdown(
                output_path=Path(args.output_blocked_md),
                command=[*sys.argv],
                reason=str(sys.exc_info()[1]),
                bundle_dir_arg=args.bundle_dir,
                bundle_root_arg=args.bundle_root,
                expect_backend=args.expect_backend,
                require_device_substring=args.require_device_substring,
                preflight_mode=args.preflight,
                dry_run_mode=args.dry_run,
            )
        raise


if __name__ == "__main__":
    main()
