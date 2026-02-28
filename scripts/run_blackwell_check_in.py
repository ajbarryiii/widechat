"""Run strict Blackwell evidence check-in validation in one command.

Example:
python -m scripts.run_blackwell_check_in --bundle-dir artifacts/blackwell_smoke
"""

import argparse
from pathlib import Path

from scripts.check_blackwell_evidence_bundle import _resolve_bundle_dir as _resolve_bundle_dir_from_checker
from scripts.check_blackwell_evidence_bundle import run_bundle_check


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
        "--allow-sample-bundle",
        action="store_true",
        help="allow sample fixture bundles (for local regression checks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only resolve paths and print planned checker invocation",
    )
    return parser.parse_args()


def _resolve_bundle_dir(bundle_dir_arg: str, bundle_root_arg: str) -> Path:
    return _resolve_bundle_dir_from_checker(bundle_dir_arg, bundle_root_arg)


def main() -> None:
    args = _parse_args()
    bundle_dir = _resolve_bundle_dir(args.bundle_dir, args.bundle_root)
    output_check_json = args.output_check_json or str(bundle_dir / "blackwell_bundle_check.json")

    if args.dry_run:
        print(
            "blackwell_check_in_dry_run_ok "
            f"bundle_dir={bundle_dir} "
            f"expect_backend={args.expect_backend} "
            f"check_json={output_check_json} "
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
    print(
        "blackwell_check_in_ok "
        f"selected={selected_backend} "
        f"bundle_dir={bundle_dir} "
        f"check_json={output_check_json}"
    )


if __name__ == "__main__":
    main()
