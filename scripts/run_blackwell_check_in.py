"""Run strict Blackwell evidence check-in validation in one command.

Example:
python -m scripts.run_blackwell_check_in --bundle-dir artifacts/blackwell_smoke
"""

import argparse
from pathlib import Path

from scripts.check_blackwell_evidence_bundle import run_bundle_check

_REQUIRED_BUNDLE_FILES = (
    "flash_backend_smoke.json",
    "flash_backend_status.log",
    "blackwell_smoke_evidence.md",
    "blackwell_smoke_runbook.md",
)


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


def _is_real_bundle_dir(bundle_dir: Path) -> bool:
    if "sample_bundle" in bundle_dir.parts:
        return False
    return all((bundle_dir / filename).is_file() for filename in _REQUIRED_BUNDLE_FILES)


def _resolve_bundle_dir(bundle_dir_arg: str, bundle_root_arg: str) -> Path:
    if bundle_dir_arg != "auto":
        return Path(bundle_dir_arg)

    bundle_root = Path(bundle_root_arg)
    if not bundle_root.is_dir():
        raise RuntimeError(
            f"bundle_root does not exist: {bundle_root}; pass --bundle-dir explicitly or emit a bundle first"
        )

    candidates = [path for path in bundle_root.rglob("flash_backend_smoke.json") if _is_real_bundle_dir(path.parent)]
    if not candidates:
        raise RuntimeError(
            f"no real Blackwell bundle found under {bundle_root}; run scripts.run_blackwell_smoke_bundle on RTX 5090 first"
        )

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0].parent


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
