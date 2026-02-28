"""Run strict Blackwell evidence check-in validation in one command.

Example:
python -m scripts.run_blackwell_check_in --bundle-dir artifacts/blackwell_smoke
"""

import argparse
from pathlib import Path

from scripts.check_blackwell_evidence_bundle import run_bundle_check


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict check-in validation for Blackwell smoke bundle")
    parser.add_argument(
        "--bundle-dir",
        default="artifacts/blackwell_smoke",
        help="directory emitted by scripts.run_blackwell_smoke_bundle",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bundle_dir = Path(args.bundle_dir)
    output_check_json = args.output_check_json or str(bundle_dir / "blackwell_bundle_check.json")

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
