"""Run strict pilot artifact check-in validation in one command.

Example:
python -m scripts.run_pilot_check_in --artifacts-dir artifacts/pilot
"""

import argparse
from pathlib import Path

from scripts.check_pilot_sweep_artifacts import run_pilot_bundle_check


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict check-in validation for pilot sweep artifacts")
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/pilot",
        help="directory containing pilot_ranked_runs.json and stage2 finalists artifacts",
    )
    parser.add_argument(
        "--ranked-json",
        default="pilot_ranked_runs.json",
        help="ranked-runs JSON filename relative to --artifacts-dir",
    )
    parser.add_argument(
        "--finalists-json",
        default="stage2_finalists.json",
        help="finalists JSON filename relative to --artifacts-dir",
    )
    parser.add_argument(
        "--finalists-md",
        default="stage2_finalists.md",
        help="finalists markdown filename relative to --artifacts-dir",
    )
    parser.add_argument(
        "--output-check-json",
        default="",
        help="optional path for checker receipt (defaults to <artifacts-dir>/pilot_bundle_check.json)",
    )
    parser.add_argument(
        "--allow-sample-input",
        action="store_true",
        help="allow sample/fixture ranked-run inputs (for local regression checks)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    ranked_json = artifacts_dir / args.ranked_json
    finalists_json = artifacts_dir / args.finalists_json
    finalists_md = artifacts_dir / args.finalists_md
    output_check_json = args.output_check_json or str(artifacts_dir / "pilot_bundle_check.json")

    finalists_count = run_pilot_bundle_check(
        ranked_json_path=ranked_json,
        finalists_json_path=finalists_json,
        finalists_md_path=finalists_md,
        require_real_input=not args.allow_sample_input,
        require_git_tracked=False,
        check_in=True,
        allow_sample_input_in_check_in=args.allow_sample_input,
        output_check_json=output_check_json,
    )

    print(
        "pilot_check_in_ok "
        f"finalists={finalists_count} "
        f"artifacts_dir={artifacts_dir} "
        f"check_json={output_check_json}"
    )


if __name__ == "__main__":
    main()
