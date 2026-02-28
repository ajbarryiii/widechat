"""Run strict pilot artifact check-in validation in one command.

Example:
python -m scripts.run_pilot_check_in --artifacts-dir auto --artifacts-root artifacts/pilot
"""

import argparse
from pathlib import Path

from scripts.check_pilot_sweep_artifacts import run_pilot_bundle_check


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict check-in validation for pilot sweep artifacts")
    parser.add_argument(
        "--artifacts-dir",
        default="auto",
        help="artifact directory, or 'auto' to discover latest real artifact bundle",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts/pilot",
        help="artifact search root used when --artifacts-dir=auto",
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


def _is_real_artifacts_dir(
    artifacts_dir: Path,
    ranked_json: str,
    finalists_json: str,
    finalists_md: str,
) -> bool:
    if any("sample" in part for part in artifacts_dir.parts):
        return False
    return all((artifacts_dir / name).is_file() for name in (ranked_json, finalists_json, finalists_md))


def _resolve_artifacts_dir(
    artifacts_dir_arg: str,
    artifacts_root_arg: str,
    ranked_json: str,
    finalists_json: str,
    finalists_md: str,
) -> Path:
    if artifacts_dir_arg != "auto":
        return Path(artifacts_dir_arg)

    artifacts_root = Path(artifacts_root_arg)
    if not artifacts_root.is_dir():
        raise RuntimeError(
            f"artifacts_root does not exist: {artifacts_root}; pass --artifacts-dir explicitly or emit pilot artifacts first"
        )

    candidates = [
        path.parent
        for path in artifacts_root.rglob(ranked_json)
        if _is_real_artifacts_dir(path.parent, ranked_json, finalists_json, finalists_md)
    ]
    if not candidates:
        raise RuntimeError(
            f"no real pilot artifact bundle found under {artifacts_root}; run scripts.pilot_sweep on target GPU(s) first"
        )

    candidates.sort(key=lambda path: (path / ranked_json).stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    args = _parse_args()
    artifacts_dir = _resolve_artifacts_dir(
        args.artifacts_dir,
        args.artifacts_root,
        args.ranked_json,
        args.finalists_json,
        args.finalists_md,
    )
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
