"""Offline validation for Stage 1 pilot sweep ranking/finalist artifacts.

Example:
python -m scripts.check_pilot_sweep_artifacts \
  --ranked-json artifacts/pilot/pilot_ranked_runs.json \
  --finalists-json artifacts/pilot/stage2_finalists.json \
  --finalists-md artifacts/pilot/stage2_finalists.md
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from nanochat.pilot_sweep import select_finalists
from scripts.pilot_promote import _load_ranked_runs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate pilot sweep ranking/finalist artifacts")
    parser.add_argument("--ranked-json", required=True, help="pilot sweep ranking JSON from scripts.pilot_sweep --output-json")
    parser.add_argument(
        "--finalists-json",
        required=True,
        help="finalists JSON from scripts.pilot_sweep --output-finalists-json or scripts.run_stage2_promotion_bundle",
    )
    parser.add_argument(
        "--finalists-md",
        required=True,
        help="finalists markdown from scripts.pilot_sweep --output-finalists-md or scripts.run_stage2_promotion_bundle",
    )
    parser.add_argument(
        "--require-real-input",
        action="store_true",
        help="reject sample/fixture ranked-run JSON inputs",
    )
    parser.add_argument(
        "--require-git-tracked",
        action="store_true",
        help="require ranking/finalists artifacts to be tracked by git",
    )
    parser.add_argument(
        "--check-in",
        action="store_true",
        help="enable strict check-in mode (requires real input + git-tracked artifacts)",
    )
    parser.add_argument(
        "--output-check-json",
        default="",
        help="optional path to write machine-readable checker receipt JSON",
    )
    return parser.parse_args()


def _assert_files_exist(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing {label} file: {path}")


def _assert_git_tracked(paths: dict[str, Path]) -> None:
    repo_root = Path.cwd().resolve()
    for path in paths.values():
        resolved_path = path.resolve()
        try:
            rel_path = resolved_path.relative_to(repo_root)
        except ValueError as exc:
            raise RuntimeError(f"artifact path is outside repository root: {path}") from exc
        cmd = ["git", "ls-files", "--error-unmatch", str(rel_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"artifact is not git-tracked: {path}")


def _load_finalists_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"finalists JSON must be an object: {path}")
    source = payload.get("source")
    if not isinstance(source, str) or not source:
        raise RuntimeError(f"finalists JSON missing non-empty source path: {path}")
    max_finalists = payload.get("max_finalists")
    if isinstance(max_finalists, bool) or not isinstance(max_finalists, int) or max_finalists <= 0:
        raise RuntimeError(f"finalists JSON missing positive integer max_finalists: {path}")
    selected_finalists = payload.get("selected_finalists")
    if not isinstance(selected_finalists, list):
        raise RuntimeError(f"finalists JSON missing selected_finalists list: {path}")
    for index, row in enumerate(selected_finalists):
        if not isinstance(row, dict):
            raise RuntimeError(f"selected_finalists[{index}] must be an object: {path}")
    return payload


def _assert_finalists_source_matches_ranking(
    *,
    finalists_payload: dict[str, object],
    finalists_json_path: Path,
    ranked_json_path: Path,
) -> None:
    source = finalists_payload["source"]
    if not isinstance(source, str) or not source:
        raise RuntimeError(f"finalists JSON missing non-empty source path: {finalists_json_path}")
    resolved_source = Path(source).resolve()
    resolved_ranked = ranked_json_path.resolve()
    if resolved_source != resolved_ranked:
        raise RuntimeError(
            "finalists JSON source does not match --ranked-json: "
            f"source={source} ranked_json={ranked_json_path}"
        )


def _assert_finalists_match_ranking(
    *,
    ranked_runs: list[dict[str, int | float | bool | str | None]],
    finalists_payload: dict[str, object],
    finalists_json_path: Path,
) -> list[dict[str, int | float | bool | str | None]]:
    max_finalists_obj = finalists_payload["max_finalists"]
    if isinstance(max_finalists_obj, bool) or not isinstance(max_finalists_obj, int):
        raise RuntimeError(
            "finalists JSON missing positive integer max_finalists: "
            f"{finalists_json_path}"
        )
    max_finalists = max_finalists_obj
    expected = select_finalists(ranked_runs, max_finalists=max_finalists)
    actual = finalists_payload["selected_finalists"]
    if expected != actual:
        raise RuntimeError(
            "selected_finalists does not match ranked_runs + max_finalists in "
            f"{finalists_json_path}"
        )
    return expected


def _assert_finalists_markdown(
    *,
    finalists_md_path: Path,
    expected_finalists: list[dict[str, int | float | bool | str | None]],
) -> None:
    markdown = finalists_md_path.read_text(encoding="utf-8")
    required_snippets = ["## Stage 2 Finalists", "## Stage 2 depth/branch flags"]
    for snippet in required_snippets:
        if snippet not in markdown:
            raise RuntimeError(f"finalists markdown missing snippet: {snippet}")

    for row in expected_finalists:
        expected_flag = (
            f"`--depth {row['depth']} --n-branches {row['n_branches']} --aspect-ratio {row['aspect_ratio']}`"
        )
        if expected_flag not in markdown:
            raise RuntimeError(
                "finalists markdown missing depth/branch flag line for "
                f"{row['config']}: {finalists_md_path}"
            )


def _write_check_receipt(
    *,
    path: Path,
    ranked_json_path: Path,
    finalists_json_path: Path,
    finalists_md_path: Path,
    finalists_count: int,
    require_real_input: bool,
    require_git_tracked: bool,
    check_in: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok",
        "command": [*sys.argv],
        "ranked_json": str(ranked_json_path),
        "finalists_json": str(finalists_json_path),
        "finalists_md": str(finalists_md_path),
        "finalists_count": finalists_count,
        "require_real_input": require_real_input,
        "require_git_tracked": require_git_tracked,
        "check_in": check_in,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    require_real_input = args.require_real_input or args.check_in
    require_git_tracked = args.require_git_tracked or args.check_in

    paths = {
        "ranked_json": Path(args.ranked_json),
        "finalists_json": Path(args.finalists_json),
        "finalists_md": Path(args.finalists_md),
    }
    _assert_files_exist(paths)
    ranked_runs = _load_ranked_runs(str(paths["ranked_json"]), require_real_input=require_real_input)
    if require_git_tracked:
        _assert_git_tracked(paths)

    finalists_payload = _load_finalists_payload(paths["finalists_json"])
    _assert_finalists_source_matches_ranking(
        finalists_payload=finalists_payload,
        finalists_json_path=paths["finalists_json"],
        ranked_json_path=paths["ranked_json"],
    )
    expected_finalists = _assert_finalists_match_ranking(
        ranked_runs=ranked_runs,
        finalists_payload=finalists_payload,
        finalists_json_path=paths["finalists_json"],
    )
    _assert_finalists_markdown(
        finalists_md_path=paths["finalists_md"],
        expected_finalists=expected_finalists,
    )

    if args.output_check_json:
        _write_check_receipt(
            path=Path(args.output_check_json),
            ranked_json_path=paths["ranked_json"],
            finalists_json_path=paths["finalists_json"],
            finalists_md_path=paths["finalists_md"],
            finalists_count=len(expected_finalists),
            require_real_input=require_real_input,
            require_git_tracked=require_git_tracked,
            check_in=args.check_in,
        )

    status_line = (
        "pilot_bundle_check_ok "
        f"finalists={len(expected_finalists)} "
        f"ranked_json={paths['ranked_json']}"
    )
    if args.output_check_json:
        status_line += f" check_json={args.output_check_json}"
    print(
        status_line
    )


if __name__ == "__main__":
    main()
