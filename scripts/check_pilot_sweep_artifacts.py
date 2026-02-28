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
from scripts.pilot_promote import _load_ranked_runs_with_source_hash


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate pilot sweep ranking/finalist artifacts")
    parser.add_argument(
        "--artifacts-dir",
        default="",
        help=(
            "artifact directory emitted by scripts.pilot_sweep/scripts.run_stage2_promotion_bundle, "
            "or 'auto' to discover latest real artifact bundle"
        ),
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts/pilot",
        help="artifact search root used when --artifacts-dir=auto",
    )
    parser.add_argument(
        "--ranked-json",
        default="",
        help=(
            "pilot sweep ranking JSON path from scripts.pilot_sweep --output-json, "
            "or filename relative to --artifacts-dir"
        ),
    )
    parser.add_argument(
        "--finalists-json",
        default="",
        help="finalists JSON from scripts.pilot_sweep --output-finalists-json or scripts.run_stage2_promotion_bundle",
    )
    parser.add_argument(
        "--finalists-md",
        default="",
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


def _classify_artifacts_dir(
    artifacts_dir: Path,
    ranked_json_name: str,
    finalists_json_name: str,
    finalists_md_name: str,
) -> str:
    if any("sample" in part.lower() for part in artifacts_dir.parts):
        return "sample path segment"

    missing_files = [
        name
        for name in (ranked_json_name, finalists_json_name, finalists_md_name)
        if not (artifacts_dir / name).is_file()
    ]
    if missing_files:
        return f"missing files: {', '.join(missing_files)}"

    return "real"


def _render_no_real_bundle_error(
    *,
    artifacts_root: Path,
    ranked_json_name: str,
    rejected_dirs: list[tuple[Path, str]],
) -> str:
    lines = [
        (
            f"no real pilot artifact bundle found under {artifacts_root}; "
            "run scripts.pilot_sweep on target GPU(s) first"
        ),
        (
            "discovery searched for "
            f"'{ranked_json_name}' files and rejected {len(rejected_dirs)} candidate bundle(s)"
        ),
    ]
    for rejected_path, reason in rejected_dirs[:5]:
        lines.append(f"- {rejected_path}: {reason}")
    if len(rejected_dirs) > 5:
        lines.append(f"- ... {len(rejected_dirs) - 5} more candidate bundle(s) omitted")
    return "\n".join(lines)


def _resolve_artifact_paths(
    *,
    artifacts_dir_arg: str,
    artifacts_root_arg: str,
    ranked_json_arg: str,
    finalists_json_arg: str,
    finalists_md_arg: str,
) -> tuple[Path, Path, Path]:
    default_ranked_name = "pilot_ranked_runs.json"
    default_finalists_json_name = "stage2_finalists.json"
    default_finalists_md_name = "stage2_finalists.md"

    if artifacts_dir_arg:
        ranked_json_name = ranked_json_arg or default_ranked_name
        finalists_json_name = finalists_json_arg or default_finalists_json_name
        finalists_md_name = finalists_md_arg or default_finalists_md_name

        if artifacts_dir_arg != "auto":
            artifacts_dir = Path(artifacts_dir_arg)
            return (
                artifacts_dir / ranked_json_name,
                artifacts_dir / finalists_json_name,
                artifacts_dir / finalists_md_name,
            )

        artifacts_root = Path(artifacts_root_arg)
        if not artifacts_root.is_dir():
            raise RuntimeError(
                f"artifacts_root does not exist: {artifacts_root}; pass --artifacts-dir explicitly or emit pilot artifacts first"
            )

        discovered_dirs = sorted({path.parent for path in artifacts_root.rglob(ranked_json_name)})
        candidates = []
        rejected_dirs = []
        for discovered_dir in discovered_dirs:
            classification = _classify_artifacts_dir(
                discovered_dir,
                ranked_json_name,
                finalists_json_name,
                finalists_md_name,
            )
            if classification == "real":
                candidates.append(discovered_dir)
            else:
                rejected_dirs.append((discovered_dir, classification))

        if not candidates:
            raise RuntimeError(
                _render_no_real_bundle_error(
                    artifacts_root=artifacts_root,
                    ranked_json_name=ranked_json_name,
                    rejected_dirs=rejected_dirs,
                )
            )

        candidates.sort(key=lambda path: (path / ranked_json_name).stat().st_mtime, reverse=True)
        latest_dir = candidates[0]
        return (
            latest_dir / ranked_json_name,
            latest_dir / finalists_json_name,
            latest_dir / finalists_md_name,
        )

    missing_args = [
        name
        for name, value in (
            ("--ranked-json", ranked_json_arg),
            ("--finalists-json", finalists_json_arg),
            ("--finalists-md", finalists_md_arg),
        )
        if not value
    ]
    if missing_args:
        raise RuntimeError(
            "missing required artifact paths: "
            f"{', '.join(missing_args)} (or pass --artifacts-dir/--artifacts-dir auto)"
        )

    return Path(ranked_json_arg), Path(finalists_json_arg), Path(finalists_md_arg)


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
    source_sha256 = payload.get("source_sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise RuntimeError(f"finalists JSON missing source_sha256 digest: {path}")
    if any(ch not in "0123456789abcdef" for ch in source_sha256):
        raise RuntimeError(f"finalists JSON source_sha256 must be lowercase hex: {path}")
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


def _assert_finalists_source_hash_matches_ranking(
    *,
    finalists_payload: dict[str, object],
    finalists_json_path: Path,
    ranked_source_sha256: str,
) -> None:
    source_sha256 = finalists_payload.get("source_sha256")
    if not isinstance(source_sha256, str) or not source_sha256:
        raise RuntimeError(f"finalists JSON missing source_sha256 digest: {finalists_json_path}")
    if source_sha256 != ranked_source_sha256:
        raise RuntimeError(
            "finalists JSON source_sha256 does not match --ranked-json contents: "
            f"source_sha256={source_sha256} ranked_sha256={ranked_source_sha256}"
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


def run_pilot_bundle_check(
    *,
    ranked_json_path: Path,
    finalists_json_path: Path,
    finalists_md_path: Path,
    require_real_input: bool,
    require_git_tracked: bool,
    check_in: bool,
    allow_sample_input_in_check_in: bool = False,
    output_check_json: str = "",
) -> int:
    effective_require_real_input = require_real_input or (check_in and not allow_sample_input_in_check_in)
    effective_require_git_tracked = require_git_tracked or check_in

    paths = {
        "ranked_json": ranked_json_path,
        "finalists_json": finalists_json_path,
        "finalists_md": finalists_md_path,
    }
    _assert_files_exist(paths)
    ranked_runs, ranked_source_sha256 = _load_ranked_runs_with_source_hash(
        str(paths["ranked_json"]),
        require_real_input=effective_require_real_input,
    )
    if effective_require_git_tracked:
        _assert_git_tracked(paths)

    finalists_payload = _load_finalists_payload(paths["finalists_json"])
    _assert_finalists_source_matches_ranking(
        finalists_payload=finalists_payload,
        finalists_json_path=paths["finalists_json"],
        ranked_json_path=paths["ranked_json"],
    )
    _assert_finalists_source_hash_matches_ranking(
        finalists_payload=finalists_payload,
        finalists_json_path=paths["finalists_json"],
        ranked_source_sha256=ranked_source_sha256,
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

    if output_check_json:
        _write_check_receipt(
            path=Path(output_check_json),
            ranked_json_path=paths["ranked_json"],
            finalists_json_path=paths["finalists_json"],
            finalists_md_path=paths["finalists_md"],
            finalists_count=len(expected_finalists),
            require_real_input=effective_require_real_input,
            require_git_tracked=effective_require_git_tracked,
            check_in=check_in,
        )
    return len(expected_finalists)


def main() -> None:
    args = _parse_args()
    ranked_json_path, finalists_json_path, finalists_md_path = _resolve_artifact_paths(
        artifacts_dir_arg=args.artifacts_dir,
        artifacts_root_arg=args.artifacts_root,
        ranked_json_arg=args.ranked_json,
        finalists_json_arg=args.finalists_json,
        finalists_md_arg=args.finalists_md,
    )
    finalists_count = run_pilot_bundle_check(
        ranked_json_path=ranked_json_path,
        finalists_json_path=finalists_json_path,
        finalists_md_path=finalists_md_path,
        require_real_input=args.require_real_input,
        require_git_tracked=args.require_git_tracked,
        check_in=args.check_in,
        output_check_json=args.output_check_json,
    )

    status_line = (
        "pilot_bundle_check_ok "
        f"finalists={finalists_count} "
        f"ranked_json={ranked_json_path}"
    )
    if args.output_check_json:
        status_line += f" check_json={args.output_check_json}"
    print(
        status_line
    )


if __name__ == "__main__":
    main()
