"""Run strict pilot artifact check-in validation in one command.

Example:
python -m scripts.run_pilot_check_in --artifacts-dir auto --artifacts-root artifacts/pilot
"""

import argparse
import json
import subprocess
import sys
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
        "--output-check-md",
        default="",
        help="optional path for markdown check-in evidence summary",
    )
    parser.add_argument(
        "--bundle-json",
        default="",
        help=(
            "optional promotion bundle receipt JSON path from "
            "scripts.run_stage2_promotion_bundle --output-bundle-json; "
            "accepts relative filename (with --artifacts-dir) or 'auto'"
        ),
    )
    parser.add_argument(
        "--bundle-json-name",
        default="stage2_promotion_bundle.json",
        help="bundle receipt filename to use when --bundle-json=auto",
    )
    parser.add_argument(
        "--allow-sample-input",
        action="store_true",
        help="allow sample/fixture ranked-run inputs (for local regression checks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only resolve paths and print planned checker invocation",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate required artifact/checker inputs before strict check-in",
    )
    parser.add_argument(
        "--output-preflight-json",
        default="",
        help="optional path to write machine-readable preflight receipt JSON",
    )
    return parser.parse_args()


def _write_check_markdown(
    *,
    output_path: Path,
    artifacts_dir: Path,
    ranked_json: Path,
    finalists_json: Path,
    finalists_md: Path,
    check_json: str,
    finalists_count: int,
    allow_sample_input: bool,
    bundle_json_path: Path | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pilot Artifact Strict Check-In Evidence",
        "",
        f"- finalists: `{finalists_count}`",
        f"- artifacts_dir: `{artifacts_dir}`",
        f"- ranked_json: `{ranked_json}`",
        f"- finalists_json: `{finalists_json}`",
        f"- finalists_md: `{finalists_md}`",
        f"- check_json: `{check_json}`",
        f"- require_real_input: `{str(not allow_sample_input).lower()}`",
        f"- check_in_mode: `true`",
    ]
    if bundle_json_path is not None:
        lines.append(f"- bundle_json: `{bundle_json_path}`")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _classify_artifacts_dir(
    artifacts_dir: Path,
    ranked_json: str,
    finalists_json: str,
    finalists_md: str,
) -> str:
    if any("sample" in part.lower() for part in artifacts_dir.parts):
        return "sample path segment"

    missing_files = [name for name in (ranked_json, finalists_json, finalists_md) if not (artifacts_dir / name).is_file()]
    if missing_files:
        return f"missing files: {', '.join(missing_files)}"

    ranked_json_path = artifacts_dir / ranked_json
    try:
        ranked_payload = json.loads(ranked_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"malformed ranked JSON ({exc.msg})"

    if not isinstance(ranked_payload, dict):
        return "ranked JSON payload must be an object"
    if ranked_payload.get("is_sample") is True:
        return "ranked JSON payload marks is_sample=true"
    if not isinstance(ranked_payload.get("ranked_runs"), list):
        return "ranked JSON missing ranked_runs list"
    return "real"


def _render_no_real_bundle_error(
    *,
    artifacts_root: Path,
    ranked_json: str,
    rejected_dirs: list[tuple[Path, str]],
) -> str:
    lines = [
        f"no real pilot artifact bundle found under {artifacts_root}; run scripts.pilot_sweep on target GPU(s) first",
        f"discovery searched for '{ranked_json}' files and rejected {len(rejected_dirs)} candidate bundle(s)",
    ]
    for rejected_path, reason in rejected_dirs[:5]:
        lines.append(f"- {rejected_path}: {reason}")
    if len(rejected_dirs) > 5:
        lines.append(f"- ... {len(rejected_dirs) - 5} more candidate bundle(s) omitted")
    return "\n".join(lines)


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

    discovered_dirs = {
        path.parent
        for path in artifacts_root.rglob(ranked_json)
    }
    discovered_dirs.update(path.parent for path in artifacts_root.rglob("*runbook*.md"))

    candidates = []
    rejected_dirs = []
    for discovered_dir in sorted(discovered_dirs):
        classification = _classify_artifacts_dir(
            discovered_dir,
            ranked_json,
            finalists_json,
            finalists_md,
        )
        if classification == "real":
            candidates.append(discovered_dir)
        else:
            rejected_dirs.append((discovered_dir, classification))

    if not candidates:
        raise RuntimeError(
            _render_no_real_bundle_error(
                artifacts_root=artifacts_root,
                ranked_json=ranked_json,
                rejected_dirs=rejected_dirs,
            )
        )

    candidates.sort(key=lambda path: (path / ranked_json).stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_bundle_json_path(
    *,
    bundle_json_arg: str,
    bundle_json_name: str,
    artifacts_dir: Path,
) -> Path | None:
    if not bundle_json_arg:
        return None

    if bundle_json_arg == "auto":
        return artifacts_dir / bundle_json_name

    bundle_json = Path(bundle_json_arg)
    if bundle_json.is_absolute():
        return bundle_json
    return artifacts_dir / bundle_json


def _is_git_tracked(path: Path) -> bool:
    try:
        rel_path = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return False

    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(rel_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _run_preflight(
    *,
    artifacts_dir: Path,
    ranked_json: Path,
    finalists_json: Path,
    finalists_md: Path,
    bundle_json_path: Path | None,
    output_check_json: str,
    output_preflight_json: str,
    allow_sample_input: bool,
) -> None:
    missing = [
        str(path)
        for path in (ranked_json, finalists_json, finalists_md)
        if not path.is_file()
    ]
    if bundle_json_path is not None and not bundle_json_path.is_file():
        missing.append(str(bundle_json_path))

    if missing:
        raise RuntimeError(
            "pilot_check_in_preflight failed: missing required file(s): " + ", ".join(missing)
        )

    git_tracked_paths = {
        str(path): _is_git_tracked(path)
        for path in (ranked_json, finalists_json, finalists_md)
    }
    if bundle_json_path is not None:
        git_tracked_paths[str(bundle_json_path)] = _is_git_tracked(bundle_json_path)

    if output_preflight_json:
        preflight_path = Path(output_preflight_json)
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "ok",
            "command": [*sys.argv],
            "artifacts_dir": str(artifacts_dir),
            "ranked_json": str(ranked_json),
            "finalists_json": str(finalists_json),
            "finalists_md": str(finalists_md),
            "check_json": output_check_json,
            "allow_sample_input": allow_sample_input,
            "git_tracked": git_tracked_paths,
        }
        if bundle_json_path is not None:
            payload["bundle_json"] = str(bundle_json_path)
        preflight_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        "pilot_check_in_preflight_ok "
        f"artifacts_dir={artifacts_dir} "
        f"ranked_json={ranked_json} "
        f"finalists_json={finalists_json} "
        f"finalists_md={finalists_md} "
        f"check_json={output_check_json} "
        f"allow_sample_input={allow_sample_input} "
        f"git_tracked={json.dumps(git_tracked_paths, sort_keys=True)}"
        + (f" bundle_json={bundle_json_path}" if bundle_json_path is not None else "")
        + (f" preflight_json={output_preflight_json}" if output_preflight_json else "")
    )


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
    output_check_md = args.output_check_md
    bundle_json_path = _resolve_bundle_json_path(
        bundle_json_arg=args.bundle_json,
        bundle_json_name=args.bundle_json_name,
        artifacts_dir=artifacts_dir,
    )

    if args.dry_run:
        print(
            "pilot_check_in_dry_run_ok "
            f"artifacts_dir={artifacts_dir} "
            f"ranked_json={ranked_json} "
            f"finalists_json={finalists_json} "
            f"finalists_md={finalists_md} "
            f"check_json={output_check_json} "
            f"allow_sample_input={args.allow_sample_input}"
            + (f" check_md={output_check_md}" if output_check_md else "")
            + (f" bundle_json={bundle_json_path}" if bundle_json_path is not None else "")
        )
        return

    if args.preflight:
        _run_preflight(
            artifacts_dir=artifacts_dir,
            ranked_json=ranked_json,
            finalists_json=finalists_json,
            finalists_md=finalists_md,
            bundle_json_path=bundle_json_path,
            output_check_json=output_check_json,
            output_preflight_json=args.output_preflight_json,
            allow_sample_input=args.allow_sample_input,
        )
        return

    finalists_count = run_pilot_bundle_check(
        ranked_json_path=ranked_json,
        finalists_json_path=finalists_json,
        finalists_md_path=finalists_md,
        require_real_input=not args.allow_sample_input,
        require_git_tracked=False,
        check_in=True,
        allow_sample_input_in_check_in=args.allow_sample_input,
        output_check_json=output_check_json,
        bundle_json_path=bundle_json_path,
    )

    if output_check_md:
        _write_check_markdown(
            output_path=Path(output_check_md),
            artifacts_dir=artifacts_dir,
            ranked_json=ranked_json,
            finalists_json=finalists_json,
            finalists_md=finalists_md,
            check_json=output_check_json,
            finalists_count=finalists_count,
            allow_sample_input=args.allow_sample_input,
            bundle_json_path=bundle_json_path,
        )

    print(
        "pilot_check_in_ok "
        f"finalists={finalists_count} "
        f"artifacts_dir={artifacts_dir} "
        f"check_json={output_check_json}"
        + (f" check_md={output_check_md}" if output_check_md else "")
        + (f" bundle_json={bundle_json_path}" if bundle_json_path is not None else "")
    )


if __name__ == "__main__":
    main()
