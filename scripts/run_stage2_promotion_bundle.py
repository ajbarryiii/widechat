"""Run Stage 2 promotion and artifact emission in one command.

Example:
python -m scripts.run_stage2_promotion_bundle --input-json artifacts/pilot/sample_ranked_runs.json --output-dir artifacts/pilot
"""

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

from nanochat.pilot_sweep import format_finalists_summary, select_finalists
from scripts.check_pilot_sweep_artifacts import run_pilot_bundle_check
from scripts.pilot_promote import _load_ranked_runs_with_source_hash, _validate_stage2_finalists


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 promotion artifact bundle")
    parser.add_argument(
        "--input-json",
        default="auto",
        help="pilot ranking JSON produced by scripts.pilot_sweep --output-json, or 'auto' to discover latest real bundle",
    )
    parser.add_argument(
        "--input-root",
        default="artifacts/pilot",
        help="artifact search root used when --input-json=auto",
    )
    parser.add_argument(
        "--input-json-name",
        default="pilot_ranked_runs.json",
        help="ranked-runs JSON filename used when --input-json=auto",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/pilot",
        help="artifact directory for stage2 finalists JSON/markdown",
    )
    parser.add_argument("--min-finalists", type=int, default=2, help="minimum number of qualified finalists required")
    parser.add_argument("--max-finalists", type=int, default=3, help="max number of qualified finalists to keep")
    parser.add_argument(
        "--output-json",
        default="",
        help="optional finalists JSON path (defaults to <output-dir>/stage2_finalists.json)",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="optional finalists markdown path (defaults to <output-dir>/stage2_finalists.md)",
    )
    parser.add_argument(
        "--require-real-input",
        action="store_true",
        help="reject sample/fixture ranked-run JSON inputs",
    )
    parser.add_argument(
        "--output-runbook-md",
        default="",
        help="optional markdown path to write a promotion/check-in runbook",
    )
    parser.add_argument(
        "--run-check-in",
        action="store_true",
        help="run strict offline check-in validation after writing finalists artifacts",
    )
    parser.add_argument(
        "--output-check-json",
        default="",
        help="optional checker receipt path (defaults to <output-dir>/pilot_bundle_check.json when --run-check-in is set)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only resolve inputs/outputs and print planned bundle execution",
    )
    parser.add_argument(
        "--output-bundle-json",
        default="",
        help="optional path to write machine-readable promotion-bundle receipt JSON",
    )
    return parser.parse_args()


def _resolve_output_paths(output_dir: str, output_json: str, output_md: str) -> tuple[Path, Path]:
    base = Path(output_dir)
    finalists_json = Path(output_json) if output_json else base / "stage2_finalists.json"
    finalists_md = Path(output_md) if output_md else base / "stage2_finalists.md"
    return finalists_json, finalists_md


def _classify_input_json_candidate(path: Path) -> str:
    if any("sample" in part.lower() for part in path.parts):
        return "sample path segment"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"unreadable JSON: {exc}"

    if not isinstance(payload, dict):
        return "input JSON must be an object"
    if payload.get("is_sample") is True:
        return "payload marked is_sample=true"

    ranked_runs = payload.get("ranked_runs")
    if not isinstance(ranked_runs, list) or not ranked_runs:
        return "missing non-empty ranked_runs list"

    return "real"


def _render_no_real_input_json_error(
    *,
    input_root: Path,
    input_json_name: str,
    rejected_paths: list[tuple[Path, str]],
) -> str:
    lines = [
        f"no real pilot ranking JSON found under {input_root}; run scripts.pilot_sweep on target GPU(s) first",
        f"discovery searched for '{input_json_name}' files and rejected {len(rejected_paths)} candidate file(s)",
    ]
    for rejected_path, reason in rejected_paths[:5]:
        lines.append(f"- {rejected_path}: {reason}")
    if len(rejected_paths) > 5:
        lines.append(f"- ... {len(rejected_paths) - 5} more candidate file(s) omitted")
    return "\n".join(lines)


def _resolve_input_json(input_json_arg: str, input_root_arg: str, input_json_name: str) -> Path:
    if input_json_arg != "auto":
        return Path(input_json_arg)

    input_root = Path(input_root_arg)
    if not input_root.is_dir():
        raise RuntimeError(
            f"input_root does not exist: {input_root}; pass --input-json explicitly or emit pilot artifacts first"
        )

    discovered_paths = sorted(path for path in input_root.rglob(input_json_name) if path.is_file())
    real_candidates = []
    rejected_paths: list[tuple[Path, str]] = []
    for path in discovered_paths:
        classification = _classify_input_json_candidate(path)
        if classification == "real":
            real_candidates.append(path)
        else:
            rejected_paths.append((path, classification))

    if not real_candidates:
        raise RuntimeError(
            _render_no_real_input_json_error(
                input_root=input_root,
                input_json_name=input_json_name,
                rejected_paths=rejected_paths,
            )
        )

    real_candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return real_candidates[0]


def _write_finalists_json(
    path: Path,
    source: str,
    source_sha256: str,
    max_finalists: int,
    finalists: list[dict[str, int | float | bool | str | None]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "source_sha256": source_sha256,
        "max_finalists": max_finalists,
        "selected_finalists": finalists,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_finalists_md(
    path: Path,
    finalists_summary: str,
    finalists: list[dict[str, int | float | bool | str | None]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "## Stage 2 Finalists",
        "",
        finalists_summary,
        "",
        "## Stage 2 depth/branch flags",
        "",
    ]
    for row in finalists:
        lines.append(
            f"- `{row['config']}`: `--depth {row['depth']} --n-branches {row['n_branches']} --aspect-ratio {row['aspect_ratio']}`"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_runbook_md(
    *,
    path: Path,
    input_json: str,
    output_dir: str,
    finalists_json: Path,
    finalists_md: Path,
    min_finalists: int,
    max_finalists: int,
    require_real_input: bool,
    run_check_in: bool,
    output_check_json: str,
    output_bundle_json: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked_json_path = Path(input_json).resolve()
    finalists_json_path = finalists_json.resolve()
    finalists_md_path = finalists_md.resolve()

    quoted_input_json = shlex.quote(input_json)
    quoted_output_dir = shlex.quote(output_dir)
    quoted_ranked_json = shlex.quote(str(ranked_json_path))
    quoted_finalists_json = shlex.quote(str(finalists_json_path))
    quoted_finalists_md = shlex.quote(str(finalists_md_path))
    check_json_path = output_check_json or str(Path(output_dir) / "pilot_bundle_check.json")
    quoted_check_json = shlex.quote(check_json_path)
    quoted_output_bundle_json = shlex.quote(output_bundle_json) if output_bundle_json else ""
    command_lines = [
        "python -m scripts.run_stage2_promotion_bundle \\",
        f"  --input-json {quoted_input_json} \\",
        f"  --output-dir {quoted_output_dir} \\",
        f"  --min-finalists {min_finalists} \\",
        f"  --max-finalists {max_finalists}",
    ]
    if require_real_input:
        command_lines[-1] += " \\"
        command_lines.append("  --require-real-input")
    if run_check_in:
        command_lines[-1] += " \\"
        command_lines.append("  --run-check-in")
        if output_check_json:
            command_lines[-1] += " \\"
            command_lines.append(f"  --output-check-json {quoted_check_json}")
    if output_bundle_json:
        command_lines[-1] += " \\"
        command_lines.append(f"  --output-bundle-json {quoted_output_bundle_json}")

    check_in_lines = [
        "python -m scripts.run_pilot_check_in \\",
        f"  --artifacts-dir {quoted_output_dir} \\",
        f"  --ranked-json {quoted_ranked_json} \\",
        f"  --finalists-json {quoted_finalists_json} \\",
        f"  --finalists-md {quoted_finalists_md} \\",
        f"  --output-check-json {quoted_check_json}",
    ]
    if output_bundle_json:
        check_in_lines[-1] += " \\"
        check_in_lines.append(f"  --bundle-json {quoted_output_bundle_json}")

    lines = [
        "# Stage 2 Promotion Bundle Runbook",
        "",
        "## Command",
        "```bash",
        *command_lines,
        "```",
        "",
        "## Expected outputs",
        f"- `{finalists_json}`",
        f"- `{finalists_md}`",
        "",
        "## Check-in command",
        "```bash",
        *check_in_lines,
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_bundle_receipt(
    *,
    path: Path,
    input_json: Path,
    finalists_json: Path,
    finalists_md: Path,
    finalists_count: int,
    source_sha256: str,
    run_check_in: bool,
    check_json_path: Path | None,
) -> None:
    def _sha256(path_obj: Path) -> str:
        return hashlib.sha256(path_obj.read_bytes()).hexdigest()

    artifact_sha256 = {
        "finalists_json": _sha256(finalists_json),
        "finalists_md": _sha256(finalists_md),
    }
    if check_json_path is not None and check_json_path.is_file():
        artifact_sha256["check_json"] = _sha256(check_json_path)

    payload = {
        "status": "ok",
        "command": [*sys.argv],
        "input_json": str(input_json),
        "source_sha256": source_sha256,
        "finalists_json": str(finalists_json),
        "finalists_md": str(finalists_md),
        "finalists_count": finalists_count,
        "run_check_in": run_check_in,
        "check_json": str(check_json_path) if check_json_path is not None else None,
        "artifact_sha256": artifact_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    input_json = _resolve_input_json(args.input_json, args.input_root, args.input_json_name)
    finalists_json, finalists_md = _resolve_output_paths(args.output_dir, args.output_json, args.output_md)
    runbook_md = Path(args.output_runbook_md) if args.output_runbook_md else None
    bundle_json_path = Path(args.output_bundle_json) if args.output_bundle_json else None
    check_json_path: Path | None = None
    if args.run_check_in:
        check_json_path = Path(args.output_check_json) if args.output_check_json else Path(args.output_dir) / "pilot_bundle_check.json"

    if args.dry_run:
        print(
            "stage2_promotion_bundle_dry_run_ok "
            f"input_json={input_json} "
            f"json={finalists_json} "
            f"md={finalists_md} "
            f"run_check_in={args.run_check_in} "
            f"require_real_input={args.require_real_input}"
            + (f" check_json={check_json_path}" if check_json_path is not None else "")
            + (f" runbook_md={runbook_md}" if runbook_md is not None else "")
            + (f" bundle_json={bundle_json_path}" if bundle_json_path is not None else "")
        )
        return

    ranked_runs, source_sha256 = _load_ranked_runs_with_source_hash(
        str(input_json),
        require_real_input=args.require_real_input,
    )

    finalists = select_finalists(ranked_runs, max_finalists=args.max_finalists)
    _validate_stage2_finalists(
        finalists,
        min_finalists=args.min_finalists,
        max_finalists=args.max_finalists,
    )

    finalists_summary = format_finalists_summary(ranked_runs, max_finalists=args.max_finalists)
    print(finalists_summary)
    print()
    print("Stage 2 depth/branch flags:")
    for row in finalists:
        print(
            f"- {row['config']}: --depth {row['depth']} --n-branches {row['n_branches']} --aspect-ratio {row['aspect_ratio']}"
        )

    _write_finalists_json(
        path=finalists_json,
        source=str(input_json),
        source_sha256=source_sha256,
        max_finalists=args.max_finalists,
        finalists=finalists,
    )
    _write_finalists_md(
        path=finalists_md,
        finalists_summary=finalists_summary,
        finalists=finalists,
    )

    if args.run_check_in:
        run_pilot_bundle_check(
            ranked_json_path=input_json,
            finalists_json_path=finalists_json,
            finalists_md_path=finalists_md,
            require_real_input=False,
            require_git_tracked=False,
            check_in=True,
            output_check_json=str(check_json_path),
        )

    if runbook_md is not None:
        _write_runbook_md(
            path=runbook_md,
            input_json=str(input_json),
            output_dir=args.output_dir,
            finalists_json=finalists_json,
            finalists_md=finalists_md,
            min_finalists=args.min_finalists,
            max_finalists=args.max_finalists,
            require_real_input=args.require_real_input,
            run_check_in=args.run_check_in,
            output_check_json=str(check_json_path) if check_json_path is not None else args.output_check_json,
            output_bundle_json=args.output_bundle_json,
        )

    if bundle_json_path is not None:
        _write_bundle_receipt(
            path=bundle_json_path,
            input_json=input_json,
            finalists_json=finalists_json,
            finalists_md=finalists_md,
            finalists_count=len(finalists),
            source_sha256=source_sha256,
            run_check_in=args.run_check_in,
            check_json_path=check_json_path,
        )

    print(
        "bundle_ok "
        f"input_json={input_json} "
        f"finalists={len(finalists)} "
        f"json={finalists_json} "
        f"md={finalists_md}"
        + (f" check_json={check_json_path}" if check_json_path is not None else "")
        + (f" runbook_md={runbook_md}" if runbook_md is not None else "")
        + (f" bundle_json={bundle_json_path}" if bundle_json_path is not None else "")
    )


if __name__ == "__main__":
    main()
