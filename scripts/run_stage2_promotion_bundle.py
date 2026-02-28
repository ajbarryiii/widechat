"""Run Stage 2 promotion and artifact emission in one command.

Example:
python -m scripts.run_stage2_promotion_bundle --input-json artifacts/pilot/sample_ranked_runs.json --output-dir artifacts/pilot
"""

import argparse
import json
from pathlib import Path

from nanochat.pilot_sweep import format_finalists_summary, select_finalists
from scripts.check_pilot_sweep_artifacts import run_pilot_bundle_check
from scripts.pilot_promote import _load_ranked_runs, _validate_stage2_finalists


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 promotion artifact bundle")
    parser.add_argument(
        "--input-json",
        required=True,
        help="pilot ranking JSON produced by scripts.pilot_sweep --output-json",
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
    return parser.parse_args()


def _resolve_output_paths(output_dir: str, output_json: str, output_md: str) -> tuple[Path, Path]:
    base = Path(output_dir)
    finalists_json = Path(output_json) if output_json else base / "stage2_finalists.json"
    finalists_md = Path(output_md) if output_md else base / "stage2_finalists.md"
    return finalists_json, finalists_md


def _write_finalists_json(
    path: Path,
    source: str,
    max_finalists: int,
    finalists: list[dict[str, int | float | bool | str | None]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command_lines = [
        "python -m scripts.run_stage2_promotion_bundle \\",
        f"  --input-json {input_json} \\",
        f"  --output-dir {output_dir} \\",
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
            command_lines.append(f"  --output-check-json {output_check_json}")

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
        "python -m scripts.run_pilot_check_in \\",
        f"  --artifacts-dir {output_dir} \\",
        f"  --ranked-json {input_json} \\",
        f"  --finalists-json {finalists_json} \\",
        f"  --finalists-md {finalists_md} \\",
        f"  --output-check-json {Path(output_dir) / 'pilot_bundle_check.json'}",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    finalists_json, finalists_md = _resolve_output_paths(args.output_dir, args.output_json, args.output_md)
    ranked_runs = _load_ranked_runs(args.input_json, require_real_input=args.require_real_input)

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
        source=args.input_json,
        max_finalists=args.max_finalists,
        finalists=finalists,
    )
    _write_finalists_md(
        path=finalists_md,
        finalists_summary=finalists_summary,
        finalists=finalists,
    )

    runbook_md = Path(args.output_runbook_md) if args.output_runbook_md else None
    check_json_path: Path | None = None
    if args.run_check_in:
        check_json_path = Path(args.output_check_json) if args.output_check_json else Path(args.output_dir) / "pilot_bundle_check.json"
        run_pilot_bundle_check(
            ranked_json_path=Path(args.input_json),
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
            input_json=args.input_json,
            output_dir=args.output_dir,
            finalists_json=finalists_json,
            finalists_md=finalists_md,
            min_finalists=args.min_finalists,
            max_finalists=args.max_finalists,
            require_real_input=args.require_real_input,
            run_check_in=args.run_check_in,
            output_check_json=str(check_json_path) if check_json_path is not None else args.output_check_json,
        )

    print(
        "bundle_ok "
        f"finalists={len(finalists)} "
        f"json={finalists_json} "
        f"md={finalists_md}"
        + (f" check_json={check_json_path}" if check_json_path is not None else "")
        + (f" runbook_md={runbook_md}" if runbook_md is not None else "")
    )


if __name__ == "__main__":
    main()
