"""Run Stage 2 promotion and artifact emission in one command.

Example:
python -m scripts.run_stage2_promotion_bundle --input-json artifacts/pilot/sample_ranked_runs.json --output-dir artifacts/pilot
"""

import argparse
import json
from pathlib import Path

from nanochat.pilot_sweep import format_finalists_summary, select_finalists
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
    print(
        "bundle_ok "
        f"finalists={len(finalists)} "
        f"json={finalists_json} "
        f"md={finalists_md}"
    )


if __name__ == "__main__":
    main()
