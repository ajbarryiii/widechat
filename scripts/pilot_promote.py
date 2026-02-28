"""Promote top pilot configs to Stage 2 training candidates.

Example:
python -m scripts.pilot_promote --input-json artifacts/pilot_ranked.json --output-md artifacts/stage2_finalists.md
"""

import argparse
import json

from nanochat.pilot_sweep import format_finalists_summary, select_finalists


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Stage 2 finalists from pilot ranking JSON")
    parser.add_argument("--input-json", type=str, required=True, help="pilot sweep JSON produced by scripts.pilot_sweep --output-json")
    parser.add_argument("--max-finalists", type=int, default=3, help="max number of qualified finalists to keep")
    parser.add_argument("--output-json", type=str, default="", help="optional path to write selected finalists JSON")
    parser.add_argument("--output-md", type=str, default="", help="optional path to write selected finalists markdown")
    return parser.parse_args()


def _load_ranked_runs(path: str) -> list[dict[str, int | float | bool | str | None]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    ranked_runs = payload.get("ranked_runs")
    if not isinstance(ranked_runs, list):
        raise ValueError("input JSON must include a ranked_runs list")
    if not ranked_runs:
        raise ValueError("ranked_runs must not be empty")
    return ranked_runs


def main() -> None:
    args = _parse_args()
    ranked_runs = _load_ranked_runs(args.input_json)

    finalists = select_finalists(ranked_runs, max_finalists=args.max_finalists)
    finalists_summary = format_finalists_summary(ranked_runs)

    print(finalists_summary)
    print()
    print("Stage 2 depth/branch flags:")
    for row in finalists:
        print(
            f"- {row['config']}: --depth {row['depth']} --n-branches {row['n_branches']} --aspect-ratio {row['aspect_ratio']}"
        )

    if args.output_json:
        payload = {
            "source": args.input_json,
            "max_finalists": args.max_finalists,
            "selected_finalists": finalists,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if args.output_md:
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
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    main()
