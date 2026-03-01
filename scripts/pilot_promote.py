"""Promote top pilot configs to Stage 2 training candidates.

Example:
python -m scripts.pilot_promote --input-json artifacts/pilot_ranked.json --output-md artifacts/stage2_finalists.md
"""

import argparse
import hashlib
import json
from pathlib import Path

from nanochat.pilot_sweep import format_finalists_summary, select_finalists


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Stage 2 finalists from pilot ranking JSON")
    parser.add_argument("--input-json", type=str, required=True, help="pilot sweep JSON produced by scripts.pilot_sweep --output-json")
    parser.add_argument("--min-finalists", type=int, default=2, help="minimum number of qualified finalists required")
    parser.add_argument("--max-finalists", type=int, default=3, help="max number of qualified finalists to keep")
    parser.add_argument("--output-json", type=str, default="", help="optional path to write selected finalists JSON")
    parser.add_argument("--output-md", type=str, default="", help="optional path to write selected finalists markdown")
    parser.add_argument(
        "--require-real-input",
        action="store_true",
        help="reject sample/fixture ranked-run JSON inputs",
    )
    return parser.parse_args()


def _is_sample_input_path(path: str) -> bool:
    name = Path(path).name.lower()
    return name.startswith("sample")


def _stable_json_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_ranked_payload(
    path: str,
    *,
    require_real_input: bool = False,
) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("input JSON must be a JSON object")

    if require_real_input and (_is_sample_input_path(path) or payload.get("is_sample") is True):
        raise ValueError("--require-real-input rejects sample/fixture ranked-run artifacts")

    return payload


def _load_ranked_runs_from_payload(payload: dict[str, object]) -> list[dict[str, int | float | bool | str | None]]:

    ranked_runs = payload.get("ranked_runs")
    if not isinstance(ranked_runs, list):
        raise ValueError("input JSON must include a ranked_runs list")
    if not ranked_runs:
        raise ValueError("ranked_runs must not be empty")

    for index, row in enumerate(ranked_runs):
        if not isinstance(row, dict):
            raise ValueError(f"ranked_runs[{index}] must be a JSON object")
        _validate_ranked_run_row(row, index=index)
    return ranked_runs


def _load_ranked_runs(
    path: str,
    *,
    require_real_input: bool = False,
) -> list[dict[str, int | float | bool | str | None]]:
    payload = _load_ranked_payload(path, require_real_input=require_real_input)
    return _load_ranked_runs_from_payload(payload)


def _load_ranked_runs_with_source_hash(
    path: str,
    *,
    require_real_input: bool = False,
) -> tuple[list[dict[str, int | float | bool | str | None]], str]:
    payload = _load_ranked_payload(path, require_real_input=require_real_input)
    ranked_runs = _load_ranked_runs_from_payload(payload)
    return ranked_runs, _stable_json_sha256(payload)


def _validate_ranked_run_row(
    row: dict[str, int | float | bool | str | None],
    *,
    index: int,
) -> None:
    _require_str_field(row, "config", index=index)
    _require_positive_int_field(row, "depth", index=index)
    _require_positive_int_field(row, "n_branches", index=index)
    _require_positive_int_field(row, "aspect_ratio", index=index)
    _require_nonnegative_number_field(row, "selected_tok_per_sec", index=index)
    _require_number_field(row, "min_val_bpb", index=index)
    _require_positive_int_field(row, "token_budget", index=index)

    qualified = _require_bool_field(row, "qualified", index=index)
    rank = _require_optional_positive_int_field(row, "rank", index=index)
    disqualify_reason = row.get("disqualify_reason")

    if qualified and rank is None:
        raise ValueError(f"ranked_runs[{index}] qualified row must include a positive integer rank")
    if not qualified and rank is not None:
        raise ValueError(f"ranked_runs[{index}] disqualified row must set rank to null")

    if qualified:
        if disqualify_reason is not None:
            raise ValueError(f"ranked_runs[{index}] qualified row must set disqualify_reason to null")
    else:
        if not isinstance(disqualify_reason, str) or not disqualify_reason:
            raise ValueError(
                f"ranked_runs[{index}] disqualified row must include non-empty disqualify_reason"
            )


def _require_str_field(
    row: dict[str, int | float | bool | str | None],
    key: str,
    *,
    index: int,
) -> None:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"ranked_runs[{index}] missing non-empty string field: {key}")


def _require_positive_int_field(
    row: dict[str, int | float | bool | str | None],
    key: str,
    *,
    index: int,
) -> None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"ranked_runs[{index}] missing positive integer field: {key}")


def _require_bool_field(
    row: dict[str, int | float | bool | str | None],
    key: str,
    *,
    index: int,
) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"ranked_runs[{index}] missing boolean field: {key}")
    return value


def _require_number_field(
    row: dict[str, int | float | bool | str | None],
    key: str,
    *,
    index: int,
) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"ranked_runs[{index}] missing numeric field: {key}")
    return float(value)


def _require_nonnegative_number_field(
    row: dict[str, int | float | bool | str | None],
    key: str,
    *,
    index: int,
) -> float:
    value = _require_number_field(row, key, index=index)
    if value < 0:
        raise ValueError(f"ranked_runs[{index}] field must be >= 0: {key}")
    return value


def _require_optional_positive_int_field(
    row: dict[str, int | float | bool | str | None],
    key: str,
    *,
    index: int,
) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"ranked_runs[{index}] field must be null or a positive integer: {key}")
    return value


def _validate_stage2_finalists(
    finalists: list[dict[str, int | float | bool | str | None]],
    *,
    min_finalists: int,
    max_finalists: int,
) -> None:
    if min_finalists <= 0:
        raise ValueError("--min-finalists must be >= 1")
    if max_finalists <= 0:
        raise ValueError("--max-finalists must be >= 1")
    if min_finalists > max_finalists:
        raise ValueError("--min-finalists must be <= --max-finalists")

    finalist_count = len(finalists)
    if finalist_count < min_finalists:
        raise RuntimeError(
            f"expected at least {min_finalists} qualified finalists, found {finalist_count}"
        )
    if finalist_count > max_finalists:
        raise RuntimeError(
            f"expected at most {max_finalists} qualified finalists, found {finalist_count}"
        )


def main() -> None:
    args = _parse_args()
    ranked_runs, source_sha256 = _load_ranked_runs_with_source_hash(
        args.input_json,
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

    if args.output_json:
        payload = {
            "source": args.input_json,
            "source_sha256": source_sha256,
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
