"""Compare Stage 2 long-run metrics against baseline config.

Example:
python -m scripts.compare_stage2_long_runs --input-json artifacts/pilot/stage2_long_runs_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Stage 2 long-run metrics against baseline")
    parser.add_argument(
        "--input-json",
        default="artifacts/pilot/stage2_long_runs_metrics.json",
        help="JSON artifact containing Stage 2 long-run metrics",
    )
    parser.add_argument(
        "--baseline-config",
        default="12x1",
        help="baseline config label used for relative comparisons",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="optional output JSON report (defaults to <input-dir>/stage2_baseline_comparison.json)",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="optional output markdown report (defaults to <input-dir>/stage2_baseline_comparison.md)",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate input schema and baseline presence without writing outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute comparison and print status without writing outputs",
    )
    return parser.parse_args()


def _resolve_output_paths(input_json: Path, output_json: str, output_md: str) -> tuple[Path, Path]:
    base_dir = input_json.parent
    json_path = Path(output_json) if output_json else base_dir / "stage2_baseline_comparison.json"
    md_path = Path(output_md) if output_md else base_dir / "stage2_baseline_comparison.md"
    return json_path, md_path


def _as_positive_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{field_name} must be a positive integer")
    return value


def _as_number(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"{field_name} must be numeric")
    return float(value)


def _load_runs(input_json: Path) -> list[dict[str, object]]:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("input JSON must be an object")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise RuntimeError("input JSON must include non-empty runs list")

    normalized: list[dict[str, object]] = []
    for idx, row in enumerate(runs, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(f"runs[{idx}] must be an object")
        config = row.get("config")
        if not isinstance(config, str) or not config:
            raise RuntimeError(f"runs[{idx}].config must be non-empty string")

        token_budget = _as_positive_int(row.get("token_budget"), f"runs[{idx}].token_budget")
        final_val_bpb = _as_number(row.get("final_val_bpb"), f"runs[{idx}].final_val_bpb")
        min_val_bpb = _as_number(row.get("min_val_bpb"), f"runs[{idx}].min_val_bpb")
        tok_per_sec = _as_number(row.get("selected_tok_per_sec"), f"runs[{idx}].selected_tok_per_sec")
        unstable = row.get("unstable", False)
        if not isinstance(unstable, bool):
            raise RuntimeError(f"runs[{idx}].unstable must be boolean when provided")

        normalized.append(
            {
                "config": config,
                "token_budget": token_budget,
                "final_val_bpb": final_val_bpb,
                "min_val_bpb": min_val_bpb,
                "selected_tok_per_sec": tok_per_sec,
                "unstable": unstable,
            }
        )

    return normalized


def _build_comparison(runs: list[dict[str, object]], baseline_config: str) -> dict[str, object]:
    runs_by_budget: dict[int, list[dict[str, object]]] = {}
    for row in runs:
        runs_by_budget.setdefault(int(row["token_budget"]), []).append(row)

    comparisons: list[dict[str, object]] = []
    winners_by_budget: list[dict[str, object]] = []

    for token_budget in sorted(runs_by_budget):
        budget_rows = runs_by_budget[token_budget]
        baseline_rows = [row for row in budget_rows if row["config"] == baseline_config]
        if len(baseline_rows) != 1:
            raise RuntimeError(
                f"token_budget={token_budget} must include exactly one baseline row for config '{baseline_config}'"
            )
        baseline = baseline_rows[0]
        baseline_final = float(baseline["final_val_bpb"])
        baseline_min = float(baseline["min_val_bpb"])
        baseline_tps = float(baseline["selected_tok_per_sec"])

        best_row = min(budget_rows, key=lambda item: float(item["final_val_bpb"]))
        winners_by_budget.append(
            {
                "token_budget": token_budget,
                "best_config": best_row["config"],
                "baseline_config": baseline_config,
                "best_final_val_bpb": float(best_row["final_val_bpb"]),
                "baseline_final_val_bpb": baseline_final,
                "delta_final_val_bpb": float(best_row["final_val_bpb"]) - baseline_final,
            }
        )

        for row in sorted(budget_rows, key=lambda item: float(item["final_val_bpb"])):
            row_tps = float(row["selected_tok_per_sec"])
            tps_delta_pct = 0.0
            if baseline_tps > 0:
                tps_delta_pct = ((row_tps / baseline_tps) - 1.0) * 100.0
            comparisons.append(
                {
                    "config": row["config"],
                    "token_budget": token_budget,
                    "unstable": bool(row["unstable"]),
                    "final_val_bpb": float(row["final_val_bpb"]),
                    "min_val_bpb": float(row["min_val_bpb"]),
                    "selected_tok_per_sec": row_tps,
                    "delta_final_val_bpb_vs_baseline": float(row["final_val_bpb"]) - baseline_final,
                    "delta_min_val_bpb_vs_baseline": float(row["min_val_bpb"]) - baseline_min,
                    "delta_tok_per_sec_pct_vs_baseline": tps_delta_pct,
                    "better_final_than_baseline": float(row["final_val_bpb"]) < baseline_final,
                }
            )

    max_budget = max(runs_by_budget)
    top_at_max_budget = [
        row
        for row in comparisons
        if int(row["token_budget"]) == max_budget
    ]
    top_at_max_budget.sort(key=lambda item: float(item["final_val_bpb"]))

    return {
        "baseline_config": baseline_config,
        "token_budgets": sorted(runs_by_budget),
        "comparisons": comparisons,
        "winners_by_budget": winners_by_budget,
        "max_token_budget": max_budget,
        "top_at_max_budget": top_at_max_budget,
    }


def _write_json(path: Path, payload: dict[str, object], source: Path) -> None:
    output = {
        "source": str(source),
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, object], source: Path) -> None:
    lines = [
        "# Stage 2 Baseline Comparison",
        "",
        f"- source: `{source}`",
        f"- baseline_config: `{payload['baseline_config']}`",
        f"- token_budgets: `{','.join(str(x) for x in payload['token_budgets'])}`",
        "",
        "## Winners by token budget",
        "",
        "| token_budget | best_config | best_final_val_bpb | baseline_final_val_bpb | delta_final_val_bpb |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in payload["winners_by_budget"]:
        lines.append(
            "| "
            f"{int(row['token_budget'])} | {row['best_config']} | "
            f"{float(row['best_final_val_bpb']):.6f} | {float(row['baseline_final_val_bpb']):.6f} | "
            f"{float(row['delta_final_val_bpb']):+.6f} |"
        )

    lines.extend(
        [
            "",
            "## Per-run comparison",
            "",
            "| config | token_budget | unstable | final_val_bpb | delta_final_vs_baseline | tok_per_sec | delta_tok_per_sec_pct |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["comparisons"]:
        lines.append(
            "| "
            f"{row['config']} | {int(row['token_budget'])} | {str(bool(row['unstable'])).lower()} | "
            f"{float(row['final_val_bpb']):.6f} | {float(row['delta_final_val_bpb_vs_baseline']):+.6f} | "
            f"{float(row['selected_tok_per_sec']):.1f} | {float(row['delta_tok_per_sec_pct_vs_baseline']):+.2f}% |"
        )

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.preflight and args.dry_run:
        raise ValueError("--preflight and --dry-run are mutually exclusive")

    input_json = Path(args.input_json)
    output_json, output_md = _resolve_output_paths(input_json, args.output_json, args.output_md)

    runs = _load_runs(input_json)
    comparison = _build_comparison(runs, baseline_config=args.baseline_config)

    if args.preflight:
        print(
            "stage2_compare_preflight_ok "
            f"runs={len(runs)} "
            f"token_budgets={len(comparison['token_budgets'])} "
            f"output_json={output_json} "
            f"output_md={output_md}"
        )
        return

    if args.dry_run:
        print(
            "stage2_compare_dry_run_ok "
            f"runs={len(runs)} "
            f"token_budgets={len(comparison['token_budgets'])} "
            f"output_json={output_json} "
            f"output_md={output_md}"
        )
        return

    _write_json(output_json, comparison, source=input_json)
    _write_markdown(output_md, comparison, source=input_json)

    print(
        "stage2_compare_ok "
        f"runs={len(runs)} "
        f"token_budgets={len(comparison['token_budgets'])} "
        f"output_json={output_json} "
        f"output_md={output_md}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"stage2_compare_error error_type={type(exc).__name__} error={exc}", file=sys.stderr)
        raise
