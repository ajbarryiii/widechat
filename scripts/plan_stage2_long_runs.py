"""Plan Stage 2 long-run training commands from promoted finalists.

Example:
python -m scripts.plan_stage2_long_runs --finalists-json artifacts/pilot/stage2_finalists.json
"""

import argparse
import json
import math
import shlex
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan Stage 2 long-run commands")
    parser.add_argument(
        "--finalists-json",
        default="artifacts/pilot/stage2_finalists.json",
        help="Stage 2 finalists artifact emitted by scripts.run_stage2_promotion_bundle",
    )
    parser.add_argument(
        "--token-budgets",
        default="1000000000,2000000000",
        help="comma-separated token budgets for long runs (e.g. 1000000000,2000000000)",
    )
    parser.add_argument(
        "--total-batch-size",
        type=int,
        default=524288,
        help="global batch size in tokens used to derive --num-iterations",
    )
    parser.add_argument(
        "--base-command",
        default="python -m scripts.base_train",
        help="base command prefix used for each planned training command",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="optional extra args appended to each generated command",
    )
    parser.add_argument(
        "--run-prefix",
        default="stage2",
        help="prefix used when generating --model-tag values",
    )
    parser.add_argument(
        "--output-plan-json",
        default="",
        help="optional output plan JSON (defaults to <finalists-dir>/stage2_long_runs_plan.json)",
    )
    parser.add_argument(
        "--output-runbook-md",
        default="",
        help="optional output markdown runbook (defaults to <finalists-dir>/stage2_long_runs_runbook.md)",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate inputs and print status without writing artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve commands and print status without writing artifacts",
    )
    return parser.parse_args()


def _parse_token_budgets(raw_value: str) -> list[int]:
    budgets: list[int] = []
    for piece in raw_value.split(","):
        trimmed = piece.strip()
        if not trimmed:
            continue
        try:
            budget = int(trimmed)
        except ValueError as exc:
            raise ValueError(f"invalid token budget '{trimmed}' (must be integer)") from exc
        if budget <= 0:
            raise ValueError(f"invalid token budget '{trimmed}' (must be > 0)")
        budgets.append(budget)
    if not budgets:
        raise ValueError("--token-budgets must include at least one positive integer")
    return budgets


def _resolve_output_paths(
    finalists_json: Path,
    output_plan_json: str,
    output_runbook_md: str,
) -> tuple[Path, Path]:
    base_dir = finalists_json.parent
    plan_json = Path(output_plan_json) if output_plan_json else base_dir / "stage2_long_runs_plan.json"
    runbook_md = Path(output_runbook_md) if output_runbook_md else base_dir / "stage2_long_runs_runbook.md"
    return plan_json, runbook_md


def _load_finalists(finalists_json: Path) -> list[dict[str, object]]:
    payload = json.loads(finalists_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("finalists JSON must be an object")
    finalists = payload.get("selected_finalists")
    if not isinstance(finalists, list) or not finalists:
        raise RuntimeError("finalists JSON must include non-empty selected_finalists list")
    required_fields = ("config", "depth", "n_branches", "aspect_ratio")
    for idx, row in enumerate(finalists, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(f"selected_finalists[{idx}] must be an object")
        for field in required_fields:
            if field not in row:
                raise RuntimeError(f"selected_finalists[{idx}] missing required field '{field}'")
    return finalists


def _iterations_for_budget(token_budget: int, total_batch_size: int) -> int:
    return math.ceil(token_budget / total_batch_size)


def _build_run_entries(
    finalists: list[dict[str, object]],
    token_budgets: list[int],
    total_batch_size: int,
    base_command: str,
    extra_args: str,
    run_prefix: str,
) -> list[dict[str, object]]:
    command_prefix = shlex.split(base_command)
    command_extra = shlex.split(extra_args)

    entries: list[dict[str, object]] = []
    for finalist in finalists:
        config = str(finalist["config"])
        depth = int(finalist["depth"])
        n_branches = int(finalist["n_branches"])
        aspect_ratio = int(finalist["aspect_ratio"])
        for token_budget in token_budgets:
            iterations = _iterations_for_budget(token_budget, total_batch_size)
            model_tag = f"{run_prefix}_{config}_tok{token_budget}"
            command_parts = [
                *command_prefix,
                "--depth",
                str(depth),
                "--n-branches",
                str(n_branches),
                "--aspect-ratio",
                str(aspect_ratio),
                "--total-batch-size",
                str(total_batch_size),
                "--num-iterations",
                str(iterations),
                "--model-tag",
                model_tag,
                *command_extra,
            ]
            entries.append(
                {
                    "config": config,
                    "depth": depth,
                    "n_branches": n_branches,
                    "aspect_ratio": aspect_ratio,
                    "token_budget": token_budget,
                    "total_batch_size": total_batch_size,
                    "num_iterations": iterations,
                    "model_tag": model_tag,
                    "command": shlex.join(command_parts),
                }
            )
    return entries


def _write_plan_json(
    path: Path,
    finalists_json: Path,
    token_budgets: list[int],
    total_batch_size: int,
    run_entries: list[dict[str, object]],
) -> None:
    payload = {
        "source": str(finalists_json),
        "token_budgets": token_budgets,
        "total_batch_size": total_batch_size,
        "runs": run_entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_runbook_md(
    path: Path,
    finalists_json: Path,
    token_budgets: list[int],
    total_batch_size: int,
    run_entries: list[dict[str, object]],
) -> None:
    lines = [
        "# Stage 2 Long-Run Training Runbook",
        "",
        f"- finalists_json: `{finalists_json}`",
        f"- token_budgets: `{','.join(str(value) for value in token_budgets)}`",
        f"- total_batch_size: `{total_batch_size}`",
        "",
        "## Commands",
        "```bash",
    ]
    for entry in run_entries:
        lines.append(entry["command"])
    lines.extend(["```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    finalists_json = Path(args.finalists_json)

    if args.preflight and args.dry_run:
        raise ValueError("--preflight and --dry-run are mutually exclusive")
    if args.total_batch_size <= 0:
        raise ValueError("--total-batch-size must be > 0")

    token_budgets = _parse_token_budgets(args.token_budgets)
    finalists = _load_finalists(finalists_json)
    run_entries = _build_run_entries(
        finalists=finalists,
        token_budgets=token_budgets,
        total_batch_size=args.total_batch_size,
        base_command=args.base_command,
        extra_args=args.extra_args,
        run_prefix=args.run_prefix,
    )
    plan_json, runbook_md = _resolve_output_paths(
        finalists_json=finalists_json,
        output_plan_json=args.output_plan_json,
        output_runbook_md=args.output_runbook_md,
    )

    if args.preflight:
        print(
            "stage2_long_run_plan_preflight_ok "
            f"finalists={len(finalists)} "
            f"runs={len(run_entries)} "
            f"plan_json={plan_json} "
            f"runbook_md={runbook_md}"
        )
        return

    if args.dry_run:
        print(
            "stage2_long_run_plan_dry_run_ok "
            f"finalists={len(finalists)} "
            f"runs={len(run_entries)} "
            f"plan_json={plan_json} "
            f"runbook_md={runbook_md}"
        )
        return

    _write_plan_json(
        path=plan_json,
        finalists_json=finalists_json,
        token_budgets=token_budgets,
        total_batch_size=args.total_batch_size,
        run_entries=run_entries,
    )
    _write_runbook_md(
        path=runbook_md,
        finalists_json=finalists_json,
        token_budgets=token_budgets,
        total_batch_size=args.total_batch_size,
        run_entries=run_entries,
    )

    print(
        "stage2_long_run_plan_ok "
        f"finalists={len(finalists)} "
        f"runs={len(run_entries)} "
        f"plan_json={plan_json} "
        f"runbook_md={runbook_md}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"stage2_long_run_plan_error error_type={type(exc).__name__} error={exc}", file=sys.stderr)
        raise
