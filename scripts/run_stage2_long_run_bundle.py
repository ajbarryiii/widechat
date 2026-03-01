"""Emit Stage 2 long-run plan/runbook artifacts in one command.

Example:
python -m scripts.run_stage2_long_run_bundle --finalists-json artifacts/pilot/stage2_finalists.json --output-dir artifacts/pilot
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from scripts import plan_stage2_long_runs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 long-run planning bundle")
    parser.add_argument(
        "--finalists-json",
        default="artifacts/pilot/stage2_finalists.json",
        help="Stage 2 finalists artifact emitted by scripts.run_stage2_promotion_bundle",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/pilot",
        help="artifact directory for plan/runbook outputs",
    )
    parser.add_argument(
        "--token-budgets",
        default="1000000000,2000000000",
        help="comma-separated token budgets for long runs",
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
        help="optional output plan JSON (defaults to <output-dir>/stage2_long_runs_plan.json)",
    )
    parser.add_argument(
        "--output-runbook-md",
        default="",
        help="optional output markdown runbook (defaults to <output-dir>/stage2_long_runs_runbook.md)",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate inputs and print status without writing plan/runbook",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve commands and print status without writing plan/runbook",
    )
    parser.add_argument(
        "--dry-run-write-runbook",
        action="store_true",
        help="when used with --dry-run, write only the markdown runbook",
    )
    parser.add_argument(
        "--output-preflight-json",
        default="",
        help="optional path to write a machine-readable preflight receipt",
    )
    parser.add_argument(
        "--output-blocked-md",
        default="",
        help="optional path to write blocker diagnostics markdown on failure",
    )
    parser.add_argument(
        "--output-bundle-command-sh",
        default="",
        help="optional path to persist the fully-resolved bundle command",
    )
    return parser.parse_args()


def _resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    default_plan = Path(args.output_dir) / "stage2_long_runs_plan.json"
    default_runbook = Path(args.output_dir) / "stage2_long_runs_runbook.md"
    plan_json = Path(args.output_plan_json) if args.output_plan_json else default_plan
    runbook_md = Path(args.output_runbook_md) if args.output_runbook_md else default_runbook
    return plan_json, runbook_md


def _resolved_bundle_command(
    *,
    finalists_json: Path,
    output_dir: str,
    token_budgets: str,
    total_batch_size: int,
    base_command: str,
    extra_args: str,
    run_prefix: str,
    plan_json: Path,
    runbook_md: Path,
    preflight: bool,
    dry_run: bool,
    dry_run_write_runbook: bool,
    output_preflight_json: Path | None,
    output_blocked_md: Path | None,
    output_bundle_command_sh: Path | None,
) -> str:
    command = [
        "python",
        "-m",
        "scripts.run_stage2_long_run_bundle",
        "--finalists-json",
        str(finalists_json),
        "--output-dir",
        output_dir,
        "--token-budgets",
        token_budgets,
        "--total-batch-size",
        str(total_batch_size),
        "--base-command",
        base_command,
        "--extra-args",
        extra_args,
        "--run-prefix",
        run_prefix,
        "--output-plan-json",
        str(plan_json),
        "--output-runbook-md",
        str(runbook_md),
    ]
    if preflight:
        command.append("--preflight")
    if dry_run:
        command.append("--dry-run")
    if dry_run_write_runbook:
        command.append("--dry-run-write-runbook")
    if output_preflight_json is not None:
        command.extend(["--output-preflight-json", str(output_preflight_json)])
    if output_blocked_md is not None:
        command.extend(["--output-blocked-md", str(output_blocked_md)])
    if output_bundle_command_sh is not None:
        command.extend(["--output-bundle-command-sh", str(output_bundle_command_sh)])
    return shlex.join(command)


def _write_bundle_command(path: Path, command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(command + "\n", encoding="utf-8")


def _write_preflight_receipt(
    *,
    path: Path,
    finalists_json: Path,
    plan_json: Path,
    runbook_md: Path,
    finalists_count: int,
    run_count: int,
    token_budgets: list[int],
    total_batch_size: int,
    output_bundle_command_sh: Path | None,
) -> None:
    payload = {
        "status": "ok",
        "command": [*sys.argv],
        "finalists_json": str(finalists_json),
        "plan_json": str(plan_json),
        "runbook_md": str(runbook_md),
        "finalists_count": finalists_count,
        "run_count": run_count,
        "token_budgets": token_budgets,
        "total_batch_size": total_batch_size,
        "bundle_command_sh": str(output_bundle_command_sh) if output_bundle_command_sh is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_blocked_md(
    *,
    path: Path,
    error: Exception,
    finalists_json: Path,
    plan_json: Path,
    runbook_md: Path,
    output_preflight_json: Path | None,
    output_bundle_command_sh: Path | None,
) -> None:
    lines = [
        "# Stage 2 Long-Run Bundle Blocked",
        "",
        "- status: blocked",
        f"- error_type: `{type(error).__name__}`",
        f"- error: `{error}`",
        f"- finalists_json: `{finalists_json}`",
        f"- plan_json: `{plan_json}`",
        f"- runbook_md: `{runbook_md}`",
        (
            f"- preflight_json: `{output_preflight_json}`"
            if output_preflight_json is not None
            else "- preflight_json: (not configured)"
        ),
        (
            f"- bundle_command_sh: `{output_bundle_command_sh}`"
            if output_bundle_command_sh is not None
            else "- bundle_command_sh: (not configured)"
        ),
        "",
        "## Command",
        "```bash",
        shlex.join(sys.argv),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    finalists_json = Path(args.finalists_json)
    plan_json, runbook_md = _resolve_output_paths(args)
    preflight_json = Path(args.output_preflight_json) if args.output_preflight_json else None
    blocked_md = Path(args.output_blocked_md) if args.output_blocked_md else None
    bundle_command_sh = Path(args.output_bundle_command_sh) if args.output_bundle_command_sh else None

    try:
        if args.preflight and args.dry_run:
            raise ValueError("--preflight and --dry-run are mutually exclusive")
        if args.total_batch_size <= 0:
            raise ValueError("--total-batch-size must be > 0")

        token_budgets = plan_stage2_long_runs._parse_token_budgets(args.token_budgets)
        finalists = plan_stage2_long_runs._load_finalists(finalists_json)
        run_entries = plan_stage2_long_runs._build_run_entries(
            finalists=finalists,
            token_budgets=token_budgets,
            total_batch_size=args.total_batch_size,
            base_command=args.base_command,
            extra_args=args.extra_args,
            run_prefix=args.run_prefix,
        )

        resolved_command = _resolved_bundle_command(
            finalists_json=finalists_json,
            output_dir=args.output_dir,
            token_budgets=args.token_budgets,
            total_batch_size=args.total_batch_size,
            base_command=args.base_command,
            extra_args=args.extra_args,
            run_prefix=args.run_prefix,
            plan_json=plan_json,
            runbook_md=runbook_md,
            preflight=args.preflight,
            dry_run=args.dry_run,
            dry_run_write_runbook=args.dry_run_write_runbook,
            output_preflight_json=preflight_json,
            output_blocked_md=blocked_md,
            output_bundle_command_sh=bundle_command_sh,
        )
        if bundle_command_sh is not None:
            _write_bundle_command(bundle_command_sh, resolved_command)

        if args.preflight:
            if preflight_json is not None:
                _write_preflight_receipt(
                    path=preflight_json,
                    finalists_json=finalists_json,
                    plan_json=plan_json,
                    runbook_md=runbook_md,
                    finalists_count=len(finalists),
                    run_count=len(run_entries),
                    token_budgets=token_budgets,
                    total_batch_size=args.total_batch_size,
                    output_bundle_command_sh=bundle_command_sh,
                )
            print(
                "stage2_long_run_bundle_preflight_ok "
                f"finalists_json={finalists_json} "
                f"finalists={len(finalists)} "
                f"runs={len(run_entries)} "
                f"plan_json={plan_json} "
                f"runbook_md={runbook_md}"
                + (f" preflight_json={preflight_json}" if preflight_json is not None else "")
                + (f" bundle_command_sh={bundle_command_sh}" if bundle_command_sh is not None else "")
            )
            return

        if args.dry_run:
            if args.dry_run_write_runbook:
                plan_stage2_long_runs._write_runbook_md(
                    path=runbook_md,
                    finalists_json=finalists_json,
                    token_budgets=token_budgets,
                    total_batch_size=args.total_batch_size,
                    run_entries=run_entries,
                )
            print(
                "stage2_long_run_bundle_dry_run_ok "
                f"finalists_json={finalists_json} "
                f"finalists={len(finalists)} "
                f"runs={len(run_entries)} "
                f"plan_json={plan_json} "
                f"runbook_md={runbook_md}"
                + (f" runbook_written={runbook_md}" if args.dry_run_write_runbook else "")
                + (f" bundle_command_sh={bundle_command_sh}" if bundle_command_sh is not None else "")
            )
            return

        plan_stage2_long_runs._write_plan_json(
            path=plan_json,
            finalists_json=finalists_json,
            token_budgets=token_budgets,
            total_batch_size=args.total_batch_size,
            run_entries=run_entries,
        )
        plan_stage2_long_runs._write_runbook_md(
            path=runbook_md,
            finalists_json=finalists_json,
            token_budgets=token_budgets,
            total_batch_size=args.total_batch_size,
            run_entries=run_entries,
        )

        print(
            "stage2_long_run_bundle_ok "
            f"finalists_json={finalists_json} "
            f"finalists={len(finalists)} "
            f"runs={len(run_entries)} "
            f"plan_json={plan_json} "
            f"runbook_md={runbook_md}"
            + (f" bundle_command_sh={bundle_command_sh}" if bundle_command_sh is not None else "")
        )
    except Exception as exc:
        if blocked_md is not None:
            _write_blocked_md(
                path=blocked_md,
                error=exc,
                finalists_json=finalists_json,
                plan_json=plan_json,
                runbook_md=runbook_md,
                output_preflight_json=preflight_json,
                output_bundle_command_sh=bundle_command_sh,
            )
            print(f"stage2_long_run_bundle_blocked blocked_md={blocked_md} error={type(exc).__name__}")
        raise


if __name__ == "__main__":
    main()
