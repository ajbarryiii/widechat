"""Run a Stage 1 pilot sweep across depth x branch configs.

Example:
python -m scripts.pilot_sweep --device-type cuda --total-batch-size 524288 --device-batch-size 16
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import shlex
import sys

from nanochat.pilot_sweep import (
    DEFAULT_PILOT_TARGETS,
    PilotTarget,
    apply_ranking_rule,
    build_pilot_command,
    format_finalists_summary,
    format_ranking_table,
    run_single_pilot,
    select_finalists,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pilot sweep and apply ranking rule")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = base_train autodetect)")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--total-batch-size", type=int, required=True)
    parser.add_argument("--device-batch-size", type=int, required=True)
    parser.add_argument("--pilot-tokens", type=int, default=250_000_000)
    parser.add_argument("--eval-every", type=int, default=75)
    parser.add_argument("--eval-tokens", type=int, default=1_048_576)
    parser.add_argument("--slowdown-threshold-pct", type=float, default=5.0)
    parser.add_argument("--clear-bpb-gain", type=float, default=0.02)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--output-json", type=str, default="", help="optional path to write machine-readable results")
    parser.add_argument("--output-md", type=str, default="", help="optional path to write markdown ranking table + finalists")
    parser.add_argument(
        "--max-finalists",
        type=int,
        default=3,
        help="max number of qualified finalists to record in artifacts",
    )
    parser.add_argument(
        "--output-finalists-json",
        type=str,
        default="",
        help="optional path to write selected finalists JSON",
    )
    parser.add_argument(
        "--output-finalists-md",
        type=str,
        default="",
        help="optional path to write selected finalists markdown",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="",
        help="optional directory for per-config logs/metrics artifacts",
    )
    parser.add_argument(
        "--output-runbook-md",
        type=str,
        default="",
        help="optional path to write reproducible sweep/resume/check-in command runbook",
    )
    parser.add_argument(
        "--resume-from-artifacts",
        action="store_true",
        help="reuse existing per-config artifacts and skip rerunning completed configs",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate planned commands/artifacts without running training",
    )
    parser.add_argument(
        "--output-preflight-json",
        type=str,
        default="",
        help="optional path to write machine-readable preflight receipt",
    )
    parser.add_argument(
        "--output-launch-manifest-json",
        type=str,
        default="",
        help="optional path to write machine-readable planned sweep commands",
    )
    parser.add_argument(
        "--output-launch-script-sh",
        type=str,
        default="",
        help="optional path to write shell script with canonical per-config pilot_sweep launch commands",
    )
    parser.add_argument(
        "--output-blocked-md",
        type=str,
        default="",
        help="optional path to write markdown blocker diagnostics when preflight/run fails",
    )
    parser.add_argument("--extra-arg", action="append", default=[], help="forward extra arg to each base_train run")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="optional config label to run (repeatable, e.g. --target 12x1 --target 6x2)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    return parser.parse_args()


def _sanitize_label(label: str) -> str:
    safe = []
    for ch in label:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("-")
    slug = "".join(safe).strip("-")
    return slug or "run"


def _stable_json_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_blocked_markdown(
    *,
    output_path: str,
    mode: str,
    reason: str,
    args: argparse.Namespace,
) -> None:
    path = os.path.abspath(output_path)
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    lines = [
        "# Pilot Sweep Blocked",
        "",
        "## Receipt",
        f"- generated_at_utc: `{_now_utc()}`",
        f"- mode: `{mode}`",
        f"- preflight: `{str(args.preflight).lower()}`",
        f"- dry_run: `{str(args.dry_run).lower()}`",
        f"- device_type: `{args.device_type or '<auto>'}`",
        f"- artifacts_dir: `{args.artifacts_dir or '<none>'}`",
        f"- output_preflight_json: `{args.output_preflight_json or '<none>'}`",
        f"- command: `{json.dumps([*sys.argv])}`",
        "",
        "## Blocker",
        "```text",
        reason,
        "```",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_run_artifacts(
    artifacts_dir: str,
    run_index: int,
    run_result: dict[str, int | float | bool | str | None],
    output_text: str,
) -> None:
    prefix = f"{run_index:02d}-{_sanitize_label(str(run_result['config']))}"
    os.makedirs(artifacts_dir, exist_ok=True)

    log_path = os.path.join(artifacts_dir, f"{prefix}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    metrics_path = os.path.join(artifacts_dir, f"{prefix}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(run_result, f, indent=2)


def _artifact_paths(artifacts_dir: str, run_index: int, config_label: str) -> tuple[str, str]:
    prefix = f"{run_index:02d}-{_sanitize_label(config_label)}"
    return (
        os.path.join(artifacts_dir, f"{prefix}.log"),
        os.path.join(artifacts_dir, f"{prefix}.json"),
    )


def _resolve_selected_targets(selected_labels: list[str]) -> list[tuple[int, PilotTarget]]:
    if not selected_labels:
        return list(enumerate(DEFAULT_PILOT_TARGETS, start=1))

    unknown = sorted({label for label in selected_labels if label not in {t.label for t in DEFAULT_PILOT_TARGETS}})
    if unknown:
        raise ValueError(f"unknown --target labels: {', '.join(unknown)}")

    duplicate_labels = sorted({label for label in selected_labels if selected_labels.count(label) > 1})
    if duplicate_labels:
        raise ValueError(f"duplicate --target labels are not allowed: {', '.join(duplicate_labels)}")

    selected_set = set(selected_labels)
    return [
        (index, target)
        for index, target in enumerate(DEFAULT_PILOT_TARGETS, start=1)
        if target.label in selected_set
    ]


def _load_existing_run_artifact(
    artifacts_dir: str,
    run_index: int,
    config_label: str,
) -> dict[str, int | float | bool | str | None] | None:
    _log_path, metrics_path = _artifact_paths(artifacts_dir, run_index, config_label)
    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"artifact metrics must be a JSON object: {metrics_path}")

    loaded_config = loaded.get("config")
    if loaded_config is not None and loaded_config != config_label:
        raise ValueError(
            f"artifact config mismatch for run {run_index}: expected {config_label}, got {loaded_config}"
        )
    return loaded


def _validate_resume_run_artifact(
    loaded_run: dict[str, int | float | bool | str | None],
    *,
    artifacts_dir: str,
    run_index: int,
    config_label: str,
    expected_token_budget: int,
) -> None:
    log_path, metrics_path = _artifact_paths(artifacts_dir, run_index, config_label)
    if not os.path.exists(log_path):
        raise ValueError(
            "resume artifact is incomplete; expected log file for "
            f"{config_label}: {log_path}"
        )

    selected_tok_per_sec = loaded_run.get("selected_tok_per_sec")
    if isinstance(selected_tok_per_sec, bool) or not isinstance(selected_tok_per_sec, (int, float)):
        raise ValueError(
            "resume artifact missing numeric selected_tok_per_sec for "
            f"{config_label}: {metrics_path}"
        )

    unstable = loaded_run.get("unstable")
    if not isinstance(unstable, bool):
        raise ValueError(
            f"resume artifact missing boolean unstable for {config_label}: {metrics_path}"
        )

    token_budget = loaded_run.get("token_budget")
    if isinstance(token_budget, bool) or not isinstance(token_budget, (int, float)):
        raise ValueError(
            f"resume artifact missing numeric token_budget for {config_label}: {metrics_path}"
        )
    if int(token_budget) != expected_token_budget:
        raise ValueError(
            "resume artifact token_budget mismatch for "
            f"{config_label}: expected {expected_token_budget}, got {int(token_budget)}"
        )


def _render_pilot_sweep_runbook(
    *,
    args: argparse.Namespace,
    ranked_json_path: str,
    ranking_md_path: str,
    finalists_json_path: str,
    finalists_md_path: str,
) -> str:
    base_command = [
        args.python_exe,
        "-m",
        "scripts.pilot_sweep",
        "--total-batch-size",
        str(args.total_batch_size),
        "--device-batch-size",
        str(args.device_batch_size),
        "--pilot-tokens",
        str(args.pilot_tokens),
        "--eval-every",
        str(args.eval_every),
        "--eval-tokens",
        str(args.eval_tokens),
        "--max-seq-len",
        str(args.max_seq_len),
        "--slowdown-threshold-pct",
        str(args.slowdown_threshold_pct),
        "--clear-bpb-gain",
        str(args.clear_bpb_gain),
        "--max-finalists",
        str(args.max_finalists),
        "--artifacts-dir",
        args.artifacts_dir,
        "--output-json",
        ranked_json_path,
        "--output-md",
        ranking_md_path,
        "--output-finalists-json",
        finalists_json_path,
        "--output-finalists-md",
        finalists_md_path,
    ]
    if args.device_type:
        base_command.extend(["--device-type", args.device_type])
    for target in args.target:
        base_command.extend(["--target", target])
    for extra_arg in args.extra_arg:
        base_command.extend(["--extra-arg", extra_arg])

    resume_command = [*base_command, "--resume-from-artifacts"]

    runbook_lines = [
        "## Pilot Sweep Runbook",
        "",
        "### Artifact paths",
        "",
        f"- ranked runs JSON: `{ranked_json_path}`",
        f"- ranking markdown: `{ranking_md_path}`",
        f"- Stage 2 finalists JSON: `{finalists_json_path}`",
        f"- Stage 2 finalists markdown: `{finalists_md_path}`",
        f"- strict check receipt JSON: `{os.path.join(args.artifacts_dir, 'pilot_bundle_check.json')}`",
        "",
        "### Commands",
        "",
        "1. Initial sweep run:",
        "",
        "```bash",
        shlex.join(base_command),
        "```",
        "",
        "2. Resume interrupted run from existing artifacts:",
        "",
        "```bash",
        shlex.join(resume_command),
        "```",
        "",
        "3. Strict check-in validation on emitted artifacts:",
        "",
        "```bash",
        shlex.join(
            [
                args.python_exe,
                "-m",
                "scripts.run_pilot_check_in",
                "--artifacts-dir",
                args.artifacts_dir,
                "--ranked-json",
                os.path.basename(ranked_json_path),
                "--finalists-json",
                os.path.basename(finalists_json_path),
                "--finalists-md",
                os.path.basename(finalists_md_path),
                "--output-check-json",
                os.path.join(args.artifacts_dir, "pilot_bundle_check.json"),
            ]
        ),
        "```",
        "",
    ]
    return "\n".join(runbook_lines)


def _render_launch_script(
    *,
    args: argparse.Namespace,
    selected_targets: list[tuple[int, PilotTarget]],
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Canonical Stage 1 pilot launch commands generated by scripts.pilot_sweep.",
        "# Run one command per target (or shard them across operators) to emit per-config log/metrics artifacts.",
        "",
    ]

    for index, target in selected_targets:
        log_path, metrics_path = _artifact_paths(args.artifacts_dir, index, target.label)
        command = [
            args.python_exe,
            "-m",
            "scripts.pilot_sweep",
            "--total-batch-size",
            str(args.total_batch_size),
            "--device-batch-size",
            str(args.device_batch_size),
            "--pilot-tokens",
            str(args.pilot_tokens),
            "--eval-every",
            str(args.eval_every),
            "--eval-tokens",
            str(args.eval_tokens),
            "--max-seq-len",
            str(args.max_seq_len),
            "--slowdown-threshold-pct",
            str(args.slowdown_threshold_pct),
            "--clear-bpb-gain",
            str(args.clear_bpb_gain),
            "--max-finalists",
            str(args.max_finalists),
            "--artifacts-dir",
            args.artifacts_dir,
            "--target",
            target.label,
        ]
        if args.device_type:
            command.extend(["--device-type", args.device_type])
        if args.resume_from_artifacts:
            command.append("--resume-from-artifacts")
        for extra_arg in args.extra_arg:
            command.extend(["--extra-arg", extra_arg])

        lines.extend(
            [
                f"# [{index:02d}] {target.label}",
                f"# expected_log: {log_path}",
                f"# expected_metrics: {metrics_path}",
                shlex.join(command),
                "",
            ]
        )

    return "\n".join(lines)


def _run_preflight(
    *,
    args: argparse.Namespace,
    selected_targets: list[tuple[int, PilotTarget]],
    is_full_grid: bool,
) -> dict[str, object]:
    errors: list[str] = []
    target_receipts: list[dict[str, object]] = []

    for index, target in selected_targets:
        target_receipt: dict[str, object] = {
            "index": index,
            "config": target.label,
            "depth": target.depth,
            "n_branches": target.n_branches,
            "aspect_ratio": target.aspect_ratio,
        }
        try:
            command, num_iterations = build_pilot_command(
                target=target,
                python_exe=args.python_exe,
                max_seq_len=args.max_seq_len,
                total_batch_size=args.total_batch_size,
                device_batch_size=args.device_batch_size,
                pilot_tokens=args.pilot_tokens,
                eval_every=args.eval_every,
                eval_tokens=args.eval_tokens,
                device_type=args.device_type,
                extra_args=args.extra_arg,
            )
        except ValueError as exc:
            errors.append(f"{target.label}: {exc}")
            target_receipt["ok"] = False
            target_receipt["error"] = str(exc)
            target_receipts.append(target_receipt)
            continue

        target_receipt["ok"] = True
        target_receipt["num_iterations"] = num_iterations
        target_receipt["token_budget"] = num_iterations * args.total_batch_size
        target_receipt["command"] = command

        if args.resume_from_artifacts:
            loaded_run = _load_existing_run_artifact(
                artifacts_dir=args.artifacts_dir,
                run_index=index,
                config_label=target.label,
            )
            target_receipt["resume_artifact_found"] = loaded_run is not None
            if loaded_run is not None:
                try:
                    _validate_resume_run_artifact(
                        loaded_run,
                        artifacts_dir=args.artifacts_dir,
                        run_index=index,
                        config_label=target.label,
                        expected_token_budget=target_receipt["token_budget"],
                    )
                except ValueError as exc:
                    errors.append(f"{target.label}: {exc}")
                    target_receipt["ok"] = False
                    target_receipt["error"] = str(exc)

        target_receipts.append(target_receipt)

    return {
        "ok": not errors,
        "is_full_grid": is_full_grid,
        "resume_from_artifacts": args.resume_from_artifacts,
        "targets": target_receipts,
        "errors": errors,
    }


def _build_launch_manifest(
    *,
    args: argparse.Namespace,
    selected_targets: list[tuple[int, PilotTarget]],
    is_full_grid: bool,
) -> dict[str, object]:
    targets: list[dict[str, object]] = []
    for index, target in selected_targets:
        command, num_iterations = build_pilot_command(
            target=target,
            python_exe=args.python_exe,
            max_seq_len=args.max_seq_len,
            total_batch_size=args.total_batch_size,
            device_batch_size=args.device_batch_size,
            pilot_tokens=args.pilot_tokens,
            eval_every=args.eval_every,
            eval_tokens=args.eval_tokens,
            device_type=args.device_type,
            extra_args=args.extra_arg,
        )
        target_manifest: dict[str, object] = {
            "index": index,
            "config": target.label,
            "depth": target.depth,
            "n_branches": target.n_branches,
            "aspect_ratio": target.aspect_ratio,
            "num_iterations": num_iterations,
            "token_budget": num_iterations * args.total_batch_size,
            "command": command,
            "command_shell": shlex.join(command),
        }
        if args.artifacts_dir:
            log_path, metrics_path = _artifact_paths(args.artifacts_dir, index, target.label)
            target_manifest["log_path"] = log_path
            target_manifest["metrics_path"] = metrics_path
        targets.append(target_manifest)

    return {
        "generated_at_utc": _now_utc(),
        "is_full_grid": is_full_grid,
        "resume_from_artifacts": args.resume_from_artifacts,
        "preflight": args.preflight,
        "dry_run": args.dry_run,
        "artifacts_dir": args.artifacts_dir,
        "targets": targets,
    }


def _run(args: argparse.Namespace) -> None:
    if args.resume_from_artifacts and not args.artifacts_dir:
        raise ValueError("--resume-from-artifacts requires --artifacts-dir")
    if args.output_runbook_md and not args.artifacts_dir:
        raise ValueError("--output-runbook-md requires --artifacts-dir")
    if args.output_preflight_json and not args.preflight:
        raise ValueError("--output-preflight-json requires --preflight")
    if args.output_launch_script_sh and not args.artifacts_dir:
        raise ValueError("--output-launch-script-sh requires --artifacts-dir")

    selected_targets = _resolve_selected_targets(args.target)
    is_full_grid = len(selected_targets) == len(DEFAULT_PILOT_TARGETS)

    if not is_full_grid and (args.output_json or args.output_md or args.output_finalists_json or args.output_finalists_md):
        raise ValueError(
            "partial --target runs cannot emit ranking/finalist artifacts; run full grid or omit artifact outputs"
        )

    if args.output_launch_manifest_json:
        launch_manifest = _build_launch_manifest(
            args=args,
            selected_targets=selected_targets,
            is_full_grid=is_full_grid,
        )
        _ensure_parent_dir(args.output_launch_manifest_json)
        with open(args.output_launch_manifest_json, "w", encoding="utf-8") as f:
            json.dump(launch_manifest, f, indent=2)

    if args.output_launch_script_sh:
        launch_script = _render_launch_script(args=args, selected_targets=selected_targets)
        _ensure_parent_dir(args.output_launch_script_sh)
        with open(args.output_launch_script_sh, "w", encoding="utf-8") as f:
            f.write(launch_script)

    ranked_json_path = args.output_json or os.path.join(args.artifacts_dir, "pilot_ranked_runs.json")
    ranking_md_path = args.output_md or os.path.join(args.artifacts_dir, "pilot_ranking.md")
    finalists_json_path = args.output_finalists_json or os.path.join(args.artifacts_dir, "stage2_finalists.json")
    finalists_md_path = args.output_finalists_md or os.path.join(args.artifacts_dir, "stage2_finalists.md")

    if args.preflight:
        receipt = _run_preflight(args=args, selected_targets=selected_targets, is_full_grid=is_full_grid)
        if args.output_preflight_json:
            _ensure_parent_dir(args.output_preflight_json)
            with open(args.output_preflight_json, "w", encoding="utf-8") as f:
                json.dump(receipt, f, indent=2)
        status = "ok" if receipt["ok"] else "fail"
        print(f"pilot_sweep_preflight: {status}")
        if not receipt["ok"]:
            receipt_errors = receipt["errors"]
            assert isinstance(receipt_errors, list)
            failures = "\n".join(f"- {msg}" for msg in receipt_errors)
            raise ValueError(f"pilot sweep preflight failed:\n{failures}")
        return

    runs = []

    for index, target in selected_targets:
        command, num_iterations = build_pilot_command(
            target=target,
            python_exe=args.python_exe,
            max_seq_len=args.max_seq_len,
            total_batch_size=args.total_batch_size,
            device_batch_size=args.device_batch_size,
            pilot_tokens=args.pilot_tokens,
            eval_every=args.eval_every,
            eval_tokens=args.eval_tokens,
            device_type=args.device_type,
            extra_args=args.extra_arg,
        )
        run_result = {
            "config": target.label,
            "depth": target.depth,
            "n_branches": target.n_branches,
            "aspect_ratio": target.aspect_ratio,
            "num_iterations": num_iterations,
            "token_budget": num_iterations * args.total_batch_size,
            "command": command,
        }
        if args.resume_from_artifacts:
            loaded_run = _load_existing_run_artifact(
                artifacts_dir=args.artifacts_dir,
                run_index=index,
                config_label=target.label,
            )
            if loaded_run is not None:
                _validate_resume_run_artifact(
                    loaded_run,
                    artifacts_dir=args.artifacts_dir,
                    run_index=index,
                    config_label=target.label,
                    expected_token_budget=run_result["token_budget"],
                )
                run_result.update(loaded_run)
                runs.append(run_result)
                print(f"resume: using existing artifacts for {target.label}")
                continue

        if args.dry_run:
            runs.append(run_result)
            print(shlex.join(command))
            continue

        output, metrics = run_single_pilot(command)
        run_result.update(metrics)
        if metrics.get("command_failed"):
            print(f"warning: pilot run {target.label} exited non-zero and was marked unstable")

        if args.artifacts_dir:
            _write_run_artifacts(
                artifacts_dir=args.artifacts_dir,
                run_index=index,
                run_result=run_result,
                output_text=output,
            )

        runs.append(run_result)

    if args.dry_run:
        if args.output_runbook_md:
            runbook = _render_pilot_sweep_runbook(
                args=args,
                ranked_json_path=ranked_json_path,
                ranking_md_path=ranking_md_path,
                finalists_json_path=finalists_json_path,
                finalists_md_path=finalists_md_path,
            )
            _ensure_parent_dir(args.output_runbook_md)
            with open(args.output_runbook_md, "w", encoding="utf-8") as f:
                f.write(runbook)
        return

    if not is_full_grid:
        print("partial sweep complete: skipping ranking/finalist generation for --target subset run")
        return

    ranked = apply_ranking_rule(
        runs,
        slowdown_threshold_pct=args.slowdown_threshold_pct,
        clear_bpb_gain=args.clear_bpb_gain,
    )
    finalists = select_finalists(ranked, max_finalists=args.max_finalists)
    ranking_table = format_ranking_table(ranked)
    finalists_summary = format_finalists_summary(ranked, max_finalists=args.max_finalists)
    print(ranking_table)
    print()
    print(finalists_summary)

    ranked_payload = {
        "max_seq_len": args.max_seq_len,
        "total_batch_size": args.total_batch_size,
        "device_batch_size": args.device_batch_size,
        "pilot_tokens": args.pilot_tokens,
        "eval_every": args.eval_every,
        "eval_tokens": args.eval_tokens,
        "slowdown_threshold_pct": args.slowdown_threshold_pct,
        "clear_bpb_gain": args.clear_bpb_gain,
        "ranked_runs": ranked,
    }
    ranked_source_sha256 = _stable_json_sha256(ranked_payload)

    if args.output_json:
        _ensure_parent_dir(args.output_json)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(ranked_payload, f, indent=2)

    if args.output_md:
        lines = [
            "## Pilot Sweep Ranking",
            "",
            ranking_table,
            "",
            "## Finalists",
            "",
            finalists_summary,
            "",
        ]
        _ensure_parent_dir(args.output_md)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    if args.output_finalists_json:
        payload = {
            "source": ranked_json_path,
            "source_sha256": ranked_source_sha256,
            "max_finalists": args.max_finalists,
            "selected_finalists": finalists,
        }
        _ensure_parent_dir(args.output_finalists_json)
        with open(args.output_finalists_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if args.output_finalists_md:
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
        _ensure_parent_dir(args.output_finalists_md)
        with open(args.output_finalists_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    if args.output_runbook_md:
        runbook = _render_pilot_sweep_runbook(
            args=args,
            ranked_json_path=ranked_json_path,
            ranking_md_path=ranking_md_path,
            finalists_json_path=finalists_json_path,
            finalists_md_path=finalists_md_path,
        )
        _ensure_parent_dir(args.output_runbook_md)
        with open(args.output_runbook_md, "w", encoding="utf-8") as f:
            f.write(runbook)


def main() -> None:
    args = _parse_args()
    try:
        _run(args)
    except (RuntimeError, ValueError) as exc:
        if args.output_blocked_md:
            _write_blocked_markdown(
                output_path=args.output_blocked_md,
                mode="preflight" if args.preflight else "run",
                reason=str(exc),
                args=args,
            )
        raise


if __name__ == "__main__":
    main()
