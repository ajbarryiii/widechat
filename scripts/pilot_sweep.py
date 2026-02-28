"""Run a Stage 1 pilot sweep across depth x branch configs.

Example:
python -m scripts.pilot_sweep --device-type cuda --total-batch-size 524288 --device-batch-size 16
"""

import argparse
import json
import shlex
import sys

from nanochat.pilot_sweep import (
    DEFAULT_PILOT_TARGETS,
    apply_ranking_rule,
    build_pilot_command,
    format_ranking_table,
    run_single_pilot,
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
    parser.add_argument("--extra-arg", action="append", default=[], help="forward extra arg to each base_train run")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runs = []

    for target in DEFAULT_PILOT_TARGETS:
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
        if args.dry_run:
            runs.append(run_result)
            print(shlex.join(command))
            continue

        _, metrics = run_single_pilot(command)
        run_result.update(metrics)
        if metrics.get("command_failed"):
            print(f"warning: pilot run {target.label} exited non-zero and was marked unstable")
        runs.append(run_result)

    if args.dry_run:
        return

    ranked = apply_ranking_rule(
        runs,
        slowdown_threshold_pct=args.slowdown_threshold_pct,
        clear_bpb_gain=args.clear_bpb_gain,
    )
    print(format_ranking_table(ranked))

    if args.output_json:
        payload = {
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
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
