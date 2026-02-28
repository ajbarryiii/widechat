"""Run fixed-shape throughput benchmark across branch configs.

Example:
python -m scripts.throughput_benchmark --device-type cuda --max-seq-len 2048 --total-batch-size 524288 --device-batch-size 16 --num-iterations 40
"""

import argparse
import json
import shlex
import sys

from nanochat.throughput_benchmark import (
    DEFAULT_TARGETS,
    build_train_command,
    format_markdown_table,
    run_single_target,
)


def _format_ratio(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark 12x1 vs 2x5 vs 1x10 throughput")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = base_train autodetect)")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--total-batch-size", type=int, required=True)
    parser.add_argument("--device-batch-size", type=int, required=True)
    parser.add_argument("--num-iterations", type=int, default=40)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--output-json", type=str, default="", help="optional path to write machine-readable results")
    parser.add_argument("--extra-arg", action="append", default=[], help="forward extra arg to each base_train run")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runs = []

    for target in DEFAULT_TARGETS:
        command = build_train_command(
            target=target,
            python_exe=args.python_exe,
            max_seq_len=args.max_seq_len,
            total_batch_size=args.total_batch_size,
            device_batch_size=args.device_batch_size,
            num_iterations=args.num_iterations,
            device_type=args.device_type,
            extra_args=args.extra_arg,
        )

        run_result = {
            "config": target.label,
            "depth": target.depth,
            "n_branches": target.n_branches,
            "aspect_ratio": target.aspect_ratio,
            "command": command,
        }
        if args.dry_run:
            runs.append(run_result)
            print(shlex.join(command))
            continue

        _, metrics = run_single_target(command)
        run_result.update(metrics)
        runs.append(run_result)

    if args.dry_run:
        return

    baseline_tok = next(run["selected_tok_per_sec"] for run in runs if run["config"] == "12x1")
    table_rows = []
    for run in runs:
        tok_per_sec = int(run["selected_tok_per_sec"])
        ratio_pct = 100.0 * (tok_per_sec / baseline_tok - 1.0)
        final_mfu = run["final_mfu"]
        peak_memory = run["peak_memory_mib"]
        table_rows.append(
            {
                "Config": run["config"],
                "tok/sec": f"{tok_per_sec:,}",
                "vs 12x1": _format_ratio(ratio_pct),
                "MFU": "n/a" if final_mfu is None else f"{final_mfu:.2f}",
                "Peak mem (MiB)": "n/a" if peak_memory is None else f"{peak_memory:.2f}",
            }
        )

    print(format_markdown_table(table_rows))

    if args.output_json:
        payload = {
            "max_seq_len": args.max_seq_len,
            "total_batch_size": args.total_batch_size,
            "device_batch_size": args.device_batch_size,
            "num_iterations": args.num_iterations,
            "runs": runs,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
