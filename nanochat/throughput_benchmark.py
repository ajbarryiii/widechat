import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ThroughputTarget:
    label: str
    depth: int
    n_branches: int
    aspect_ratio: int


DEFAULT_TARGETS = (
    ThroughputTarget(label="12x1", depth=12, n_branches=1, aspect_ratio=64),
    ThroughputTarget(label="2x5", depth=2, n_branches=5, aspect_ratio=384),
    ThroughputTarget(label="1x10", depth=1, n_branches=10, aspect_ratio=768),
)


def build_train_command(
    target: ThroughputTarget,
    python_exe: str,
    max_seq_len: int,
    total_batch_size: int,
    device_batch_size: int,
    num_iterations: int,
    device_type: str,
    extra_args: list[str],
) -> list[str]:
    command = [
        python_exe,
        "-m",
        "scripts.base_train",
        "--run",
        "dummy",
        "--depth",
        str(target.depth),
        "--n-branches",
        str(target.n_branches),
        "--aspect-ratio",
        str(target.aspect_ratio),
        "--max-seq-len",
        str(max_seq_len),
        "--total-batch-size",
        str(total_batch_size),
        "--device-batch-size",
        str(device_batch_size),
        "--num-iterations",
        str(num_iterations),
        "--target-param-data-ratio",
        "-1",
        "--eval-every",
        "-1",
        "--core-metric-every",
        "-1",
        "--sample-every",
        "-1",
        "--save-every",
        "-1",
        "--model-tag",
        f"throughput_d{target.depth}b{target.n_branches}",
        "--window-pattern",
        "L",
    ]
    if device_type:
        command.extend(["--device-type", device_type])
    command.extend(extra_args)
    return command


def _extract_last_number(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    return int(matches[-1].replace(",", ""))


def _extract_last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    return float(matches[-1])


def parse_train_output(output_text: str) -> dict[str, int | float | None]:
    avg_tok_per_sec = _extract_last_number(r"Average tok/sec \(post-warmup\): ([0-9,]+)", output_text)
    final_tok_per_sec = _extract_last_number(r"tok/sec: ([0-9,]+)", output_text)
    peak_memory_mib = _extract_last_float(r"Peak memory usage: ([0-9]+(?:\.[0-9]+)?)MiB", output_text)
    final_mfu = _extract_last_float(r"bf16_mfu: ([0-9]+(?:\.[0-9]+)?)", output_text)

    selected_tok_per_sec = avg_tok_per_sec if avg_tok_per_sec is not None else final_tok_per_sec
    if selected_tok_per_sec is None:
        raise ValueError("Could not parse throughput metrics from base_train output")

    return {
        "avg_tok_per_sec": avg_tok_per_sec,
        "final_tok_per_sec": final_tok_per_sec,
        "selected_tok_per_sec": selected_tok_per_sec,
        "final_mfu": final_mfu,
        "peak_memory_mib": peak_memory_mib,
    }


def run_single_target(command: list[str]) -> tuple[str, dict[str, int | float | None]]:
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-60:])
        raise RuntimeError(f"Benchmark command failed ({proc.returncode}): {' '.join(command)}\n{tail}")
    return output, parse_train_output(output)


def format_markdown_table(rows: list[dict[str, str]]) -> str:
    headers = ["Config", "tok/sec", "vs 12x1", "MFU", "Peak mem (MiB)"]
    output_lines = ["| " + " | ".join(headers) + " |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        output_lines.append(
            "| " + " | ".join([
                row["Config"],
                row["tok/sec"],
                row["vs 12x1"],
                row["MFU"],
                row["Peak mem (MiB)"],
            ]) + " |"
        )
    return "\n".join(output_lines)
