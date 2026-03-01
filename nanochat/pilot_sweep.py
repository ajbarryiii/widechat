import re
import subprocess
from dataclasses import dataclass

from nanochat.throughput_benchmark import parse_train_output


@dataclass(frozen=True)
class PilotTarget:
    label: str
    depth: int
    n_branches: int
    aspect_ratio: int


DEFAULT_PILOT_TARGETS = (
    PilotTarget(label="12x1", depth=12, n_branches=1, aspect_ratio=64),
    PilotTarget(label="6x2", depth=6, n_branches=2, aspect_ratio=128),
    PilotTarget(label="4x3", depth=4, n_branches=3, aspect_ratio=192),
    PilotTarget(label="3x4", depth=3, n_branches=4, aspect_ratio=256),
    PilotTarget(label="2x5", depth=2, n_branches=5, aspect_ratio=384),
    PilotTarget(label="2x6", depth=2, n_branches=6, aspect_ratio=384),
    PilotTarget(label="1x10", depth=1, n_branches=10, aspect_ratio=768),
)

MIN_RECOMMENDED_EVAL_EVERY = 50
MAX_RECOMMENDED_EVAL_EVERY = 100


def _as_float(value: int | float | bool | str | None, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric, got {value!r}")
    return float(value)


def _as_optional_float(value: int | float | bool | str | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def build_pilot_command(
    target: PilotTarget,
    python_exe: str,
    max_seq_len: int,
    total_batch_size: int,
    device_batch_size: int,
    pilot_tokens: int,
    eval_every: int,
    eval_tokens: int,
    device_type: str,
    extra_args: list[str],
) -> tuple[list[str], int]:
    if total_batch_size <= 0:
        raise ValueError("total_batch_size must be > 0")
    if pilot_tokens < total_batch_size:
        raise ValueError("pilot_tokens must be >= total_batch_size")
    if eval_every <= 0:
        raise ValueError("eval_every must be > 0")
    if eval_every < MIN_RECOMMENDED_EVAL_EVERY or eval_every > MAX_RECOMMENDED_EVAL_EVERY:
        raise ValueError(
            "eval_every must be between "
            f"{MIN_RECOMMENDED_EVAL_EVERY} and {MAX_RECOMMENDED_EVAL_EVERY} "
            "to keep pilot ranking comparable"
        )
    if eval_tokens <= 0:
        raise ValueError("eval_tokens must be > 0")

    num_iterations = pilot_tokens // total_batch_size
    if num_iterations < eval_every:
        raise ValueError(
            "pilot_tokens budget is too small for eval_every; "
            "need at least one in-training validation point"
        )
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
        str(eval_every),
        "--eval-tokens",
        str(eval_tokens),
        "--core-metric-every",
        "-1",
        "--sample-every",
        "-1",
        "--save-every",
        "-1",
        "--model-tag",
        f"pilot_d{target.depth}b{target.n_branches}",
        "--window-pattern",
        "L",
    ]
    if device_type:
        command.extend(["--device-type", device_type])
    command.extend(extra_args)
    return command, num_iterations


def extract_val_bpb_trace(output_text: str) -> list[float]:
    return [float(value) for value in re.findall(r"Validation bpb: ([0-9]+(?:\.[0-9]+)?)", output_text)]


def summarize_pilot_output(output_text: str) -> dict[str, int | float | bool | None]:
    parse_failed = False
    try:
        throughput_metrics = parse_train_output(output_text)
    except ValueError:
        parse_failed = True
        throughput_metrics = {
            "avg_tok_per_sec": None,
            "final_tok_per_sec": None,
            "selected_tok_per_sec": 0,
            "final_mfu": None,
            "peak_memory_mib": None,
        }
    val_trace = extract_val_bpb_trace(output_text)
    final_val_bpb = val_trace[-1] if val_trace else None
    min_val_bpb = min(val_trace) if val_trace else None

    if min_val_bpb is None:
        min_match = re.findall(r"Minimum validation bpb: ([0-9]+(?:\.[0-9]+)?)", output_text)
        if min_match:
            min_val_bpb = float(min_match[-1])

    unstable = parse_failed or bool(re.search(r"\b(?:nan|inf)\b", output_text, flags=re.IGNORECASE))
    return {
        **throughput_metrics,
        "final_val_bpb": final_val_bpb,
        "min_val_bpb": min_val_bpb,
        "unstable": unstable,
    }


def run_single_pilot(command: list[str]) -> tuple[str, dict[str, int | float | bool | None]]:
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    metrics = summarize_pilot_output(output)
    if proc.returncode != 0:
        metrics["unstable"] = True
        metrics["command_failed"] = True
        metrics["failure_returncode"] = proc.returncode
    else:
        metrics["command_failed"] = False
        metrics["failure_returncode"] = None
    return output, metrics


def apply_ranking_rule(
    runs: list[dict[str, int | float | bool | str | None]],
    slowdown_threshold_pct: float = 5.0,
    clear_bpb_gain: float = 0.02,
) -> list[dict[str, int | float | bool | str | None]]:
    baseline = next(run for run in runs if run["config"] == "12x1")
    baseline_tok = _as_float(baseline.get("selected_tok_per_sec"), "selected_tok_per_sec")
    if baseline_tok <= 0:
        raise ValueError("baseline selected_tok_per_sec must be > 0")
    baseline_bpb = _as_optional_float(baseline.get("min_val_bpb"))

    ranked: list[dict[str, int | float | bool | str | None]] = []
    for run in runs:
        tok_per_sec = _as_float(run.get("selected_tok_per_sec"), "selected_tok_per_sec")
        slowdown_pct = 100.0 * (1.0 - tok_per_sec / baseline_tok)
        unstable = bool(run.get("unstable", False))
        baseline_token_budget = baseline.get("token_budget")
        run_token_budget = run.get("token_budget")
        min_val_bpb = _as_optional_float(run.get("min_val_bpb"))
        clearly_better = (
            baseline_bpb is not None
            and min_val_bpb is not None
            and min_val_bpb <= baseline_bpb - clear_bpb_gain
        )

        disqualify_reason = None
        if unstable:
            disqualify_reason = "unstable"
        elif baseline_token_budget is not None and run_token_budget != baseline_token_budget:
            disqualify_reason = "token-budget-mismatch"
        elif slowdown_pct > slowdown_threshold_pct and not clearly_better:
            disqualify_reason = f"slow>{slowdown_threshold_pct:.1f}%"

        ranked.append(
            {
                **run,
                "slowdown_pct": slowdown_pct,
                "qualified": disqualify_reason is None,
                "disqualify_reason": disqualify_reason,
            }
        )

    ranked.sort(
        key=lambda row: (
            not bool(row["qualified"]),
            float("inf") if _as_optional_float(row.get("min_val_bpb")) is None else _as_float(row.get("min_val_bpb"), "min_val_bpb"),
            -_as_float(row.get("selected_tok_per_sec"), "selected_tok_per_sec"),
        )
    )

    rank = 1
    for row in ranked:
        if row["qualified"]:
            row["rank"] = rank
            rank += 1
        else:
            row["rank"] = None
    return ranked


def format_ranking_table(rows: list[dict[str, int | float | bool | str | None]]) -> str:
    baseline_row = next(row for row in rows if row["config"] == "12x1")
    baseline_tok = _as_float(baseline_row.get("selected_tok_per_sec"), "selected_tok_per_sec")
    output_lines = [
        "| Rank | Config | tok/sec | vs 12x1 | min val bpb | token budget | Status |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        status = "qualified"
        if not row["qualified"]:
            status = f"disqualified ({row['disqualify_reason']})"
        selected_tok_per_sec = _as_float(row.get("selected_tok_per_sec"), "selected_tok_per_sec")
        ratio_pct = 100.0 * (selected_tok_per_sec / baseline_tok - 1.0)
        sign = "+" if ratio_pct >= 0 else ""
        min_val_bpb = _as_optional_float(row.get("min_val_bpb"))
        token_budget = row.get("token_budget")
        output_lines.append(
            "| "
            + " | ".join(
                [
                    "-" if row["rank"] is None else str(row["rank"]),
                    str(row["config"]),
                    f"{int(selected_tok_per_sec):,}",
                    f"{sign}{ratio_pct:.1f}%",
                    "n/a" if min_val_bpb is None else f"{min_val_bpb:.4f}",
                    "n/a" if token_budget is None else f"{int(token_budget):,}",
                    status,
                ]
            )
            + " |"
        )
    return "\n".join(output_lines)


def select_finalists(
    rows: list[dict[str, int | float | bool | str | None]],
    max_finalists: int = 3,
) -> list[dict[str, int | float | bool | str | None]]:
    if max_finalists <= 0:
        raise ValueError("max_finalists must be > 0")
    qualified = [row for row in rows if bool(row.get("qualified", False))]
    return qualified[:max_finalists]


def format_finalists_summary(
    rows: list[dict[str, int | float | bool | str | None]],
    max_finalists: int = 3,
) -> str:
    finalists = select_finalists(rows, max_finalists=max_finalists)
    if not finalists:
        return "No qualified finalists were selected."
    baseline_row = next(row for row in rows if row["config"] == "12x1")
    baseline_tok = _as_float(baseline_row.get("selected_tok_per_sec"), "selected_tok_per_sec")
    output_lines = ["Selected finalists:"]
    for row in finalists:
        selected_tok_per_sec = _as_float(row.get("selected_tok_per_sec"), "selected_tok_per_sec")
        ratio_pct = 100.0 * (selected_tok_per_sec / baseline_tok - 1.0)
        sign = "+" if ratio_pct >= 0 else ""
        min_val_bpb = _as_optional_float(row.get("min_val_bpb"))
        output_lines.append(
            f"- {row['config']}: rank={row['rank']}, tok/sec={int(selected_tok_per_sec):,} "
            f"({sign}{ratio_pct:.1f}% vs 12x1), "
            f"min_val_bpb={'n/a' if min_val_bpb is None else f'{min_val_bpb:.4f}'}"
        )
    return "\n".join(output_lines)
