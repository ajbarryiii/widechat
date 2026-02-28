"""Offline validation for a recorded Blackwell smoke artifact bundle.

Example:
python -m scripts.check_blackwell_evidence_bundle --bundle-dir artifacts/blackwell_smoke --expect-backend fa4
"""

import argparse
import json
import subprocess
from pathlib import Path

from scripts.validate_blackwell_smoke_artifact import (
    _load_artifact,
    _load_status_line,
    _validate_artifact,
    _validate_status_line_consistency,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate recorded Blackwell smoke bundle directory")
    parser.add_argument(
        "--bundle-dir",
        default="artifacts/blackwell_smoke",
        help="directory emitted by scripts.run_blackwell_smoke_bundle",
    )
    parser.add_argument("--expect-backend", choices=["fa4", "fa3", "sdpa"], default="fa4")
    parser.add_argument(
        "--require-blackwell",
        action="store_true",
        help="require artifact CUDA capability to be sm100+",
    )
    parser.add_argument(
        "--require-git-tracked",
        action="store_true",
        help="require bundle artifacts to be tracked by git (for check-in verification)",
    )
    parser.add_argument(
        "--check-in",
        action="store_true",
        help="enable strict check-in mode (requires Blackwell and git-tracked artifacts)",
    )
    parser.add_argument(
        "--require-real-bundle",
        action="store_true",
        help="reject sample fixture bundles and require real emitted artifacts",
    )
    parser.add_argument(
        "--output-check-json",
        default="",
        help="optional path for machine-readable bundle-check receipt",
    )
    return parser.parse_args()


def _required_paths(bundle_dir: Path) -> dict[str, Path]:
    return {
        "artifact_json": bundle_dir / "flash_backend_smoke.json",
        "status_line": bundle_dir / "flash_backend_status.log",
        "evidence_md": bundle_dir / "blackwell_smoke_evidence.md",
        "runbook_md": bundle_dir / "blackwell_smoke_runbook.md",
    }


def _assert_files_exist(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing {label} file: {path}")


def _assert_git_tracked(paths: dict[str, Path], bundle_dir: Path) -> None:
    for path in paths.values():
        cmd = ["git", "-C", str(bundle_dir), "ls-files", "--error-unmatch", path.name]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"bundle artifact is not git-tracked: {path}")


def _assert_evidence_content(evidence_text: str, selected_backend: str) -> None:
    required_lines = [
        "# Blackwell Flash Backend Smoke Evidence",
        f"- selected_backend: `{selected_backend}`",
        "- status_line_ok: `true`",
    ]
    for line in required_lines:
        if line not in evidence_text:
            raise RuntimeError(f"evidence markdown missing line: {line}")


def _assert_real_bundle_dir(bundle_dir: Path) -> None:
    if "sample_bundle" in bundle_dir.parts:
        raise RuntimeError(
            "bundle_dir points to sample fixture artifacts; use emitted RTX 5090 bundle artifacts"
        )


def _assert_runbook_content(runbook_text: str, bundle_dir: Path, expect_backend: str) -> None:
    expected_path = str(bundle_dir)
    expected_evidence = str(bundle_dir / "blackwell_smoke_evidence.md")
    expected_check_json = str(bundle_dir / "blackwell_bundle_check.json")
    expected_snippets = [
        "# Blackwell Smoke Bundle Runbook",
        "python -m scripts.run_blackwell_smoke_bundle",
        f"--output-dir {expected_path}",
        f"--expect-backend {expect_backend}",
        "--output-check-json",
        expected_check_json,
        f"- `{expected_evidence}`",
        f"- Ensure command prints `bundle_ok selected={expect_backend}`.",
    ]
    for snippet in expected_snippets:
        if snippet not in runbook_text:
            raise RuntimeError(f"runbook markdown missing snippet: {snippet}")

    has_direct_command = "python -m scripts.check_blackwell_evidence_bundle --bundle-dir" in runbook_text
    has_helper_command = "python -m scripts.run_blackwell_check_in --bundle-dir" in runbook_text
    if not has_direct_command and not has_helper_command:
        raise RuntimeError(
            "runbook markdown missing snippet: "
            "python -m scripts.check_blackwell_evidence_bundle --bundle-dir"
        )
    if has_direct_command and "--check-in" not in runbook_text:
        raise RuntimeError("runbook markdown missing snippet: --check-in")


def _write_check_report(
    path: str,
    *,
    bundle_dir: Path,
    expect_backend: str,
    selected_backend: str,
    check_in: bool,
    require_blackwell: bool,
    require_git_tracked: bool,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bundle_dir": str(bundle_dir),
        "expect_backend": expect_backend,
        "selected_backend": selected_backend,
        "check_in": check_in,
        "require_blackwell": require_blackwell,
        "require_git_tracked": require_git_tracked,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_bundle_check(
    *,
    bundle_dir: Path,
    expect_backend: str,
    check_in: bool,
    require_blackwell: bool,
    require_git_tracked: bool,
    require_real_bundle: bool,
    output_check_json: str,
) -> str:
    effective_require_blackwell = require_blackwell or check_in
    effective_require_git_tracked = require_git_tracked or check_in

    paths = _required_paths(bundle_dir)
    _assert_files_exist(paths)
    if effective_require_git_tracked:
        _assert_git_tracked(paths, bundle_dir)
    if require_real_bundle:
        _assert_real_bundle_dir(bundle_dir)

    payload = _load_artifact(str(paths["artifact_json"]))
    selected_backend, _capability = _validate_artifact(payload, expect_backend, effective_require_blackwell)

    status_line = _load_status_line(str(paths["status_line"]))
    status_backend = _validate_status_line_consistency(payload, status_line)
    if status_backend != expect_backend:
        raise RuntimeError(f"expected backend {expect_backend}, got {status_backend} in status-line file")

    evidence_text = paths["evidence_md"].read_text(encoding="utf-8")
    _assert_evidence_content(evidence_text, selected_backend)

    runbook_text = paths["runbook_md"].read_text(encoding="utf-8")
    _assert_runbook_content(runbook_text, bundle_dir, expect_backend)

    if output_check_json:
        _write_check_report(
            output_check_json,
            bundle_dir=bundle_dir,
            expect_backend=expect_backend,
            selected_backend=selected_backend,
            check_in=check_in,
            require_blackwell=effective_require_blackwell,
            require_git_tracked=effective_require_git_tracked,
        )

    return selected_backend


def main() -> None:
    args = _parse_args()
    selected_backend = run_bundle_check(
        bundle_dir=Path(args.bundle_dir),
        expect_backend=args.expect_backend,
        check_in=args.check_in,
        require_blackwell=args.require_blackwell,
        require_git_tracked=args.require_git_tracked,
        require_real_bundle=args.require_real_bundle,
        output_check_json=args.output_check_json,
    )

    print(f"bundle_check_ok selected={selected_backend} bundle_dir={args.bundle_dir}")


if __name__ == "__main__":
    main()
