"""Offline validation for a recorded Blackwell smoke artifact bundle.

Example:
python -m scripts.check_blackwell_evidence_bundle --bundle-dir artifacts/blackwell_smoke --expect-backend fa4
"""

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

from scripts.validate_blackwell_smoke_artifact import (
    _load_artifact,
    _load_status_line,
    _validate_artifact,
    _validate_status_line_consistency,
)


_REQUIRED_BUNDLE_FILES = (
    "flash_backend_smoke.json",
    "flash_backend_status.log",
    "blackwell_smoke_evidence.md",
    "blackwell_smoke_runbook.md",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate recorded Blackwell smoke bundle directory")
    parser.add_argument(
        "--bundle-dir",
        default="auto",
        help="directory emitted by scripts.run_blackwell_smoke_bundle, or 'auto' to discover latest real bundle",
    )
    parser.add_argument(
        "--bundle-root",
        default="artifacts/blackwell",
        help="bundle search root used when --bundle-dir=auto",
    )
    parser.add_argument("--expect-backend", choices=["fa4", "fa3", "sdpa"], default="fa4")
    parser.add_argument(
        "--require-blackwell",
        action="store_true",
        help="require artifact CUDA capability to be sm100+",
    )
    parser.add_argument(
        "--require-device-substring",
        default="",
        help=(
            "optional case-insensitive device_name substring required in artifact metadata "
            "(for example 'RTX 5090'); check-in mode defaults to RTX 5090"
        ),
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only resolve bundle/check settings and print planned validation",
    )
    return parser.parse_args()


def _required_paths(bundle_dir: Path) -> dict[str, Path]:
    return {
        "artifact_json": bundle_dir / "flash_backend_smoke.json",
        "status_line": bundle_dir / "flash_backend_status.log",
        "evidence_md": bundle_dir / "blackwell_smoke_evidence.md",
        "runbook_md": bundle_dir / "blackwell_smoke_runbook.md",
    }


def _classify_bundle_dir(bundle_dir: Path) -> str:
    if any("sample" in part.lower() for part in bundle_dir.parts):
        return "sample path segment"

    missing_files = [name for name in _REQUIRED_BUNDLE_FILES if not (bundle_dir / name).is_file()]
    if missing_files:
        return f"missing files: {', '.join(missing_files)}"

    artifact_path = bundle_dir / "flash_backend_smoke.json"
    try:
        payload = _load_artifact(str(artifact_path))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        return f"invalid flash_backend_smoke.json: {exc}"

    if payload.get("is_sample") is True:
        return "payload marked is_sample=true"

    return "real"


def _render_no_real_bundle_error(
    *,
    bundle_root: Path,
    rejected_dirs: list[tuple[Path, str]],
) -> str:
    lines = [
        f"no real Blackwell bundle found under {bundle_root}; run scripts.run_blackwell_smoke_bundle on RTX 5090 first",
        "discovery searched for 'flash_backend_smoke.json' files and "
        f"rejected {len(rejected_dirs)} candidate bundle(s)",
    ]
    for rejected_path, reason in rejected_dirs[:5]:
        lines.append(f"- {rejected_path}: {reason}")
    if len(rejected_dirs) > 5:
        lines.append(f"- ... {len(rejected_dirs) - 5} more candidate bundle(s) omitted")
    return "\n".join(lines)


def _resolve_bundle_dir(bundle_dir_arg: str, bundle_root_arg: str) -> Path:
    if bundle_dir_arg != "auto":
        return Path(bundle_dir_arg)

    bundle_root = Path(bundle_root_arg)
    if not bundle_root.is_dir():
        raise RuntimeError(
            f"bundle_root does not exist: {bundle_root}; pass --bundle-dir explicitly or emit a bundle first"
        )

    discovered_dirs = sorted({path.parent for path in bundle_root.rglob("flash_backend_smoke.json") if path.is_file()})
    candidates: list[Path] = []
    rejected_dirs: list[tuple[Path, str]] = []
    for discovered_dir in discovered_dirs:
        classification = _classify_bundle_dir(discovered_dir)
        if classification == "real":
            candidates.append(discovered_dir)
        else:
            rejected_dirs.append((discovered_dir, classification))

    if not candidates:
        raise RuntimeError(
            _render_no_real_bundle_error(
                bundle_root=bundle_root,
                rejected_dirs=rejected_dirs,
            )
        )

    candidates.sort(key=lambda path: (path / "flash_backend_smoke.json").stat().st_mtime, reverse=True)
    return candidates[0]


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


def _extract_evidence_value(evidence_text: str, key: str) -> str:
    prefix = f"- {key}: `"
    for raw_line in evidence_text.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix) and line.endswith("`"):
            return line[len(prefix) : -1]
    raise RuntimeError(f"evidence markdown missing line: - {key}: `...`")


def _assert_evidence_content(
    evidence_text: str,
    selected_backend: str,
    generated_at_utc: str,
    git_commit: str,
) -> None:
    if "# Blackwell Flash Backend Smoke Evidence" not in evidence_text:
        raise RuntimeError("evidence markdown missing line: # Blackwell Flash Backend Smoke Evidence")

    evidence_selected_backend = _extract_evidence_value(evidence_text, "selected_backend")
    if evidence_selected_backend != selected_backend:
        raise RuntimeError(
            "evidence markdown selected_backend mismatch: "
            f"expected {selected_backend}, got {evidence_selected_backend}"
        )

    evidence_generated_at = _extract_evidence_value(evidence_text, "generated_at_utc")
    if evidence_generated_at != generated_at_utc:
        raise RuntimeError(
            "evidence markdown generated_at_utc mismatch: "
            f"expected {generated_at_utc}, got {evidence_generated_at}"
        )

    evidence_git_commit = _extract_evidence_value(evidence_text, "git_commit")
    if evidence_git_commit != git_commit:
        raise RuntimeError(
            "evidence markdown git_commit mismatch: "
            f"expected {git_commit}, got {evidence_git_commit}"
        )

    evidence_status_line_ok = _extract_evidence_value(evidence_text, "status_line_ok")
    if evidence_status_line_ok.lower() != "true":
        raise RuntimeError("evidence markdown status_line_ok mismatch: expected true")


def _assert_real_bundle_dir(bundle_dir: Path) -> None:
    if "sample_bundle" in bundle_dir.parts:
        raise RuntimeError(
            "bundle_dir points to sample fixture artifacts; use emitted RTX 5090 bundle artifacts"
        )


def _assert_real_bundle_payload(payload: dict[str, object], artifact_path: Path) -> None:
    if payload.get("is_sample") is True:
        raise RuntimeError(
            "bundle artifact payload is marked as sample fixture; use emitted RTX 5090 bundle artifacts: "
            f"{artifact_path}"
        )


def _assert_runbook_content(runbook_text: str, bundle_dir: Path, expect_backend: str) -> None:
    expected_path = str(bundle_dir)
    quoted_expected_path = shlex.quote(expected_path)
    expected_evidence = str(bundle_dir / "blackwell_smoke_evidence.md")
    expected_check_json = str(bundle_dir / "blackwell_bundle_check.json")
    quoted_expect_backend = shlex.quote(expect_backend)
    require_device_substring = "RTX 5090"
    quoted_require_device_substring = shlex.quote(require_device_substring)
    quoted_check_json = shlex.quote(expected_check_json)
    expected_snippets = [
        "# Blackwell Smoke Bundle Runbook",
        "python -m scripts.run_blackwell_smoke_bundle",
        "--output-check-json",
        expected_evidence,
        f"- `{expected_evidence}`",
        f"- Ensure command prints `bundle_ok selected={expect_backend}`.",
    ]
    for snippet in expected_snippets:
        if snippet not in runbook_text:
            raise RuntimeError(f"runbook markdown missing snippet: {snippet}")

    output_dir_snippets = [
        f"--output-dir {expected_path}",
        f"--output-dir {quoted_expected_path}",
    ]
    if not any(snippet in runbook_text for snippet in output_dir_snippets):
        raise RuntimeError(f"runbook markdown missing snippet: {output_dir_snippets[0]}")

    expect_backend_snippets = [
        f"--expect-backend {expect_backend}",
        f"--expect-backend {quoted_expect_backend}",
    ]
    if not any(snippet in runbook_text for snippet in expect_backend_snippets):
        raise RuntimeError(f"runbook markdown missing snippet: {expect_backend_snippets[0]}")

    check_json_snippets = [expected_check_json, quoted_check_json]
    if not any(snippet in runbook_text for snippet in check_json_snippets):
        raise RuntimeError(f"runbook markdown missing snippet: {expected_check_json}")

    require_device_snippets = [
        f"--require-device-substring {require_device_substring}",
        f"--require-device-substring {quoted_require_device_substring}",
    ]
    if not any(snippet in runbook_text for snippet in require_device_snippets):
        raise RuntimeError(f"runbook markdown missing snippet: {require_device_snippets[0]}")

    has_direct_command = "python -m scripts.check_blackwell_evidence_bundle --bundle-dir" in runbook_text
    has_helper_command = "python -m scripts.run_blackwell_check_in --bundle-dir" in runbook_text
    if not has_direct_command and not has_helper_command:
        raise RuntimeError(
            "runbook markdown missing snippet: "
            "python -m scripts.check_blackwell_evidence_bundle --bundle-dir"
        )
    if has_direct_command and "--check-in" not in runbook_text:
        raise RuntimeError("runbook markdown missing snippet: --check-in")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_check_report(
    path: str,
    *,
    bundle_dir: Path,
    paths: dict[str, Path],
    expect_backend: str,
    selected_backend: str,
    check_in: bool,
    require_blackwell: bool,
    require_git_tracked: bool,
    require_real_bundle: bool,
    require_device_substring: str,
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
        "require_real_bundle": require_real_bundle,
        "require_device_substring": require_device_substring,
        "artifact_sha256": {
            label: _sha256_file(file_path)
            for label, file_path in paths.items()
        },
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
    require_device_substring: str,
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
    if require_real_bundle:
        _assert_real_bundle_payload(payload, paths["artifact_json"])
    selected_backend, _capability = _validate_artifact(
        payload,
        expect_backend,
        effective_require_blackwell,
        require_device_substring,
    )

    status_line = _load_status_line(str(paths["status_line"]))
    status_backend = _validate_status_line_consistency(payload, status_line)
    if status_backend != expect_backend:
        raise RuntimeError(f"expected backend {expect_backend}, got {status_backend} in status-line file")

    evidence_text = paths["evidence_md"].read_text(encoding="utf-8")
    _assert_evidence_content(
        evidence_text,
        selected_backend,
        str(payload.get("generated_at_utc") or ""),
        str(payload.get("git_commit") or ""),
    )

    runbook_text = paths["runbook_md"].read_text(encoding="utf-8")
    _assert_runbook_content(runbook_text, bundle_dir, expect_backend)

    if output_check_json:
        _write_check_report(
            output_check_json,
            bundle_dir=bundle_dir,
            paths=paths,
            expect_backend=expect_backend,
            selected_backend=selected_backend,
            check_in=check_in,
            require_blackwell=effective_require_blackwell,
            require_git_tracked=effective_require_git_tracked,
            require_real_bundle=require_real_bundle,
            require_device_substring=require_device_substring,
        )

    return selected_backend


def main() -> None:
    args = _parse_args()
    bundle_dir = _resolve_bundle_dir(args.bundle_dir, args.bundle_root)
    effective_require_blackwell = args.require_blackwell or args.check_in
    effective_require_git_tracked = args.require_git_tracked or args.check_in
    effective_require_device_substring = args.require_device_substring or ("RTX 5090" if args.check_in else "")

    if args.dry_run:
        print(
            "bundle_check_dry_run_ok "
            f"bundle_dir={bundle_dir} "
            f"expect_backend={args.expect_backend} "
            f"check_in={args.check_in} "
            f"require_blackwell={effective_require_blackwell} "
            f"require_git_tracked={effective_require_git_tracked} "
            f"require_real_bundle={args.require_real_bundle} "
            f"require_device_substring={effective_require_device_substring or '<none>'} "
            f"output_check_json={args.output_check_json or '<none>'}"
        )
        return

    selected_backend = run_bundle_check(
        bundle_dir=bundle_dir,
        expect_backend=args.expect_backend,
        check_in=args.check_in,
        require_blackwell=args.require_blackwell,
        require_git_tracked=args.require_git_tracked,
        require_real_bundle=args.require_real_bundle,
        require_device_substring=effective_require_device_substring,
        output_check_json=args.output_check_json,
    )

    print(f"bundle_check_ok selected={selected_backend} bundle_dir={bundle_dir}")


if __name__ == "__main__":
    main()
