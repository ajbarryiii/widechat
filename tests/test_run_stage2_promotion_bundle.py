import json
import os
import shlex
import hashlib
from pathlib import Path

import pytest

from scripts import run_stage2_promotion_bundle as bundle
from scripts import pilot_promote


def test_main_writes_stage2_finalists_bundle(tmp_path, monkeypatch, capsys):
    input_json = tmp_path / "ranked_runs.json"
    output_dir = tmp_path / "artifacts"
    ranked_runs = {
        "ranked_runs": [
            {
                "config": "4x3",
                "depth": 4,
                "n_branches": 3,
                "aspect_ratio": 192,
                "selected_tok_per_sec": 572110.0,
                "min_val_bpb": 4.0123,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 1,
                "disqualify_reason": None,
            },
            {
                "config": "12x1",
                "depth": 12,
                "n_branches": 1,
                "aspect_ratio": 64,
                "selected_tok_per_sec": 565800.0,
                "min_val_bpb": 4.0310,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 2,
                "disqualify_reason": None,
            },
            {
                "config": "2x5",
                "depth": 2,
                "n_branches": 5,
                "aspect_ratio": 384,
                "selected_tok_per_sec": 525100.0,
                "min_val_bpb": 4.0388,
                "token_budget": 250000000,
                "qualified": False,
                "rank": None,
                "disqualify_reason": "slow>5.0%",
            },
        ]
    }
    input_json.write_text(json.dumps(ranked_runs), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(output_dir),
            "--max-finalists",
            "1",
            "--min-finalists",
            "1",
        ],
    )

    bundle.main()

    finalists_json = output_dir / "stage2_finalists.json"
    finalists_md = output_dir / "stage2_finalists.md"
    payload = json.loads(finalists_json.read_text(encoding="utf-8"))
    assert payload["source"] == str(input_json)
    assert payload["source_sha256"] == pilot_promote._stable_json_sha256(ranked_runs)
    assert payload["max_finalists"] == 1
    assert [row["config"] for row in payload["selected_finalists"]] == ["4x3"]

    body = finalists_md.read_text(encoding="utf-8")
    assert "## Stage 2 Finalists" in body
    assert "`4x3`: `--depth 4 --n-branches 3 --aspect-ratio 192`" in body
    assert "`12x1`: `--depth 12 --n-branches 1 --aspect-ratio 64`" not in body

    stdout = capsys.readouterr().out
    assert "bundle_ok" in stdout
    assert "finalists=1" in stdout
    assert f"input_json={input_json}" in stdout


def test_main_rejects_when_no_qualified_finalists(tmp_path, monkeypatch):
    input_json = tmp_path / "ranked_runs.json"
    ranked_runs = {
        "ranked_runs": [
            {
                "config": "2x5",
                "depth": 2,
                "n_branches": 5,
                "aspect_ratio": 384,
                "selected_tok_per_sec": 525100.0,
                "min_val_bpb": 4.0388,
                "token_budget": 250000000,
                "qualified": False,
                "rank": None,
                "disqualify_reason": "slow>5.0%",
            }
        ]
    }
    input_json.write_text(json.dumps(ranked_runs), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ],
    )

    with pytest.raises(RuntimeError, match="expected at least 2 qualified finalists"):
        bundle.main()


def test_resolve_input_json_auto_skips_sample_and_selects_latest(tmp_path):
    artifacts_root = tmp_path / "pilot_artifacts"
    sample_dir = artifacts_root / "sample_bundle"
    real_old_dir = artifacts_root / "2026-02-01"
    real_new_dir = artifacts_root / "2026-02-02"
    for path in (sample_dir, real_old_dir, real_new_dir):
        path.mkdir(parents=True)

    sample_json = sample_dir / "pilot_ranked_runs.json"
    old_json = real_old_dir / "pilot_ranked_runs.json"
    new_json = real_new_dir / "pilot_ranked_runs.json"

    sample_json.write_text(json.dumps({"is_sample": True, "ranked_runs": [{}]}), encoding="utf-8")
    valid_payload = {
        "ranked_runs": [
            {
                "config": "12x1",
                "depth": 12,
                "n_branches": 1,
                "aspect_ratio": 64,
                "selected_tok_per_sec": 565800.0,
                "min_val_bpb": 4.0310,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 1,
                "disqualify_reason": None,
            }
        ]
    }
    old_json.write_text(json.dumps(valid_payload), encoding="utf-8")
    new_json.write_text(json.dumps(valid_payload), encoding="utf-8")

    os.utime(old_json, (1, 1))
    os.utime(new_json, None)

    resolved = bundle._resolve_input_json("auto", str(artifacts_root), "pilot_ranked_runs.json")

    assert resolved == new_json
    assert resolved != sample_json
    assert resolved.stat().st_mtime >= old_json.stat().st_mtime


def test_resolve_input_json_auto_errors_when_no_real_candidates(tmp_path):
    artifacts_root = tmp_path / "pilot_artifacts"
    (artifacts_root / "sample_bundle").mkdir(parents=True)
    (artifacts_root / "sample_bundle" / "pilot_ranked_runs.json").write_text(
        json.dumps({"is_sample": True, "ranked_runs": [{}]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="no real pilot ranking JSON found") as exc_info:
        bundle._resolve_input_json("auto", str(artifacts_root), "pilot_ranked_runs.json")

    assert "sample path segment" in str(exc_info.value)


def test_resolve_input_json_auto_rejects_payload_marked_sample(tmp_path):
    artifacts_root = tmp_path / "pilot_artifacts"
    disguised_sample_dir = artifacts_root / "2026-02-03"
    disguised_sample_dir.mkdir(parents=True)
    (disguised_sample_dir / "pilot_ranked_runs.json").write_text(
        json.dumps(
            {
                "is_sample": True,
                "ranked_runs": [
                    {
                        "config": "12x1",
                        "depth": 12,
                        "n_branches": 1,
                        "aspect_ratio": 64,
                        "selected_tok_per_sec": 565800.0,
                        "min_val_bpb": 4.0310,
                        "token_budget": 250000000,
                        "qualified": True,
                        "rank": 1,
                        "disqualify_reason": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="payload marked is_sample=true"):
        bundle._resolve_input_json("auto", str(artifacts_root), "pilot_ranked_runs.json")


def test_resolve_input_json_auto_reports_invalid_json_diagnostics(tmp_path):
    artifacts_root = tmp_path / "pilot_artifacts"
    malformed_dir = artifacts_root / "2026-02-03"
    malformed_dir.mkdir(parents=True)
    (malformed_dir / "pilot_ranked_runs.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="rejected 1 candidate file") as exc_info:
        bundle._resolve_input_json("auto", str(artifacts_root), "pilot_ranked_runs.json")

    assert "unreadable JSON" in str(exc_info.value)


def test_main_writes_blocked_markdown_on_failure(tmp_path, monkeypatch, capsys):
    artifacts_root = tmp_path / "pilot_artifacts"
    sample_dir = artifacts_root / "sample_bundle"
    sample_dir.mkdir(parents=True)
    (sample_dir / "pilot_ranked_runs.json").write_text(
        json.dumps({"is_sample": True, "ranked_runs": [{}]}),
        encoding="utf-8",
    )
    blocked_md = tmp_path / "receipts" / "stage2_promotion_blocked.md"
    discovery_json = tmp_path / "receipts" / "stage2_promotion_discovery.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            "auto",
            "--input-root",
            str(artifacts_root),
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--output-blocked-md",
            str(blocked_md),
            "--output-discovery-json",
            str(discovery_json),
        ],
    )

    with pytest.raises(RuntimeError, match="no real pilot ranking JSON found"):
        bundle.main()

    blocked = blocked_md.read_text(encoding="utf-8")
    assert "# Stage 2 Promotion Bundle Blocked" in blocked
    assert "- status: blocked" in blocked
    assert "- error_type: `RuntimeError`" in blocked
    assert "--input-json auto" in blocked
    assert f"- finalists_json: `{tmp_path / 'artifacts' / 'stage2_finalists.json'}`" in blocked
    assert f"- discovery_json: `{discovery_json}`" in blocked

    discovery = json.loads(discovery_json.read_text(encoding="utf-8"))
    assert discovery["status"] == "blocked"
    assert discovery["mode"] == "auto"
    assert discovery["selected_input_json"] is None
    assert discovery["rejected_candidates"][0]["reason"] == "sample path segment"

    stdout = capsys.readouterr().out
    assert "stage2_promotion_bundle_blocked" in stdout
    assert f"blocked_md={blocked_md}" in stdout


def test_main_writes_discovery_receipt_for_auto_resolution_success(tmp_path, monkeypatch, capsys):
    artifacts_root = tmp_path / "pilot_artifacts"
    sample_dir = artifacts_root / "sample_bundle"
    real_dir = artifacts_root / "2026-02-03"
    sample_dir.mkdir(parents=True)
    real_dir.mkdir(parents=True)

    (sample_dir / "pilot_ranked_runs.json").write_text(
        json.dumps({"is_sample": True, "ranked_runs": [{}]}),
        encoding="utf-8",
    )
    ranked_runs = {
        "ranked_runs": [
            {
                "config": "4x3",
                "depth": 4,
                "n_branches": 3,
                "aspect_ratio": 192,
                "selected_tok_per_sec": 572110.0,
                "min_val_bpb": 4.0123,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 1,
                "disqualify_reason": None,
            },
            {
                "config": "12x1",
                "depth": 12,
                "n_branches": 1,
                "aspect_ratio": 64,
                "selected_tok_per_sec": 565800.0,
                "min_val_bpb": 4.0310,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 2,
                "disqualify_reason": None,
            },
        ]
    }
    real_ranked = real_dir / "pilot_ranked_runs.json"
    real_ranked.write_text(json.dumps(ranked_runs), encoding="utf-8")

    output_dir = tmp_path / "artifacts"
    discovery_json = tmp_path / "receipts" / "stage2_promotion_discovery.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            "auto",
            "--input-root",
            str(artifacts_root),
            "--output-dir",
            str(output_dir),
            "--min-finalists",
            "1",
            "--max-finalists",
            "2",
            "--output-discovery-json",
            str(discovery_json),
        ],
    )

    bundle.main()

    discovery = json.loads(discovery_json.read_text(encoding="utf-8"))
    assert discovery["status"] == "ok"
    assert discovery["mode"] == "auto"
    assert discovery["selected_input_json"] == str(real_ranked)
    assert str(real_ranked) in discovery["discovered_candidates"]
    assert discovery["rejected_candidates"][0]["reason"] == "sample path segment"

    stdout = capsys.readouterr().out
    assert "bundle_ok" in stdout
    assert f"discovery_json={discovery_json}" in stdout


def test_main_rejects_invalid_finalist_bounds(tmp_path, monkeypatch):
    input_json = tmp_path / "ranked_runs.json"
    ranked_runs = {
        "ranked_runs": [
            {
                "config": "4x3",
                "depth": 4,
                "n_branches": 3,
                "aspect_ratio": 192,
                "selected_tok_per_sec": 572110.0,
                "min_val_bpb": 4.0123,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 1,
                "disqualify_reason": None,
            }
        ]
    }
    input_json.write_text(json.dumps(ranked_runs), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--min-finalists",
            "3",
            "--max-finalists",
            "2",
        ],
    )

    with pytest.raises(ValueError, match="--min-finalists must be <= --max-finalists"):
        bundle.main()


def test_main_require_real_input_rejects_sample_fixture(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    ranked_runs_json = repo_root / "artifacts" / "pilot" / "sample_ranked_runs.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(ranked_runs_json),
            "--output-dir",
            str(repo_root / "artifacts" / "pilot"),
            "--min-finalists",
            "1",
            "--require-real-input",
        ],
    )

    with pytest.raises(ValueError, match="--require-real-input rejects sample/fixture"):
        bundle.main()


def test_main_writes_runbook_when_requested(tmp_path, monkeypatch, capsys):
    input_json = tmp_path / "ranked_runs.json"
    output_dir = tmp_path / "artifacts"
    runbook_md = tmp_path / "docs" / "stage2_runbook.md"
    ranked_runs = {
        "ranked_runs": [
            {
                "config": "4x3",
                "depth": 4,
                "n_branches": 3,
                "aspect_ratio": 192,
                "selected_tok_per_sec": 572110.0,
                "min_val_bpb": 4.0123,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 1,
                "disqualify_reason": None,
            },
            {
                "config": "12x1",
                "depth": 12,
                "n_branches": 1,
                "aspect_ratio": 64,
                "selected_tok_per_sec": 565800.0,
                "min_val_bpb": 4.0310,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 2,
                "disqualify_reason": None,
            },
        ]
    }
    input_json.write_text(json.dumps(ranked_runs), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(output_dir),
            "--min-finalists",
            "1",
            "--max-finalists",
            "2",
            "--require-real-input",
            "--output-runbook-md",
            str(runbook_md),
        ],
    )

    bundle.main()

    runbook = runbook_md.read_text(encoding="utf-8")
    assert "# Stage 2 Promotion Bundle Runbook" in runbook
    assert "python -m scripts.run_stage2_promotion_bundle" in runbook
    assert f"--input-json {input_json}" in runbook
    assert f"--output-dir {output_dir}" in runbook
    assert "--require-real-input" in runbook
    assert "python -m scripts.run_pilot_check_in" in runbook
    assert f"--ranked-json {input_json}" in runbook
    assert f"--finalists-json {output_dir / 'stage2_finalists.json'}" in runbook
    assert f"--finalists-md {output_dir / 'stage2_finalists.md'}" in runbook
    assert f"--output-check-json {output_dir / 'pilot_bundle_check.json'}" in runbook

    stdout = capsys.readouterr().out
    assert f"runbook_md={runbook_md}" in stdout


def test_main_runs_strict_check_in_when_requested(tmp_path, monkeypatch, capsys):
    input_json = tmp_path / "ranked_runs.json"
    output_dir = tmp_path / "artifacts"
    output_check_json = tmp_path / "receipts" / "stage2_check.json"
    output_bundle_json = tmp_path / "receipts" / "stage2_bundle.json"
    output_evidence_md = tmp_path / "receipts" / "stage2_evidence.md"
    ranked_runs = {
        "ranked_runs": [
            {
                "config": "4x3",
                "depth": 4,
                "n_branches": 3,
                "aspect_ratio": 192,
                "selected_tok_per_sec": 572110.0,
                "min_val_bpb": 4.0123,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 1,
                "disqualify_reason": None,
            },
            {
                "config": "12x1",
                "depth": 12,
                "n_branches": 1,
                "aspect_ratio": 64,
                "selected_tok_per_sec": 565800.0,
                "min_val_bpb": 4.0310,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 2,
                "disqualify_reason": None,
            },
        ]
    }
    input_json.write_text(json.dumps(ranked_runs), encoding="utf-8")

    check_call: dict[str, object] = {}

    def _fake_run_pilot_bundle_check(**kwargs):
        check_call.update(kwargs)
        Path(kwargs["output_check_json"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_check_json"]).write_text('{"status":"ok"}\n', encoding="utf-8")
        return 2

    monkeypatch.setattr(bundle, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(output_dir),
            "--min-finalists",
            "1",
            "--max-finalists",
            "2",
            "--run-check-in",
            "--output-check-json",
            str(output_check_json),
            "--output-bundle-json",
            str(output_bundle_json),
            "--output-evidence-md",
            str(output_evidence_md),
        ],
    )

    bundle.main()

    assert check_call["ranked_json_path"] == Path(input_json)
    assert check_call["finalists_json_path"] == output_dir / "stage2_finalists.json"
    assert check_call["finalists_md_path"] == output_dir / "stage2_finalists.md"
    assert check_call["check_in"] is True
    assert check_call["output_check_json"] == str(output_check_json)

    bundle_payload = json.loads(output_bundle_json.read_text(encoding="utf-8"))
    assert bundle_payload["status"] == "ok"
    assert bundle_payload["input_json"] == str(input_json)
    assert bundle_payload["finalists_json"] == str(output_dir / "stage2_finalists.json")
    assert bundle_payload["finalists_md"] == str(output_dir / "stage2_finalists.md")
    assert bundle_payload["finalists_count"] == 2
    assert bundle_payload["run_check_in"] is True
    assert bundle_payload["check_json"] == str(output_check_json)
    assert bundle_payload["artifact_sha256"]["finalists_json"] == hashlib.sha256(
        (output_dir / "stage2_finalists.json").read_bytes()
    ).hexdigest()
    assert bundle_payload["artifact_sha256"]["finalists_md"] == hashlib.sha256(
        (output_dir / "stage2_finalists.md").read_bytes()
    ).hexdigest()
    assert bundle_payload["artifact_sha256"]["check_json"] == hashlib.sha256(
        output_check_json.read_bytes()
    ).hexdigest()

    evidence = output_evidence_md.read_text(encoding="utf-8")
    assert "# Stage 2 Promotion Evidence" in evidence
    assert f"- input_json: `{input_json}`" in evidence
    assert f"- finalists_json: `{output_dir / 'stage2_finalists.json'}`" in evidence
    assert f"- finalists_md: `{output_dir / 'stage2_finalists.md'}`" in evidence
    assert "- finalists_count: 2" in evidence
    assert "- run_check_in: true" in evidence
    assert f"- check_json: `{output_check_json}`" in evidence
    assert f"- bundle_json: `{output_bundle_json}`" in evidence

    stdout = capsys.readouterr().out
    assert f"check_json={output_check_json}" in stdout
    assert f"bundle_json={output_bundle_json}" in stdout
    assert f"evidence_md={output_evidence_md}" in stdout


def test_main_dry_run_preflights_paths_without_writing_or_checking(tmp_path, monkeypatch, capsys):
    input_json = tmp_path / "ranked_runs.json"
    output_dir = tmp_path / "artifacts"
    runbook_md = tmp_path / "docs" / "stage2_runbook.md"
    output_check_json = tmp_path / "receipts" / "stage2_check.json"
    output_bundle_json = tmp_path / "receipts" / "stage2_bundle.json"
    output_evidence_md = tmp_path / "receipts" / "stage2_evidence.md"
    input_json.write_text("{}", encoding="utf-8")

    check_called = False

    def _fake_run_pilot_bundle_check(**_kwargs):
        nonlocal check_called
        check_called = True
        return 2

    monkeypatch.setattr(bundle, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(output_dir),
            "--run-check-in",
            "--output-check-json",
            str(output_check_json),
            "--output-runbook-md",
            str(runbook_md),
            "--output-bundle-json",
            str(output_bundle_json),
            "--output-evidence-md",
            str(output_evidence_md),
            "--require-real-input",
            "--dry-run",
        ],
    )

    bundle.main()

    assert not check_called
    assert not (output_dir / "stage2_finalists.json").exists()
    assert not (output_dir / "stage2_finalists.md").exists()
    assert not runbook_md.exists()
    assert not output_bundle_json.exists()
    assert not output_evidence_md.exists()

    stdout = capsys.readouterr().out
    assert "stage2_promotion_bundle_dry_run_ok" in stdout
    assert f"input_json={input_json}" in stdout
    assert f"json={output_dir / 'stage2_finalists.json'}" in stdout
    assert f"md={output_dir / 'stage2_finalists.md'}" in stdout
    assert "run_check_in=True" in stdout
    assert "require_real_input=True" in stdout
    assert f"check_json={output_check_json}" in stdout
    assert f"runbook_md={runbook_md}" in stdout
    assert f"bundle_json={output_bundle_json}" in stdout
    assert f"evidence_md={output_evidence_md}" in stdout


def test_main_dry_run_can_write_runbook_when_enabled(tmp_path, monkeypatch, capsys):
    input_json = tmp_path / "ranked_runs.json"
    output_dir = tmp_path / "artifacts"
    runbook_md = tmp_path / "docs" / "stage2_runbook.md"
    input_json.write_text(
        json.dumps(
            {
                "ranked_runs": [
                    {
                        "config": "4x3",
                        "depth": 4,
                        "n_branches": 3,
                        "aspect_ratio": 192,
                        "selected_tok_per_sec": 572110.0,
                        "min_val_bpb": 4.0123,
                        "token_budget": 250000000,
                        "qualified": True,
                        "rank": 1,
                        "disqualify_reason": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(output_dir),
            "--output-runbook-md",
            str(runbook_md),
            "--dry-run",
            "--dry-run-write-runbook",
        ],
    )

    bundle.main()

    assert runbook_md.exists()
    runbook = runbook_md.read_text(encoding="utf-8")
    assert "python -m scripts.run_stage2_promotion_bundle" in runbook
    assert f"--input-json {input_json}" in runbook
    assert f"--output-dir {output_dir}" in runbook
    assert not (output_dir / "stage2_finalists.json").exists()
    assert not (output_dir / "stage2_finalists.md").exists()

    stdout = capsys.readouterr().out
    assert "stage2_promotion_bundle_dry_run_ok" in stdout
    assert f"runbook_written={runbook_md}" in stdout


def test_main_preflight_validates_without_writing_finalists(tmp_path, monkeypatch, capsys):
    input_json = tmp_path / "ranked_runs.json"
    output_dir = tmp_path / "artifacts"
    output_check_json = tmp_path / "receipts" / "stage2_check.json"
    output_bundle_json = tmp_path / "receipts" / "stage2_bundle.json"
    output_evidence_md = tmp_path / "receipts" / "stage2_evidence.md"
    output_preflight_json = tmp_path / "receipts" / "stage2_preflight.json"
    output_discovery_json = tmp_path / "receipts" / "stage2_discovery.json"

    ranked_runs = {
        "ranked_runs": [
            {
                "config": "4x3",
                "depth": 4,
                "n_branches": 3,
                "aspect_ratio": 192,
                "selected_tok_per_sec": 572110.0,
                "min_val_bpb": 4.0123,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 1,
                "disqualify_reason": None,
            },
            {
                "config": "12x1",
                "depth": 12,
                "n_branches": 1,
                "aspect_ratio": 64,
                "selected_tok_per_sec": 565800.0,
                "min_val_bpb": 4.0310,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 2,
                "disqualify_reason": None,
            },
        ]
    }
    input_json.write_text(json.dumps(ranked_runs), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(output_dir),
            "--min-finalists",
            "1",
            "--max-finalists",
            "2",
            "--run-check-in",
            "--output-check-json",
            str(output_check_json),
            "--output-bundle-json",
            str(output_bundle_json),
            "--output-evidence-md",
            str(output_evidence_md),
            "--preflight",
            "--output-preflight-json",
            str(output_preflight_json),
            "--output-discovery-json",
            str(output_discovery_json),
        ],
    )

    bundle.main()

    assert not (output_dir / "stage2_finalists.json").exists()
    assert not (output_dir / "stage2_finalists.md").exists()
    assert not output_check_json.exists()
    assert not output_bundle_json.exists()
    assert not output_evidence_md.exists()

    preflight_payload = json.loads(output_preflight_json.read_text(encoding="utf-8"))
    assert preflight_payload["status"] == "ok"
    assert preflight_payload["input_json"] == str(input_json)
    assert preflight_payload["finalists_json"] == str(output_dir / "stage2_finalists.json")
    assert preflight_payload["finalists_md"] == str(output_dir / "stage2_finalists.md")
    assert preflight_payload["finalists_count"] == 2
    assert preflight_payload["run_check_in"] is True
    assert preflight_payload["check_json"] == str(output_check_json)
    assert preflight_payload["bundle_json"] == str(output_bundle_json)
    assert preflight_payload["evidence_md"] == str(output_evidence_md)
    assert preflight_payload["discovery_json"] == str(output_discovery_json)

    stdout = capsys.readouterr().out
    assert "stage2_promotion_bundle_preflight_ok" in stdout
    assert f"input_json={input_json}" in stdout
    assert f"json={output_dir / 'stage2_finalists.json'}" in stdout
    assert f"md={output_dir / 'stage2_finalists.md'}" in stdout
    assert f"check_json={output_check_json}" in stdout
    assert f"bundle_json={output_bundle_json}" in stdout
    assert f"evidence_md={output_evidence_md}" in stdout
    assert f"discovery_json={output_discovery_json}" in stdout
    assert f"preflight_json={output_preflight_json}" in stdout


def test_main_preflight_rejects_mutually_exclusive_dry_run(tmp_path, monkeypatch):
    input_json = tmp_path / "ranked_runs.json"
    input_json.write_text(json.dumps({"ranked_runs": []}), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--dry-run",
            "--preflight",
        ],
    )

    with pytest.raises(ValueError, match="--dry-run and --preflight are mutually exclusive"):
        bundle.main()


def test_runbook_includes_check_in_flags_when_enabled(tmp_path):
    runbook_md = tmp_path / "stage2_runbook.md"
    check_json = tmp_path / "checks" / "bundle.json"
    input_json = Path("artifacts/pilot/pilot_ranked_runs.json")
    finalists_json = Path("artifacts/pilot/stage2_finalists.json")
    finalists_md = Path("artifacts/pilot/stage2_finalists.md")

    bundle._write_runbook_md(
        path=runbook_md,
        input_json=str(input_json),
        output_dir="artifacts/pilot",
        finalists_json=finalists_json,
        finalists_md=finalists_md,
        min_finalists=2,
        max_finalists=3,
        require_real_input=True,
        run_check_in=True,
        output_check_json=str(check_json),
        output_bundle_json="",
        output_blocked_md="",
    )

    runbook = runbook_md.read_text(encoding="utf-8")
    assert "--require-real-input" in runbook
    assert "--run-check-in" in runbook
    assert f"--output-check-json {check_json}" in runbook
    assert f"--ranked-json {input_json.resolve()}" in runbook
    assert f"--finalists-json {finalists_json.resolve()}" in runbook
    assert f"--finalists-md {finalists_md.resolve()}" in runbook


def test_runbook_shell_quotes_spaced_paths(tmp_path):
    runbook_md = tmp_path / "stage2 runbook.md"
    output_dir = tmp_path / "pilot artifacts"
    input_json = output_dir / "pilot ranked runs.json"
    finalists_json = output_dir / "stage2 finalists.json"
    finalists_md = output_dir / "stage2 finalists.md"
    check_json = output_dir / "pilot bundle check.json"

    bundle._write_runbook_md(
        path=runbook_md,
        input_json=str(input_json),
        output_dir=str(output_dir),
        finalists_json=finalists_json,
        finalists_md=finalists_md,
        min_finalists=2,
        max_finalists=3,
        require_real_input=False,
        run_check_in=True,
        output_check_json=str(check_json),
        output_bundle_json="",
        output_blocked_md="",
    )

    runbook = runbook_md.read_text(encoding="utf-8")
    quoted_input_json = shlex.quote(str(input_json))
    quoted_output_dir = shlex.quote(str(output_dir))
    quoted_finalists_json = shlex.quote(str(finalists_json))
    quoted_finalists_md = shlex.quote(str(finalists_md))
    quoted_check_json = shlex.quote(str(check_json))

    assert f"--input-json {quoted_input_json}" in runbook
    assert f"--output-dir {quoted_output_dir}" in runbook
    assert f"--artifacts-dir {quoted_output_dir}" in runbook
    assert f"--ranked-json {quoted_input_json}" in runbook
    assert f"--finalists-json {quoted_finalists_json}" in runbook
    assert f"--finalists-md {quoted_finalists_md}" in runbook
    assert f"--output-check-json {quoted_check_json}" in runbook


def test_runbook_includes_bundle_receipt_when_requested(tmp_path):
    runbook_md = tmp_path / "stage2_runbook.md"
    output_dir = tmp_path / "pilot_artifacts"
    input_json = output_dir / "pilot_ranked_runs.json"
    finalists_json = output_dir / "stage2_finalists.json"
    finalists_md = output_dir / "stage2_finalists.md"
    check_json = output_dir / "pilot_bundle_check.json"
    bundle_json = output_dir / "stage2_promotion_bundle.json"

    bundle._write_runbook_md(
        path=runbook_md,
        input_json=str(input_json),
        output_dir=str(output_dir),
        finalists_json=finalists_json,
        finalists_md=finalists_md,
        min_finalists=2,
        max_finalists=3,
        require_real_input=True,
        run_check_in=True,
        output_check_json=str(check_json),
        output_bundle_json=str(bundle_json),
        output_blocked_md="",
    )

    runbook = runbook_md.read_text(encoding="utf-8")
    assert f"--output-bundle-json {bundle_json}" in runbook
    assert f"--bundle-json {bundle_json}" in runbook


def test_main_runbook_check_in_command_uses_absolute_ranked_path(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)

    output_dir = tmp_path / "artifacts"
    runbook_md = tmp_path / "stage2_runbook.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            "artifacts/pilot/sample_ranked_runs.json",
            "--output-dir",
            str(output_dir),
            "--min-finalists",
            "1",
            "--max-finalists",
            "2",
            "--output-runbook-md",
            str(runbook_md),
        ],
    )

    bundle.main()

    runbook = runbook_md.read_text(encoding="utf-8")
    assert "--artifacts-dir" in runbook
    assert f"--ranked-json {Path('artifacts/pilot/sample_ranked_runs.json').resolve()}" in runbook
    assert f"--finalists-json {(output_dir / 'stage2_finalists.json').resolve()}" in runbook
    assert f"--finalists-md {(output_dir / 'stage2_finalists.md').resolve()}" in runbook


def test_main_writes_resolved_bundle_command_artifact(tmp_path, monkeypatch, capsys):
    input_json = tmp_path / "ranked_runs.json"
    output_dir = tmp_path / "promotion artifacts"
    output_check_json = tmp_path / "receipts" / "stage2 check.json"
    output_bundle_command = tmp_path / "receipts" / "stage2_bundle_command.sh"
    ranked_runs = {
        "ranked_runs": [
            {
                "config": "4x3",
                "depth": 4,
                "n_branches": 3,
                "aspect_ratio": 192,
                "selected_tok_per_sec": 572110.0,
                "min_val_bpb": 4.0123,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 1,
                "disqualify_reason": None,
            },
            {
                "config": "12x1",
                "depth": 12,
                "n_branches": 1,
                "aspect_ratio": 64,
                "selected_tok_per_sec": 565800.0,
                "min_val_bpb": 4.0310,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 2,
                "disqualify_reason": None,
            },
        ]
    }
    input_json.write_text(json.dumps(ranked_runs), encoding="utf-8")

    def _fake_run_pilot_bundle_check(**kwargs):
        Path(kwargs["output_check_json"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_check_json"]).write_text('{"status":"ok"}\n', encoding="utf-8")
        return 2

    monkeypatch.setattr(bundle, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(output_dir),
            "--min-finalists",
            "1",
            "--max-finalists",
            "2",
            "--run-check-in",
            "--output-check-json",
            str(output_check_json),
            "--output-bundle-command-sh",
            str(output_bundle_command),
        ],
    )

    bundle.main()

    command = output_bundle_command.read_text(encoding="utf-8").strip()
    assert command.startswith("python -m scripts.run_stage2_promotion_bundle")
    assert f"--input-json {shlex.quote(str(input_json))}" in command
    assert f"--output-dir {shlex.quote(str(output_dir))}" in command
    assert f"--output-json {shlex.quote(str(output_dir / 'stage2_finalists.json'))}" in command
    assert f"--output-md {shlex.quote(str(output_dir / 'stage2_finalists.md'))}" in command
    assert "--run-check-in" in command
    assert f"--output-check-json {shlex.quote(str(output_check_json))}" in command
    assert f"--output-bundle-command-sh {shlex.quote(str(output_bundle_command))}" in command

    stdout = capsys.readouterr().out
    assert f"bundle_command_sh={output_bundle_command}" in stdout


def test_main_preflight_writes_bundle_command_path_in_receipt(tmp_path, monkeypatch):
    input_json = tmp_path / "ranked_runs.json"
    output_dir = tmp_path / "artifacts"
    output_preflight_json = tmp_path / "receipts" / "stage2_preflight.json"
    output_bundle_command = tmp_path / "receipts" / "stage2_bundle_command.sh"
    ranked_runs = {
        "ranked_runs": [
            {
                "config": "4x3",
                "depth": 4,
                "n_branches": 3,
                "aspect_ratio": 192,
                "selected_tok_per_sec": 572110.0,
                "min_val_bpb": 4.0123,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 1,
                "disqualify_reason": None,
            },
            {
                "config": "12x1",
                "depth": 12,
                "n_branches": 1,
                "aspect_ratio": 64,
                "selected_tok_per_sec": 565800.0,
                "min_val_bpb": 4.0310,
                "token_budget": 250000000,
                "qualified": True,
                "rank": 2,
                "disqualify_reason": None,
            },
        ]
    }
    input_json.write_text(json.dumps(ranked_runs), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_promotion_bundle.py",
            "--input-json",
            str(input_json),
            "--output-dir",
            str(output_dir),
            "--preflight",
            "--output-preflight-json",
            str(output_preflight_json),
            "--output-bundle-command-sh",
            str(output_bundle_command),
            "--min-finalists",
            "1",
            "--max-finalists",
            "2",
        ],
    )

    bundle.main()

    payload = json.loads(output_preflight_json.read_text(encoding="utf-8"))
    assert payload["bundle_command_sh"] == str(output_bundle_command)
    assert output_bundle_command.is_file()
