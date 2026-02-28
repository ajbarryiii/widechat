from scripts import run_pilot_check_in as runner


def test_main_runs_strict_check_in_with_default_receipt(tmp_path, monkeypatch, capsys):
    artifacts_dir = tmp_path / "pilot"
    calls = {}

    def _fake_run_pilot_bundle_check(**kwargs):
        calls.update(kwargs)
        return 2

    monkeypatch.setattr(runner, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            str(artifacts_dir),
        ],
    )

    runner.main()

    assert calls["ranked_json_path"] == artifacts_dir / "pilot_ranked_runs.json"
    assert calls["finalists_json_path"] == artifacts_dir / "stage2_finalists.json"
    assert calls["finalists_md_path"] == artifacts_dir / "stage2_finalists.md"
    assert calls["require_real_input"] is True
    assert calls["require_git_tracked"] is False
    assert calls["check_in"] is True
    assert calls["allow_sample_input_in_check_in"] is False
    assert calls["output_check_json"] == str(artifacts_dir / "pilot_bundle_check.json")

    stdout = capsys.readouterr().out
    assert "pilot_check_in_ok finalists=2" in stdout


def test_main_honors_custom_paths(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "pilot"
    receipt_path = tmp_path / "receipts" / "check.json"
    calls = {}

    def _fake_run_pilot_bundle_check(**kwargs):
        calls.update(kwargs)
        return 3

    monkeypatch.setattr(runner, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            str(artifacts_dir),
            "--ranked-json",
            "real_ranked.json",
            "--finalists-json",
            "real_finalists.json",
            "--finalists-md",
            "real_finalists.md",
            "--output-check-json",
            str(receipt_path),
        ],
    )

    runner.main()

    assert calls["ranked_json_path"] == artifacts_dir / "real_ranked.json"
    assert calls["finalists_json_path"] == artifacts_dir / "real_finalists.json"
    assert calls["finalists_md_path"] == artifacts_dir / "real_finalists.md"
    assert calls["require_real_input"] is True
    assert calls["allow_sample_input_in_check_in"] is False
    assert calls["output_check_json"] == str(receipt_path)


def test_main_allow_sample_input_disables_real_input_guard(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "pilot"
    calls = {}

    def _fake_run_pilot_bundle_check(**kwargs):
        calls.update(kwargs)
        return 1

    monkeypatch.setattr(runner, "run_pilot_bundle_check", _fake_run_pilot_bundle_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pilot_check_in.py",
            "--artifacts-dir",
            str(artifacts_dir),
            "--allow-sample-input",
        ],
    )

    runner.main()

    assert calls["require_real_input"] is False
    assert calls["allow_sample_input_in_check_in"] is True
