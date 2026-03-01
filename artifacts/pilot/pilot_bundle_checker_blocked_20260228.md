# Pilot Bundle Checker Blocked

## Context
- command: `["/home/aj/workspace/github.com/widechat/scripts/check_pilot_sweep_artifacts.py", "--artifacts-dir", "auto", "--artifacts-root", "artifacts/pilot", "--check-in", "--dry-run", "--output-blocked-md", "artifacts/pilot/pilot_bundle_checker_blocked_20260228.md"]`
- artifacts_dir_arg: `auto`
- artifacts_root_arg: `artifacts/pilot`
- ranked_json_arg: ``
- finalists_json_arg: ``
- finalists_md_arg: ``
- bundle_json_arg: ``
- check_in_mode: `true`
- dry_run_mode: `true`
- output_check_json: ``
- resolved_ranked_json: ``
- resolved_finalists_json: ``
- resolved_finalists_md: ``
- resolved_bundle_json: ``

## Blocker
```text
no real pilot artifact bundle found under artifacts/pilot; run scripts.pilot_sweep on target GPU(s) first
discovery searched for 'pilot_ranked_runs.json' files and rejected 1 candidate bundle(s)
- artifacts/pilot: missing files: pilot_ranked_runs.json, stage2_finalists.json, stage2_finalists.md
```
