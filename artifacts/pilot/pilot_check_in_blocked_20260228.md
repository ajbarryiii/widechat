# Pilot Artifact Strict Check-In Blocked

## Context
- command: `["/home/aj/workspace/github.com/widechat/scripts/run_pilot_check_in.py", "--artifacts-dir", "auto", "--artifacts-root", "artifacts/pilot", "--preflight", "--output-discovery-json", "artifacts/pilot/pilot_check_in_discovery_20260228.json", "--output-preflight-json", "artifacts/pilot/pilot_check_in_preflight_20260228.json", "--output-blocked-md", "artifacts/pilot/pilot_check_in_blocked_20260228.md"]`
- artifacts_dir_arg: `auto`
- artifacts_root_arg: `artifacts/pilot`
- preflight_mode: `true`
- dry_run_mode: `false`

## Blocker
```text
no real pilot artifact bundle found under artifacts/pilot; run scripts.pilot_sweep on target GPU(s) first
discovery searched for 'pilot_ranked_runs.json' files and rejected 0 candidate bundle(s)
```
