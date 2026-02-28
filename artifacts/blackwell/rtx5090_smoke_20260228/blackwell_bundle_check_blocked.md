# Blackwell Bundle Checker Blocked

## Context
- command: `["/home/aj/workspace/github.com/widechat/scripts/check_blackwell_evidence_bundle.py", "--bundle-dir", "artifacts/blackwell/rtx5090_smoke_20260228", "--expect-backend", "fa4", "--check-in", "--output-check-json", "artifacts/blackwell/rtx5090_smoke_20260228/blackwell_bundle_check.json", "--output-blocked-md", "artifacts/blackwell/rtx5090_smoke_20260228/blackwell_bundle_check_blocked.md"]`
- bundle_dir_arg: `artifacts/blackwell/rtx5090_smoke_20260228`
- bundle_root_arg: `artifacts/blackwell`
- expect_backend: `fa4`
- check_in: `true`
- require_blackwell: `false`
- require_git_tracked: `false`
- require_real_bundle: `false`
- require_device_substring: ``
- preflight: `false`
- dry_run: `false`
- output_check_json: `artifacts/blackwell/rtx5090_smoke_20260228/blackwell_bundle_check.json`
- output_preflight_json: ``
- output_discovery_json: ``
- resolved_bundle_dir: `artifacts/blackwell/rtx5090_smoke_20260228`

## Blocker
```text
missing artifact_json file: artifacts/blackwell/rtx5090_smoke_20260228/flash_backend_smoke.json
```
