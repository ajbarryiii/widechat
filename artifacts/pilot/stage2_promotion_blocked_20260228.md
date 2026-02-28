# Stage 2 Promotion Bundle Blocked

- status: blocked
- error_type: `RuntimeError`
- error: `no real pilot ranking JSON found under artifacts/pilot; run scripts.pilot_sweep on target GPU(s) first
discovery searched for 'pilot_ranked_runs.json' files and rejected 0 candidate file(s)`
- input_json: `auto`
- finalists_json: `artifacts/pilot/stage2_finalists.json`
- finalists_md: `artifacts/pilot/stage2_finalists.md`
- run_check_in: false
- check_json: (not configured)
- runbook_md: (not configured)
- bundle_json: `artifacts/pilot/stage2_promotion_bundle_20260228.json`
- evidence_md: `artifacts/pilot/stage2_promotion_evidence_20260228.md`
- preflight_json: (not configured)
- bundle_command_sh: (not configured)
- discovery_json: `artifacts/pilot/stage2_promotion_discovery_20260228.json`

## Command
```bash
/home/aj/workspace/github.com/widechat/scripts/run_stage2_promotion_bundle.py --input-json auto --input-root artifacts/pilot --output-dir artifacts/pilot --require-real-input --output-bundle-json artifacts/pilot/stage2_promotion_bundle_20260228.json --output-evidence-md artifacts/pilot/stage2_promotion_evidence_20260228.md --output-blocked-md artifacts/pilot/stage2_promotion_blocked_20260228.md --output-discovery-json artifacts/pilot/stage2_promotion_discovery_20260228.json
```
