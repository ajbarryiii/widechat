# Blackwell Smoke Bundle Runbook

## Command
```bash
python -m scripts.run_blackwell_smoke_bundle \
  --output-dir artifacts/blackwell/rtx5090_smoke_20260228 \
  --expect-backend fa4 \
  --require-device-substring 'RTX 5090'
```

## Expected outputs
- `artifacts/blackwell/rtx5090_smoke_20260228/flash_backend_smoke.json`
- `artifacts/blackwell/rtx5090_smoke_20260228/flash_backend_status.log`
- `artifacts/blackwell/rtx5090_smoke_20260228/blackwell_smoke_evidence.md`

## Check-in checklist
- Ensure command prints `bundle_ok selected=fa4`.
- Run `python -m scripts.run_blackwell_check_in --bundle-dir artifacts/blackwell/rtx5090_smoke_20260228 --expect-backend fa4 --output-check-json artifacts/blackwell/rtx5090_smoke_20260228/blackwell_bundle_check.json --require-device-substring 'RTX 5090'`.
- Equivalent strict checker command: `python -m scripts.check_blackwell_evidence_bundle --bundle-dir artifacts/blackwell/rtx5090_smoke_20260228 --expect-backend fa4 --check-in --output-check-json artifacts/blackwell/rtx5090_smoke_20260228/blackwell_bundle_check.json --require-device-substring 'RTX 5090'`.
- Verify evidence markdown includes device metadata and `status_line_ok: true`.
- Confirm `artifacts/blackwell/rtx5090_smoke_20260228/blackwell_bundle_check.json` records `selected_backend: fa4`.
- Commit the emitted evidence artifacts from this run.
