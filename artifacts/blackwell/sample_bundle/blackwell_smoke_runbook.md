# Blackwell Smoke Bundle Runbook

## Command

```bash
python -m scripts.run_blackwell_smoke_bundle \
  --output-dir artifacts/blackwell/sample_bundle \
  --expect-backend fa4 \
  --require-device-substring 'RTX 5090'
```

## Expected outputs

- `artifacts/blackwell/sample_bundle/flash_backend_smoke.json`
- `artifacts/blackwell/sample_bundle/flash_backend_status.log`
- `artifacts/blackwell/sample_bundle/blackwell_smoke_evidence.md`

## Check-in checklist

- Ensure command prints `bundle_ok selected=fa4`.
- Run `python -m scripts.run_blackwell_check_in --bundle-dir artifacts/blackwell/sample_bundle --expect-backend fa4 --output-check-json artifacts/blackwell/sample_bundle/blackwell_bundle_check.json --require-device-substring 'RTX 5090'`.
