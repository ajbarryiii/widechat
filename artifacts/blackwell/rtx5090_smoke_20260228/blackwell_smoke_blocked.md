# Blackwell Smoke Runtime Blocker

Blackwell FA4 smoke execution failed before a complete evidence bundle was emitted.

## Receipt
- mode: `smoke`
- generated_at_utc: `2026-02-28T16:15:20Z`
- ready: `false`
- error: `CUDA is required but not available. nvidia-smi reports GPU(s): NVIDIA GeForce RTX 5090, 580.119.02. This usually means the active PyTorch build lacks CUDA support or is mismatched with the system CUDA driver/runtime.`
- expect_backend: `fa4`
- require_device_substring: `RTX 5090`
- cuda_available: `false`
- device_name: `None`
- cuda_capability: `None`
- nvidia_smi_ok: `true`
- nvidia_smi_error: `None`

## Next command on RTX 5090
```bash
python -m scripts.run_blackwell_smoke_bundle \
  --output-dir artifacts/blackwell/rtx5090_smoke_20260228 \
  --expect-backend fa4 \
  --require-device-substring 'RTX 5090'
```
