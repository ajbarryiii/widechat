## Pilot Sweep Runbook

### Artifact paths

- ranked runs JSON: `artifacts/pilot/target_sweep_20260228/pilot_ranked_runs.json`
- ranking markdown: `artifacts/pilot/target_sweep_20260228/pilot_ranking.md`
- Stage 2 finalists JSON: `artifacts/pilot/target_sweep_20260228/stage2_finalists.json`
- Stage 2 finalists markdown: `artifacts/pilot/target_sweep_20260228/stage2_finalists.md`
- strict check receipt JSON: `artifacts/pilot/target_sweep_20260228/pilot_bundle_check.json`

### Commands

1. Initial sweep run:

```bash
/home/aj/workspace/github.com/widechat/.venv/bin/python -m scripts.pilot_sweep --total-batch-size 524288 --device-batch-size 16 --pilot-tokens 250000000 --eval-every 75 --eval-tokens 1048576 --max-seq-len 2048 --slowdown-threshold-pct 5.0 --clear-bpb-gain 0.02 --max-finalists 3 --artifacts-dir artifacts/pilot/target_sweep_20260228 --output-json artifacts/pilot/target_sweep_20260228/pilot_ranked_runs.json --output-md artifacts/pilot/target_sweep_20260228/pilot_ranking.md --output-finalists-json artifacts/pilot/target_sweep_20260228/stage2_finalists.json --output-finalists-md artifacts/pilot/target_sweep_20260228/stage2_finalists.md --device-type cuda
```

2. Resume interrupted run from existing artifacts:

```bash
/home/aj/workspace/github.com/widechat/.venv/bin/python -m scripts.pilot_sweep --total-batch-size 524288 --device-batch-size 16 --pilot-tokens 250000000 --eval-every 75 --eval-tokens 1048576 --max-seq-len 2048 --slowdown-threshold-pct 5.0 --clear-bpb-gain 0.02 --max-finalists 3 --artifacts-dir artifacts/pilot/target_sweep_20260228 --output-json artifacts/pilot/target_sweep_20260228/pilot_ranked_runs.json --output-md artifacts/pilot/target_sweep_20260228/pilot_ranking.md --output-finalists-json artifacts/pilot/target_sweep_20260228/stage2_finalists.json --output-finalists-md artifacts/pilot/target_sweep_20260228/stage2_finalists.md --device-type cuda --resume-from-artifacts
```

3. Strict check-in validation on emitted artifacts:

```bash
/home/aj/workspace/github.com/widechat/.venv/bin/python -m scripts.run_pilot_check_in --artifacts-dir artifacts/pilot/target_sweep_20260228 --ranked-json pilot_ranked_runs.json --finalists-json stage2_finalists.json --finalists-md stage2_finalists.md --output-check-json artifacts/pilot/target_sweep_20260228/pilot_bundle_check.json
```
