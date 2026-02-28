# Branch Throughput Report

This report covers the roadmap benchmark comparison for the baseline and key breadth-heavy configs:

- Baseline: `12x1`
- Breadth-heavy: `2x5`, `1x10`

## Benchmark command

```bash
python -m scripts.throughput_benchmark \
  --device-type cpu \
  --max-seq-len 128 \
  --total-batch-size 4096 \
  --device-batch-size 2 \
  --num-iterations 5
```

## Results

| Config | tok/sec | vs 12x1 | MFU | Peak mem (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 12x1 | 642 | +0.0% | 0.00 | 0.00 |
| 2x5 | 839 | +30.7% | 0.00 | 0.00 |
| 1x10 | 804 | +25.2% | 0.00 | 0.00 |

## Notes

- This run was executed on `cpu` to provide a reproducible, local smoke benchmark.
- The breadth-heavy configs were faster than `12x1` under this fixed-shape setup.
- For architecture decisions, repeat on the target GPU setup (same command with `--device-type cuda` and production-scale sequence/batch settings).
