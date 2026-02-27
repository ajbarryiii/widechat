# Architectural Roadmap: Sequential Depth vs Parallel Breadth (Revised)

## Objective
Investigate the tradeoff between sequential depth and parallel breadth while keeping model width fixed and scaling parameters within +/-10% of baseline.

This experiment targets one architectural question: can we trade some sequential layers for parallel branches and improve hardware utilization without hurting early convergence?

## Success Criteria
1. Correctness parity: `n_branches=1` matches current behavior (loss/gradients within numerical tolerance).
2. Performance win: at least one `n_branches>1` config is >= baseline tokens/sec at equal global batch and sequence length.
3. Stability: no NaN/Inf, no optimizer instability, no checkpoint/load regressions.
4. Training signal: short pilots identify at least 2 candidate configs worth longer runs.

## Scope and Non-Goals
- Scope: base model training path first (`scripts/base_train.py`, `nanochat/gpt.py`, optimizer plumbing, tests).
- Scope: preserve existing features unless explicitly disabled for clean ablation (window pattern, GQA, value embedding gates, residual scalars).
- Non-goal (phase 1): full inference engine parity for branched models. Keep inference work as phase 2 after training path is stable.
- Non-goal: remove all Python loops. The depth loop over `D` remains; the goal is to remove per-branch Python loops.

## Notation (Unambiguous)
- `N`: batch size
- `T`: sequence length
- `C`: embedding width (`n_embd`)
- `Hq`: query head count (`n_head`)
- `Hkv`: KV head count (`n_kv_head`)
- `Dh`: head dim (`C / Hq`)
- `D`: sequential depth (`n_layer`)
- `R`: number of parallel branches (`n_branches`)

## Architecture Specification

### 1) Input and branch expansion
- `x0 = norm(wte(tokens))` -> `(N, T, C)`
- `x = linear_in(x0)` -> `(N, T, R*C)`
- reshape `x` -> `(N, T, R, C)`

### 2) Parallel branch trunk over depth
- For `i in [0, D)`:
  - `x = BatchedParallelBlock_i(x)`
- No loop over branches in Python.
- Branch operations are batched with tensor ops.

### 3) Attention kernel layout
- `q`: `(N, T, R, Hq, Dh)`
- `k,v`: `(N, T, R, Hkv, Dh)`
- reshape for attention call:
  - `q -> (N*R, T, Hq, Dh)`
  - `k,v -> (N*R, T, Hkv, Dh)`
- call `flash_attn.flash_attn_func(...)` or fallback via existing wrapper.
- reshape output back to `(N, T, R, C)`.

### 4) Branch collection
- reshape trunk output to `(N, T, R*C)`
- `x = linear_out(...)` -> `(N, T, C)`
- keep this phase minimal: no extra macro residual on top of existing residual logic.

### 5) Output head
- unchanged: `logits = lm_head(norm(x))`

## Parameter Accounting (for config gating)

For fixed `C=768` and per-branch block size `P_block = 12*C^2`:
- parallel trunk params: `D * R * P_block`
- branch split/collect params: `2 * R * C^2`
- transformer matrix params (experiment focus):
  - `P_transformer = D*R*P_block + 2*R*C^2`

Baseline reference (old architecture, no split/collect):
- `P_base = 12 * P_block = 84,934,656`

Candidate grid (all within +/-10% of `P_base`):

| Label | D x R | `P_transformer` | Delta vs baseline |
|---|---:|---:|---:|
| Baseline-like | 12 x 1 | 86,114,304 | +1.4% |
| A | 6 x 2 | 87,293,952 | +2.8% |
| B | 4 x 3 | 88,473,600 | +4.2% |
| C | 3 x 4 | 89,653,248 | +5.6% |
| D | 2 x 5 | 76,677,120 | -9.7% |
| E | 2 x 6 | 92,012,544 | +8.3% |
| F | 1 x 10 | 82,575,360 | -2.8% |

Note: in this repo, training horizon currently keys off `transformer_matrices + lm_head` via `num_scaling_params()`. The new layers must be included there.

## Config Mapping in This Repo

To isolate depth vs breadth with fixed width (`C=768`) in current `scripts/base_train.py`:
- keep `head_dim=128` so `n_head=6` stays constant.
- set `--aspect-ratio = 768 / depth` per config.
- add `--n-branches` as a new CLI/model config knob.

Examples:
- `D=12` -> `--aspect-ratio 64`
- `D=6` -> `--aspect-ratio 128`
- `D=4` -> `--aspect-ratio 192`
- `D=3` -> `--aspect-ratio 256`
- `D=2` -> `--aspect-ratio 384`
- `D=1` -> `--aspect-ratio 768`

## Implementation Plan by File

### A) `nanochat/gpt.py`
- [x] Phase 0: add `n_branches` to `GPTConfig` with default `1` and explicitly guard `n_branches>1` as not-yet-implemented.
- [x] Add `BatchedLinear`:
  - [x] weight shape `(R, O, I)`
  - [x] forward contract: `x (N,T,R,I)` -> `y (N,T,R,O)`
  - [x] einsum form: `torch.einsum('ntri,roi->ntro', x, w)`
- [ ] Add branch-aware block modules (attention + MLP) for shape `(N,T,R,C)`.
- [ ] Keep baseline path for `n_branches=1` as close as possible to current behavior.
- [ ] Update `forward()` with:
  - [ ] `linear_in -> branch blocks -> linear_out`
- [ ] Ensure `num_scaling_params()` and `estimate_flops()` account for new matrices.

### B) `scripts/base_train.py`
- [x] Add `--n-branches` CLI arg.
- [x] Pass `n_branches` into `GPTConfig`.
- [x] Update default `model_tag` format to avoid collisions (e.g. `d{depth}b{branches}`).

### C) `nanochat/checkpoint_manager.py`
- [x] Backward compatibility patch for missing config key:
  - [x] if `n_branches` absent in old checkpoints, default to `1`.

### D) `nanochat/optim.py` and `nanochat/gpt.py`
- Phase 1 (simplest): put new batched branch projection weights on AdamW for correctness bring-up.
- Phase 2: add Muon support for batched weights with explicit flattening strategy and matching distributed path.
- Ensure both `MuonAdamW` and `DistMuonAdamW` behavior are defined for new parameter groups.

### E) `nanochat/engine.py`
- Phase 2: KV cache layout update for branch-flattened batch (`N*R`) where needed.
- Preserve non-branched inference path unchanged.

## Testing Plan (Rigorous)

### Unit tests
- [ ] `BatchedLinear` matches equivalent per-branch `nn.Linear` outputs.
- [ ] Gradient check for `BatchedLinear` (finite gradients, shape checks).
- [ ] `n_branches=1` parity smoke test against baseline path (forward loss and backward).
- [ ] Attention reshape/unreshape roundtrip correctness.
- [ ] Parameter counting test for at least `1x10` and `2x6`.
- [ ] Checkpoint load test: old checkpoints without `n_branches` still load.
- [x] Config-plumbing tests: `n_branches` defaulting/patching and explicit guard for unsupported `n_branches>1`.

### Integration tests
1. Short train smoke on CPU (very small model): no crashes, finite loss.
2. Short train smoke on CUDA: compile path works, no graph breaks from branch reshape logic.

### Benchmark tests
1. Throughput benchmark at fixed `N`, `T`, and total batch:
   - compare `12x1` vs `2x5` vs `1x10`
2. Track:
   - `train/tok_per_sec`
   - `train/mfu`
   - peak memory

## Experiment Plan (Two-Stage)

### Stage 0: Correctness bring-up
- Run only `12x1` and `6x2` for short checks.
- Exit criteria: all tests pass, no instability.

### Stage 1: Short pilot sweep (all configs)
- Run each config for a short pilot budget (recommended: 250M tokens each).
- Keep eval cheap and frequent enough to rank trends (e.g. val bpb every 50-100 steps).
- Ranking rule:
  1. Disqualify unstable runs.
  2. Disqualify runs >5% slower than baseline unless loss is clearly better.
  3. Rank remaining by val bpb at equal token budget.

### Stage 2: Long runs (top candidates only)
- Promote top 2-3 configs from pilots.
- Run 1-2B token training for these only.
- Compare convergence and final quality against baseline.

## Risks and Mitigations
- Optimizer complexity for 3D batched weights: stage Muon support after correctness path is stable.
- Hidden shape bugs in cache/attention: lock in shape contracts and unit tests first.
- Parameter-count drift vs intended budget: gate configs with `num_scaling_params()` checks before training.
- Benchmark noise: use fixed seeds, warmup steps, and identical data/loader settings for fair comparisons.

## Deliverables
1. Branch-capable training code path with `n_branches` config.
2. Tests covering correctness, checkpoint compatibility, and parameter counting.
3. Throughput report for baseline and key breadth-heavy configs.
4. Pilot sweep table with speed/quality ranking and selected finalists.
