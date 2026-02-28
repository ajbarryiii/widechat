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
- [x] Add branch-aware block modules (attention + MLP) for shape `(N,T,R,C)`.
  - [x] Add `ParallelMLP` using `BatchedLinear` with `relu^2` activation, plus unit test parity vs per-branch `nn.Linear` reference.
  - [x] Add branch-aware attention module with explicit `(N,T,R,*) <-> (N*R,T,*)` reshape contract.
  - [x] Compose branch-aware block (`ParallelBlock`) with residual updates for `(N,T,R,C)`.
- [x] Keep baseline path for `n_branches=1` as close as possible to current behavior.
- [x] Update `forward()` with:
  - [x] `linear_in -> branch blocks -> linear_out`
- [x] Ensure `num_scaling_params()` and `estimate_flops()` account for new matrices.

### B) `scripts/base_train.py`
- [x] Add `--n-branches` CLI arg.
- [x] Pass `n_branches` into `GPTConfig`.
- [x] Update default `model_tag` format to avoid collisions (e.g. `d{depth}b{branches}`).

### C) `nanochat/checkpoint_manager.py`
- [x] Backward compatibility patch for missing config key:
  - [x] if `n_branches` absent in old checkpoints, default to `1`.

### D) `nanochat/optim.py` and `nanochat/gpt.py`
- [x] Phase 1 (simplest): put new batched branch projection weights on AdamW for correctness bring-up.
- [x] Phase 2: add Muon support for batched weights with explicit flattening strategy and matching distributed path.
- [x] Ensure both `MuonAdamW` and `DistMuonAdamW` behavior are defined for new parameter groups.

### E) `nanochat/engine.py`
- [x] Phase 2: KV cache layout update for branch-flattened batch (`N*R`) where needed.
- [x] Preserve non-branched inference path unchanged.

### F) `nanochat/flash_attention.py` (Blackwell prerequisite)
- [x] Migrate runtime backend from Flash Attention 3-first to Flash Attention 4-first.
- [ ] Target NVIDIA Blackwell (RTX 5090) as primary path, with SDPA fallback retained.
  - [x] Add automated Blackwell load-failure fallback coverage (sm100 + FA4 load failure => SDPA fallback).
  - [x] Add reproducible backend smoke command (`python -m scripts.flash_backend_smoke --expect-backend fa4 --require-cuda --require-blackwell`) so on-device validation emits a canonical `selected=...` log line.
  - [x] Add machine-readable smoke artifact output (`--output-json`) so one-device validation can persist canonical backend-selection evidence alongside logs.
  - [x] Add explicit status-line artifact output (`--output-status-line`) so on-device validation can persist the canonical `selected=...` line without scraping stdout.
  - [x] Add smoke-artifact validation helper (`python -m scripts.validate_blackwell_smoke_artifact --artifact-json ... --expect-backend fa4 --require-blackwell`) and include CUDA device metadata in smoke JSON for auditable evidence checks.
  - [ ] Run one on-device smoke on RTX 5090 and record log line with `selected=fa4`.
    - [x] Add single-command artifact bundle output (`--output-dir`) in `scripts.flash_backend_smoke` so on-device validation can emit canonical JSON + status-line evidence in one run.
    - [x] Add status-line/JSON consistency validation support in `scripts.validate_blackwell_smoke_artifact` (`--status-line-file`) so on-device evidence checks fail fast when bundle artifacts drift.
    - [x] Add check-in-ready evidence markdown output in `scripts.validate_blackwell_smoke_artifact` (`--output-evidence-md`) so one-device runs can record canonical `selected=...` plus device metadata in a reviewable artifact.
    - [x] Add single-command Blackwell bundle runner (`python -m scripts.run_blackwell_smoke_bundle --output-dir ...`) that executes smoke capture + artifact validation + evidence markdown emission in one step.
    - [ ] Execute the bundle runner on target RTX 5090 and check in emitted evidence artifacts.
      - [x] Add runbook artifact output (`--output-runbook-md`) in `scripts.run_blackwell_smoke_bundle` so RTX 5090 operators can persist the exact execution/check-in checklist with smoke artifacts.
      - [ ] Execute the documented runbook flow on target RTX 5090 and check in emitted evidence artifacts.
        - [x] Harden generated runbook commands with shell-escaped paths so RTX 5090 operators can execute/check in artifact flows from directories containing spaces without manual command edits.
        - [x] Add `--dry-run` mode in `scripts.run_blackwell_smoke_bundle` so operators can emit/check runbook + canonical artifact paths before RTX 5090 execution without requiring CUDA.
        - [x] Add offline Blackwell evidence-bundle checker (`python -m scripts.check_blackwell_evidence_bundle --bundle-dir ... --expect-backend fa4`) so checked-in artifacts can be validated for completeness/consistency without requiring GPU access.
        - [x] Add git-tracked bundle validation mode (`--require-git-tracked`) and require the checker command in generated runbooks so check-in reviews can enforce offline evidence verification from committed artifacts.
        - [ ] Run the checker against emitted RTX 5090 artifacts during check-in.
            - [x] Add strict check-in mode (`--check-in`) in `scripts.check_blackwell_evidence_bundle` and wire the generated runbook command to use it so check-in verification consistently enforces Blackwell capability plus git-tracked artifacts.
            - [x] Add machine-readable checker receipt output (`--output-check-json`) and include it in the generated runbook check-in command so check-in reviews can verify the exact checker invocation/result from a committed artifact.
            - [x] Add single-command strict check-in helper (`python -m scripts.run_blackwell_check_in --bundle-dir ... --expect-backend fa4`) that defaults `--output-check-json` to `<bundle-dir>/blackwell_bundle_check.json` and always enforces checker `--check-in` requirements.
            - [ ] Execute `python -m scripts.check_blackwell_evidence_bundle --bundle-dir ... --expect-backend fa4 --check-in` against emitted RTX 5090 artifacts during check-in.
              - [x] Add checker auto-discovery mode (`--bundle-dir auto`, `--bundle-root`) in `scripts.check_blackwell_evidence_bundle` so strict check-in commands can resolve the latest real emitted bundle without hand-editing artifact paths.
              - [x] Add `--dry-run` mode in `scripts.run_blackwell_check_in` so operators can resolve/record canonical bundle + receipt paths before strict check-in execution.
              - [x] Add reproducible local Blackwell evidence fixture bundle (`artifacts/blackwell/sample_bundle/*`) plus regression coverage to keep `scripts.run_blackwell_check_in` receipt output in sync with checked-in artifacts.
              - [x] Add strict real-artifact guardrails to check-in validation (`--require-real-bundle` in checker + default-on in `scripts.run_blackwell_check_in`, with `--allow-sample-bundle` override for fixture regression tests) so check-in commands fail fast when pointed at sample fixture bundles.
              - [x] Add `require_real_bundle` to machine-readable checker receipts (`--output-check-json`) with regression coverage so check-in evidence captures whether strict real-artifact enforcement was active.
              - [x] Add `--bundle-dir auto` discovery mode in `scripts.run_blackwell_check_in` (with `--bundle-root`) so check-in commands can resolve the latest real emitted bundle and fail fast when only sample fixtures exist.
- [x] Keep backend selection explicit in logs so benchmarks confirm FA4 is actually active.

## Testing Plan (Rigorous)

### Unit tests
- [x] `BatchedLinear` matches equivalent per-branch `nn.Linear` outputs.
- [x] Gradient check for `BatchedLinear` (finite gradients, shape checks).
- [x] `n_branches=1` parity smoke test against baseline path (forward loss and backward).
- [x] Attention reshape/unreshape roundtrip correctness.
- [x] Parameter counting test for at least `1x10` and `2x6`.
- [x] Checkpoint load test: old checkpoints without `n_branches` still load.
- [x] Config-plumbing tests: `n_branches` defaulting/patching and explicit guard for unsupported `n_branches>1`.

### Integration tests
- [x] Short train smoke on CPU (very small model): no crashes, finite loss.
- [x] Short train smoke on CUDA: compile path works, no graph breaks from branch reshape logic.
- [ ] Flash backend smoke on Blackwell: verify Flash Attention 4 path is selected (not SDPA fallback).
  - [x] Add backend-selection tests that simulate Blackwell (`sm100`) and assert FA4 is preferred/loaded.
  - [x] Add reusable smoke script + unit tests for backend-status parsing and Blackwell environment gating.
  - [x] Add status-line artifact output coverage so Blackwell smoke evidence can be persisted as a canonical one-line log file.
  - [x] Add artifact-validation tests for recorded smoke JSON (backend expectation + Blackwell capability checks).
  - [x] Add artifact-validation coverage for status-line/JSON bundle consistency checks.
  - [x] Add one-command smoke-bundle test coverage (`scripts.run_blackwell_smoke_bundle`) so RTX 5090 validation workflow remains reproducible and regression-tested.
  - [ ] Run one on-device smoke on RTX 5090 and record log line with `selected=fa4`.

### Benchmark tests
- [x] Throughput benchmark at fixed `N`, `T`, and total batch.
  - [x] Compare `12x1` vs `2x5` vs `1x10`.
- [x] Track benchmark metrics.
  - [x] `train/tok_per_sec`
  - [x] `train/mfu`
  - [x] Peak memory.

## Experiment Plan (Two-Stage)

### Stage 0: Correctness bring-up
- [x] Run only `12x1` and `6x2` for short checks.
- [x] Exit criteria: all tests pass, no instability.

### Stage 1: Short pilot sweep (all configs)
- [ ] Run each config for a short pilot budget (recommended: 250M tokens each).
   - [x] Add pilot sweep automation script (`scripts/pilot_sweep.py`) that runs the full config grid with fixed pilot token budget and eval cadence.
   - [x] Add pilot reporting output (`--output-md`) that writes a ranking table plus selected finalists for Stage 2 promotion decisions.
   - [x] Add per-config artifact capture (`--artifacts-dir`) so each pilot run saves raw logs and per-config metrics JSON for later audit/ranking.
   - [ ] Execute the full pilot sweep on target GPU(s) and collect per-config logs/artifacts.
      - [x] Add resumable sweep support (`--resume-from-artifacts`) so interrupted long pilot runs can continue from existing per-config JSON artifacts.
        - [ ] Use resume mode for the real target-GPU sweep and persist final ranking/finalist artifacts.
            - [x] Add pilot-sweep finalist artifact outputs (`--max-finalists`, `--output-finalists-json`, `--output-finalists-md`) so resumed sweeps can persist promotion-ready evidence in one run.
            - [x] Add resume-artifact preflight validation for log presence plus required metrics/token-budget consistency, so target-GPU resume runs fail fast on incomplete or stale artifacts.
            - [x] Add generated pilot sweep runbook output (`--output-runbook-md`) that records canonical initial-run, resume, and strict check-in commands with artifact paths so target-GPU operators can execute/check in reproducibly.
            - [x] Add offline pilot artifact-bundle checker (`python -m scripts.check_pilot_sweep_artifacts --ranked-json ... --finalists-json ... --finalists-md ...`) so check-in review can validate ranking/finalist artifact consistency without rerunning long GPU sweeps.
             - [ ] Execute resume mode on target GPU(s) and check in the resulting ranking/finalist artifacts.
             - [ ] Run the pilot artifact-bundle checker against the emitted real target-GPU ranking/finalist artifacts during check-in.
                  - [x] Add machine-readable pilot bundle checker receipt output (`--output-check-json`) so check-in reviews can verify the exact checker invocation/result from a committed artifact.
                  - [x] Add single-command strict pilot check-in helper (`python -m scripts.run_pilot_check_in --artifacts-dir ...`) that defaults `--output-check-json` to `<artifacts-dir>/pilot_bundle_check.json` and always enforces checker `--check-in` requirements.
                  - [x] Mark sample pilot ranking fixtures with `"is_sample": true` and add check-in regression coverage so `--check-in` rejects relabeled sample artifacts via payload metadata, not filename alone.
                  - [ ] Execute `python -m scripts.run_pilot_check_in --artifacts-dir ...` against emitted real target-GPU ranking/finalist artifacts during check-in.
                      - [x] Add explicit sample-fixture override to strict helper (`--allow-sample-input`) while defaulting to real-input enforcement, with regression coverage for both default and override paths.
                      - [x] Fix strict-helper sample override plumbing so `--allow-sample-input` bypasses only the real-input guard (while preserving `--check-in` git-tracked checks) for fixture-based end-to-end validation.
                      - [x] Add auto-discovery mode to strict helper (`--artifacts-dir auto`, `--artifacts-root`) so check-in commands can resolve the latest real artifact bundle and fail fast when only sample directories exist.
                      - [x] Improve auto-discovery failure diagnostics in strict helper to report rejected artifact-bundle candidates (for example sample-path and missing-file reasons), so target-GPU check-in operators can fix bundle issues without trial-and-error.
                      - [x] Add strict-helper dry-run mode (`--dry-run`) that resolves artifact/check-receipt paths and prints a canonical preflight status line without executing checker validation, so target-GPU operators can verify invocation inputs before running check-in.
- [x] Keep eval cheap and frequent enough to rank trends (e.g. val bpb every 50-100 steps).
- [x] Apply ranking rule.
   - [x] Disqualify unstable runs.
   - [x] Disqualify runs >5% slower than baseline unless loss is clearly better.
   - [x] Rank remaining by val bpb at equal token budget.

### Stage 2: Long runs (top candidates only)
- [ ] Promote top 2-3 configs from pilots.
     - [x] Add promotion helper (`python -m scripts.pilot_promote --input-json ...`) to select qualified finalists and emit Stage 2 depth/branch flags from pilot ranking artifacts.
     - [ ] Promotion artifact robustness + execution.
       - [x] Validate `stage2_finalists.json` source provenance in offline checks by requiring its `source` path to match the supplied `--ranked-json`, with regression coverage, so check-in reviews fail fast on mismatched ranking/finalist bundles.
       - [x] Enforce Stage 2 finalist-count policy in promotion tooling (`--min-finalists` default 2 with `--max-finalists` bounds checks) and cover it with regression tests so malformed promotion requests fail fast before real artifact generation.
       - [x] Add ranked-run artifact schema validation in `scripts/pilot_promote.py` so malformed pilot JSON fails fast before finalist selection.
       - [x] Add ranked-run consistency validation (`qualified`/`rank`/`disqualify_reason` coherence plus throughput/loss/token-budget types) so finalist promotion fails fast on semantically invalid ranking artifacts.
       - [x] Add reproducible local promotion fixture artifacts (`artifacts/pilot/sample_ranked_runs.json` => `sample_stage2_finalists.{json,md}`) plus an end-to-end regression test that verifies checked-in finalists stay in sync with `scripts.pilot_promote` output.
       - [x] Bind finalist artifacts to ranked-run inputs with a canonical `source_sha256` digest (emitted by promotion/sweep tooling and enforced by offline checkers) so check-in reviews fail fast on stale or tampered ranking sources.
         - [ ] Run promotion helper on real pilot output JSON and record selected finalist configs in repo artifacts.
            - [x] Add single-command promotion bundle runner (`python -m scripts.run_stage2_promotion_bundle --input-json ... --output-dir artifacts/pilot`) so real ranked outputs can emit canonical Stage 2 finalists JSON/markdown artifacts in one run.
           - [x] Add `--require-real-input` guardrails to `scripts.pilot_promote` and `scripts.run_stage2_promotion_bundle` (with regression tests) so check-in commands fail fast when pointed at sample fixture ranked-run JSON.
            - [x] Add promotion-bundle runbook artifact output (`--output-runbook-md`) in `scripts.run_stage2_promotion_bundle` so operators can record canonical promotion + strict check-in commands with emitted artifact paths.
             - [x] Add optional strict post-promotion check-in execution (`--run-check-in`, `--output-check-json`) in `scripts.run_stage2_promotion_bundle` so one command can emit finalists artifacts and a check-in receipt for review.
             - [x] Harden generated promotion runbook commands with shell-escaped artifact paths (including strict check-in receipt paths) so target-GPU operators can execute/check in Stage 2 promotion flows from directories containing spaces without manual command edits.
              - [x] Add real-artifact auto-discovery mode to `scripts.run_stage2_promotion_bundle` (`--input-json auto`, `--input-root`, `--input-json-name`) so promotion/check-in operators can resolve the latest non-sample `pilot_ranked_runs.json` bundle without hand-editing paths.
              - [x] Fix Stage 2 promotion runbook check-in path handling by emitting absolute shell-escaped ranked/finalist paths so strict check-in commands work when `--input-json` is repo-relative or outside `--output-dir`.
              - [x] Add promotion-bundle preflight dry-run mode (`--dry-run`) so operators can resolve/record canonical ranked-input, finalists-output, optional check-receipt, and runbook paths before executing real target-GPU promotion/check-in runs.
              - [ ] Execute promotion bundle runner on real pilot output JSON and check in emitted Stage 2 finalist artifacts.
- [ ] Run 1-2B token training for these only.
- [ ] Compare convergence and final quality against baseline.

## Risks and Mitigations
- Optimizer complexity for 3D batched weights: stage Muon support after correctness path is stable.
- Hidden shape bugs in cache/attention: lock in shape contracts and unit tests first.
- Parameter-count drift vs intended budget: gate configs with `num_scaling_params()` checks before training.
- Benchmark noise: use fixed seeds, warmup steps, and identical data/loader settings for fair comparisons.

## Deliverables
- [x] Branch-capable training code path with `n_branches` config.
- [x] Tests covering correctness, checkpoint compatibility, and parameter counting.
- [ ] Flash Attention 4 migration for Blackwell GPUs (RTX 5090), with verified runtime backend selection.
- [x] Throughput report for baseline and key breadth-heavy configs.
- [ ] Pilot sweep table with speed/quality ranking and selected finalists.
