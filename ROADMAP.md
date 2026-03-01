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
  - [x] Add backend probe diagnostics (`fa4_probe`/`fa3_probe`) to backend status logs and smoke artifacts so Blackwell fallback root cause is auditable from checked-in evidence.
  - [x] Add automated Blackwell load-failure fallback coverage (sm100 + FA4 load failure => SDPA fallback).
  - [x] Add reproducible backend smoke command (`python -m scripts.flash_backend_smoke --expect-backend fa4 --require-cuda --require-blackwell`) so on-device validation emits a canonical `selected=...` log line.
  - [x] Add machine-readable smoke artifact output (`--output-json`) so one-device validation can persist canonical backend-selection evidence alongside logs.
  - [x] Add explicit status-line artifact output (`--output-status-line`) so on-device validation can persist the canonical `selected=...` line without scraping stdout.
  - [x] Add smoke-artifact validation helper (`python -m scripts.validate_blackwell_smoke_artifact --artifact-json ... --expect-backend fa4 --require-blackwell`) and include CUDA device metadata in smoke JSON for auditable evidence checks.
  - [ ] Run one on-device smoke on RTX 5090 and record log line with `selected=fa4`.
    - [x] Add explicit RTX 5090 device-name preflight enforcement to `scripts.flash_backend_smoke` (`--require-device-substring`) and thread it through `scripts.run_blackwell_smoke_bundle` so non-5090 runs fail fast before emitting evidence artifacts.
    - [x] Add CUDA preflight diagnostics in `scripts.flash_backend_smoke` so on-device smoke failures report `nvidia-smi` GPU visibility and actionable PyTorch CUDA mismatch guidance when `torch.cuda.is_available()` is false.
    - [x] Add single-command artifact bundle output (`--output-dir`) in `scripts.flash_backend_smoke` so on-device validation can emit canonical JSON + status-line evidence in one run.
    - [x] Add status-line/JSON consistency validation support in `scripts.validate_blackwell_smoke_artifact` (`--status-line-file`) so on-device evidence checks fail fast when bundle artifacts drift.
    - [x] Add check-in-ready evidence markdown output in `scripts.validate_blackwell_smoke_artifact` (`--output-evidence-md`) so one-device runs can record canonical `selected=...` plus device metadata in a reviewable artifact.
    - [x] Add single-command Blackwell bundle runner (`python -m scripts.run_blackwell_smoke_bundle --output-dir ...`) that executes smoke capture + artifact validation + evidence markdown emission in one step.
    - [ ] Execute the bundle runner on target RTX 5090 and check in emitted evidence artifacts.
      - [x] Add runbook artifact output (`--output-runbook-md`) in `scripts.run_blackwell_smoke_bundle` so RTX 5090 operators can persist the exact execution/check-in checklist with smoke artifacts.
      - [ ] Execute the documented runbook flow on target RTX 5090 and check in emitted evidence artifacts.
         - [x] Harden generated runbook commands with shell-escaped paths so RTX 5090 operators can execute/check in artifact flows from directories containing spaces without manual command edits.
          - [x] Add `--dry-run` mode in `scripts.run_blackwell_smoke_bundle` so operators can emit/check runbook + canonical artifact paths before RTX 5090 execution without requiring CUDA.
          - [x] Add resolved smoke-bundle command artifact output (`--output-bundle-command-sh`) in `scripts.run_blackwell_smoke_bundle` so RTX 5090 operators can persist the exact invocation (with resolved artifact/checker paths) before on-device execution.
          - [x] Add runtime blocked-receipt emission in `scripts.run_blackwell_smoke_bundle` so failed on-device smoke attempts still persist reviewable blocker diagnostics (`blackwell_smoke_blocked.md`) with selected backend/status context.
          - [x] Add a resolved-invocation section to generated Blackwell runbooks so `blackwell_smoke_runbook.md` records the exact smoke command (including optional checker flags/receipt paths) for copy/paste on RTX 5090 hosts.
         - [x] Execute the runbook smoke command on the RTX 5090 host and check in the runtime blocker receipt showing current PyTorch CUDA mismatch diagnostics (`python -m scripts.run_blackwell_smoke_bundle --output-dir artifacts/blackwell/rtx5090_smoke_20260228 --expect-backend fa4 --require-device-substring "RTX 5090" --run-bundle-check` -> `artifacts/blackwell/rtx5090_smoke_20260228/blackwell_smoke_blocked.md`).
          - [x] Add smoke-bundle environment preflight receipts (`--preflight`, `--output-preflight-json`) in `scripts.run_blackwell_smoke_bundle` so operators can capture auditable blocker diagnostics before RTX 5090 execution.
         - [x] Enrich Blackwell smoke-bundle preflight receipts with CUDA device metadata plus `nvidia-smi` inventory/error fields so blocked RTX 5090 runs preserve actionable environment evidence in one artifact.
         - [x] Enrich CUDA-unavailable smoke diagnostics with PyTorch runtime metadata (`torch.__version__`, `torch.version.cuda`, `CUDA_VISIBLE_DEVICES`) plus explicit `nvidia-smi` probe-failure details so blocked RTX 5090 runs surface actionable mismatch evidence directly in failure messages.
         - [x] Add blocked-preflight markdown receipt output (`--output-blocked-md`) in `scripts.run_blackwell_smoke_bundle` so failed preflight runs produce a review-friendly blocker artifact alongside machine-readable diagnostics.
         - [x] Capture and check in an RTX 5090 preflight blocker receipt when CUDA/PyTorch mismatch prevents FA4 smoke execution (`artifacts/blackwell/rtx5090_smoke_20260228/{blackwell_smoke_preflight.json,blackwell_smoke_blocked.md}`).
         - [x] Add offline Blackwell evidence-bundle checker (`python -m scripts.check_blackwell_evidence_bundle --bundle-dir ... --expect-backend fa4`) so checked-in artifacts can be validated for completeness/consistency without requiring GPU access.
        - [x] Add git-tracked bundle validation mode (`--require-git-tracked`) and require the checker command in generated runbooks so check-in reviews can enforce offline evidence verification from committed artifacts.
        - [ ] Run the checker against emitted RTX 5090 artifacts during check-in.
            - [x] Run strict checker preflight against the emitted RTX 5090 blocker bundle and check in the missing-artifact receipt (`python -m scripts.check_blackwell_evidence_bundle --bundle-dir artifacts/blackwell/rtx5090_smoke_20260228 --expect-backend fa4 --check-in --require-real-bundle --preflight --output-preflight-json artifacts/blackwell/rtx5090_smoke_20260228/blackwell_bundle_preflight_check.json`).
            - [ ] Execute strict checker without `--preflight` once a full RTX 5090 smoke bundle is emitted.
            - [x] Add strict-helper auto-discovery receipt output (`--output-discovery-json`) in `scripts.run_blackwell_check_in` so `--bundle-dir auto` check-in attempts persist resolved-bundle/rejection diagnostics for audit.
            - [x] Add optional strict post-smoke check-in execution in `scripts.run_blackwell_smoke_bundle` (`--run-strict-check-in`, `--output-strict-check-json`) so on-device runs can emit both standard and strict checker receipts in one command.
            - [x] Add strict check-in mode (`--check-in`) in `scripts.check_blackwell_evidence_bundle` and wire the generated runbook command to use it so check-in verification consistently enforces Blackwell capability plus git-tracked artifacts.
            - [x] Add machine-readable checker receipt output (`--output-check-json`) and include it in the generated runbook check-in command so check-in reviews can verify the exact checker invocation/result from a committed artifact.
            - [x] Add single-command strict check-in helper (`python -m scripts.run_blackwell_check_in --bundle-dir ... --expect-backend fa4`) that defaults `--output-check-json` to `<bundle-dir>/blackwell_bundle_check.json` and always enforces checker `--check-in` requirements.
            - [x] Cross-validate evidence markdown provenance fields (`selected_backend`, `generated_at_utc`, `git_commit`, `status_line_ok`) against `flash_backend_smoke.json` in strict bundle checks so check-in review fails fast on drifted evidence summaries.
            - [x] Add strict-helper preflight receipt output (`--output-preflight-json`) in `scripts.run_blackwell_check_in` so check-in operators can persist auditable blocker diagnostics (including auto-discovery failures) before strict validation.
              - [x] Execute `python -m scripts.check_blackwell_evidence_bundle --bundle-dir ... --expect-backend fa4 --check-in` against emitted RTX 5090 artifacts during check-in.
                 - [x] Add checker blocked-receipt markdown output (`--output-blocked-md`) in `scripts.check_blackwell_evidence_bundle` so direct strict checker attempts persist review-friendly diagnostics when real RTX 5090 artifacts are unavailable.
                 - [x] Capture strict-checker runtime blocker receipt from emitted RTX 5090 preflight artifacts (`python -m scripts.check_blackwell_evidence_bundle --bundle-dir artifacts/blackwell/rtx5090_smoke_20260228 --expect-backend fa4 --check-in --output-check-json artifacts/blackwell/rtx5090_smoke_20260228/blackwell_bundle_check.json --output-blocked-md artifacts/blackwell/rtx5090_smoke_20260228/blackwell_bundle_check_blocked.md` -> `artifacts/blackwell/rtx5090_smoke_20260228/blackwell_bundle_check_blocked.md`).
                 - [x] Add canonical strict-checker command output in generated Blackwell smoke runbooks so operators can execute the exact `scripts.check_blackwell_evidence_bundle --check-in` invocation during check-in.
                - [x] Add optional post-smoke checker execution in `scripts.run_blackwell_smoke_bundle` (`--run-bundle-check`, `--output-check-json`) so one-device validation can emit a machine-readable checker receipt alongside smoke artifacts before strict check-in.
                - [x] Add checker preflight dry-run mode (`--dry-run`) to print resolved bundle/check settings before strict check-in execution, so operators can validate command inputs without running artifact validation.
                - [x] Add explicit bundle-file preflight mode in `scripts.check_blackwell_evidence_bundle` (`--preflight`) so operators can fail fast with actionable missing-artifact diagnostics before running strict check-in validation against RTX 5090 bundles.
                - [x] Add machine-readable preflight receipt output in `scripts.check_blackwell_evidence_bundle` (`--output-preflight-json`) so operators can persist missing-artifact diagnostics before strict check-in execution.
                - [x] Add blocked strict-check-in markdown receipt output in `scripts.run_blackwell_check_in` (`--output-blocked-md`) so failed strict checker attempts persist review-friendly diagnostics during check-in.
                - [x] Add checker auto-discovery mode (`--bundle-dir auto`, `--bundle-root`) in `scripts.check_blackwell_evidence_bundle` so strict check-in commands can resolve the latest real emitted bundle without hand-editing artifact paths.
                - [x] Add machine-readable auto-discovery receipt output in `scripts.check_blackwell_evidence_bundle` (`--output-discovery-json`) so strict check-in attempts with `--bundle-dir auto` can persist resolved-bundle/rejection diagnostics for audit.
                - [x] Add auto-discovery rejection diagnostics in `scripts.check_blackwell_evidence_bundle` so strict check-in failures list why candidate bundles were rejected (sample path, missing files, malformed JSON).
                - [x] Expand checker auto-discovery candidate scanning to include runbook-only/dry-run directories so strict check-in failures surface missing-artifact diagnostics for those emitted paths.
                - [x] Keep generated smoke runbooks/checkers consistent with custom checker-receipt paths (`--output-check-json`) so one-device evidence workflows can override receipt locations without runbook drift.
                - [x] Add `--dry-run` mode in `scripts.run_blackwell_check_in` so operators can resolve/record canonical bundle + receipt paths before strict check-in execution.
                - [x] Add strict-helper preflight mode in `scripts.run_blackwell_check_in` (`--preflight`) so operators can validate required bundle files/real-bundle guardrails before strict check-in execution.
                - [x] Add strict-helper command artifact output in `scripts.run_blackwell_check_in` (`--output-check-command-sh`) so operators can persist the fully-resolved `scripts.check_blackwell_evidence_bundle --check-in` invocation (including resolved bundle/receipt paths) before executing strict check-in.
                - [x] Add reproducible local Blackwell evidence fixture bundle (`artifacts/blackwell/sample_bundle/*`) plus regression coverage to keep `scripts.run_blackwell_check_in` receipt output in sync with checked-in artifacts.
                - [x] Add strict real-artifact guardrails to check-in validation (`--require-real-bundle` in checker + default-on in `scripts.run_blackwell_check_in`, with `--allow-sample-bundle` override for fixture regression tests) so check-in commands fail fast when pointed at sample fixture bundles.
                - [x] Add `require_real_bundle` to machine-readable checker receipts (`--output-check-json`) with regression coverage so check-in evidence captures whether strict real-artifact enforcement was active.
                - [x] Add `--bundle-dir auto` discovery mode in `scripts.run_blackwell_check_in` (with `--bundle-root`) so check-in commands can resolve the latest real emitted bundle and fail fast when only sample fixtures exist.
                - [x] Route `scripts.run_blackwell_check_in` auto-discovery through checker bundle classification so payload-marked sample artifacts (`"is_sample": true`) are rejected with the same candidate-rejection diagnostics as strict checker auto-discovery.
                - [x] Add per-artifact SHA256 digests to checker receipts (`artifact_sha256`) so strict check-in evidence binds the exact JSON/status/evidence/runbook files reviewed.
                - [x] Add payload-level sample-fixture guardrails (`is_sample`) in Blackwell smoke artifacts/checkers so strict real-bundle validation rejects relabeled sample JSON even outside `sample_bundle` paths.
                - [x] Add smoke-artifact provenance fields (`generated_at_utc`, `git_commit`) and enforce them in artifact validation/evidence checks so check-in reviews can verify when and from which repo revision each bundle was produced.
                - [x] Add strict RTX 5090 device-name guardrails (`--require-device-substring`, default `RTX 5090` for check-in helpers) so strict bundle validation fails fast when artifacts come from non-5090 hardware.
                - [x] Add markdown check-in evidence output in `scripts.run_blackwell_check_in` (`--output-check-md`) so strict checker runs can emit a review-friendly summary alongside machine-readable checker receipts.
                - [x] Enforce `nvidia-smi` provenance in strict Blackwell check-in validation (artifact must carry successful GPU inventory lines matching required device substring), with regression coverage, so check-in receipts bind backend selection to auditable on-device inventory evidence.
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
  - [x] Add an opt-in on-device integration test (`tests/test_blackwell_on_device_smoke.py`) that runs `python -m scripts.flash_backend_smoke --expect-backend fa4 --require-cuda --require-blackwell --require-device-substring "RTX 5090" --output-dir ...` and asserts artifact/status evidence.
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
      - [x] Check in canonical Stage 1 target-execution planning artifacts by running `python -m scripts.pilot_sweep --device-type cuda --total-batch-size 524288 --device-batch-size 16 --pilot-tokens 250000000 --eval-every 75 --eval-tokens 1048576 --artifacts-dir artifacts/pilot/target_sweep_20260228 --preflight --output-preflight-json artifacts/pilot/pilot_stage1_preflight_20260228.json --output-launch-manifest-json artifacts/pilot/pilot_stage1_launch_manifest_20260228.json` and `python -m scripts.pilot_sweep --device-type cuda --total-batch-size 524288 --device-batch-size 16 --pilot-tokens 250000000 --eval-every 75 --eval-tokens 1048576 --artifacts-dir artifacts/pilot/target_sweep_20260228 --dry-run --output-runbook-md artifacts/pilot/pilot_stage1_runbook_20260228.md`, then check in the emitted full-grid preflight/manifest/runbook artifacts for operators.
      - [x] Add pilot sweep launch-manifest output (`--output-launch-manifest-json`) that records canonical per-config commands, global config indices, token budgets, and expected artifact paths so target-GPU operators can stage/shard real Stage 1 execution without ad-hoc command reconstruction.
      - [x] Add resumable sweep support (`--resume-from-artifacts`) so interrupted long pilot runs can continue from existing per-config JSON artifacts.
        - [ ] Use resume mode for the real target-GPU sweep and persist final ranking/finalist artifacts.
              - [x] Add pilot-sweep finalist artifact outputs (`--max-finalists`, `--output-finalists-json`, `--output-finalists-md`) so resumed sweeps can persist promotion-ready evidence in one run.
              - [x] Add resume-artifact preflight validation for log presence plus required metrics/token-budget consistency, so target-GPU resume runs fail fast on incomplete or stale artifacts.
              - [x] Add generated pilot sweep runbook output (`--output-runbook-md`) that records canonical initial-run, resume, and strict check-in commands with artifact paths so target-GPU operators can execute/check in reproducibly.
              - [x] Add partial target execution mode (`--target`) in `scripts/pilot_sweep.py` with canonical config ordering/global artifact indices, so multi-GPU operators can shard long sweeps and later aggregate via full-grid resume without artifact-name drift.
              - [x] Add pilot-sweep execution preflight mode (`--preflight`, `--output-preflight-json`) so target-GPU operators can validate planned commands and resumable artifacts (without launching training) before long Stage 1 runs.
              - [x] Add offline pilot artifact-bundle checker (`python -m scripts.check_pilot_sweep_artifacts --ranked-json ... --finalists-json ... --finalists-md ...`) so check-in review can validate ranking/finalist artifact consistency without rerunning long GPU sweeps.
             - [ ] Execute resume mode on target GPU(s) and check in the resulting ranking/finalist artifacts.
                - [x] Add blocked-receipt markdown output in `scripts.pilot_sweep.py` (`--output-blocked-md`) so failed preflight/runtime attempts can persist check-in-friendly diagnostics when target-GPU execution is unavailable.
                - [x] Auto-create parent directories for pilot ranking/finalist outputs (`--output-json`, `--output-md`, `--output-finalists-json`, `--output-finalists-md`, `--output-runbook-md`) so resumed target-GPU sweeps do not fail on missing nested report directories.
                 - [ ] Run the pilot artifact-bundle checker against the emitted real target-GPU ranking/finalist artifacts during check-in.
                     - [x] Capture the current direct-checker blocker by running `python -m scripts.check_pilot_sweep_artifacts --artifacts-dir auto --artifacts-root artifacts/pilot --check-in --dry-run --output-blocked-md artifacts/pilot/pilot_bundle_checker_blocked_20260228.md`, and check in the blocked-receipt artifact showing that no real Stage 1 ranking/finalist bundle is available yet.
                     - [ ] Re-run the checker in strict mode (without `--dry-run`) once real target-GPU ranking/finalist artifacts are present, and check in the resulting validation receipt.
                     - [x] Add pilot artifact-checker auto-discovery mode (`--artifacts-dir auto`, `--artifacts-root`) with candidate-rejection diagnostics so strict check-in commands can target the latest real bundle without hand-editing artifact paths.
                     - [x] Add machine-readable pilot bundle checker receipt output (`--output-check-json`) so check-in reviews can verify the exact checker invocation/result from a committed artifact.
                     - [x] Add per-artifact SHA256 digests to pilot bundle checker receipts (`artifact_sha256`) so check-in evidence binds the exact ranked/finalists artifacts reviewed.
                     - [x] Add pilot artifact-checker preflight dry-run mode (`--dry-run`) that resolves artifact inputs and prints a canonical status line without executing validation, so operators can verify checker invocation inputs before strict check-in runs.
                     - [x] Add checker blocked-receipt markdown output (`--output-blocked-md`) in `scripts.check_pilot_sweep_artifacts` so direct strict-checker attempts can persist review-friendly diagnostics when validation fails before real artifacts are available.
                     - [x] Add single-command strict pilot check-in helper (`python -m scripts.run_pilot_check_in --artifacts-dir ...`) that defaults `--output-check-json` to `<artifacts-dir>/pilot_bundle_check.json` and always enforces checker `--check-in` requirements.
                  - [x] Mark sample pilot ranking fixtures with `"is_sample": true` and add check-in regression coverage so `--check-in` rejects relabeled sample artifacts via payload metadata, not filename alone.
                  - [ ] Execute `python -m scripts.run_pilot_check_in --artifacts-dir ...` against emitted real target-GPU ranking/finalist artifacts during check-in.
                      - [x] Capture the current strict-check-in blocker by running `python -m scripts.run_pilot_check_in --artifacts-dir auto --artifacts-root artifacts/pilot --preflight --output-discovery-json artifacts/pilot/pilot_check_in_discovery_20260228.json --output-preflight-json artifacts/pilot/pilot_check_in_preflight_20260228.json --output-blocked-md artifacts/pilot/pilot_check_in_blocked_20260228.md`, and check in the emitted discovery + blocked-receipt artifacts showing no real target-GPU pilot bundle is available yet.
                      - [x] Harden strict-helper auto-discovery candidate filtering to reject payload-marked sample ranked artifacts and malformed ranking JSON with explicit rejection diagnostics, so check-in commands fail fast before invoking strict validation on non-real bundles.
                      - [x] Add strict-helper promotion bundle receipt passthrough (`--bundle-json`, `--bundle-json-name`) so `scripts.run_pilot_check_in` can validate Stage 2 promotion receipts during strict check-in without dropping to the lower-level checker command.
                      - [x] Add explicit sample-fixture override to strict helper (`--allow-sample-input`) while defaulting to real-input enforcement, with regression coverage for both default and override paths.
                      - [x] Fix strict-helper sample override plumbing so `--allow-sample-input` bypasses only the real-input guard (while preserving `--check-in` git-tracked checks) for fixture-based end-to-end validation.
                      - [x] Add auto-discovery mode to strict helper (`--artifacts-dir auto`, `--artifacts-root`) so check-in commands can resolve the latest real artifact bundle and fail fast when only sample directories exist.
                      - [x] Expand strict-helper/checker auto-discovery candidate scanning to include runbook-only artifact directories (`*runbook*.md`) so missing ranked/finalists files are surfaced as explicit rejection diagnostics instead of a generic "no bundle found" failure.
                    - [x] Improve auto-discovery failure diagnostics in strict helper to report rejected artifact-bundle candidates (for example sample-path and missing-file reasons), so target-GPU check-in operators can fix bundle issues without trial-and-error.
                    - [x] Add strict-helper dry-run mode (`--dry-run`) that resolves artifact/check-receipt paths and prints a canonical preflight status line without executing checker validation, so target-GPU operators can verify invocation inputs before running check-in.
                    - [x] Add strict-helper resolved-command artifact output (`--output-check-command-sh`) in `scripts.run_pilot_check_in` so target-GPU operators can persist the exact strict helper invocation (with resolved paths/options) before executing check-in.
                     - [x] Add strict-helper preflight mode (`--preflight`) in `scripts.run_pilot_check_in` so operators can fail fast on missing required artifacts (and optional promotion bundle receipt) before running strict check-in.
                    - [x] Add machine-readable strict-helper preflight receipt output (`--output-preflight-json`) in `scripts.run_pilot_check_in` so operators can check in auditable preflight diagnostics before strict validation.
                     - [x] Add blocked-check-in markdown receipt output (`--output-blocked-md`) in `scripts.run_pilot_check_in` so failed strict check-in attempts can persist review-friendly blocker evidence (including auto-discovery and preflight failures) during target-GPU check-in.
                     - [x] Add optional markdown evidence output in `scripts.run_pilot_check_in` (`--output-check-md`) so strict check-in runs can emit a review-friendly summary alongside machine-readable checker receipts.
                     - [x] Add machine-readable strict-helper auto-discovery receipt output (`--output-discovery-json`) in `scripts.run_pilot_check_in` so target-GPU check-in attempts can persist resolved-bundle/rejection diagnostics for audit.
                     - [x] Enforce full default pilot-grid coverage in strict check-in validation (all canonical configs present exactly once with expected depth/branch/aspect tuples) so target-GPU check-in fails fast on partial sweeps before promotion.
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
        - [x] Add checker bundle-receipt auto-resolution support in `scripts.check_pilot_sweep_artifacts` (`--bundle-json auto`, `--bundle-json-name`) with regression coverage, so strict check-in commands can validate promotion receipts from artifact directories without hand-editing receipt paths.
          - [ ] Run promotion helper on real pilot output JSON and record selected finalist configs in repo artifacts.
             - [x] Add single-command promotion bundle runner (`python -m scripts.run_stage2_promotion_bundle --input-json ... --output-dir artifacts/pilot`) so real ranked outputs can emit canonical Stage 2 finalists JSON/markdown artifacts in one run.
            - [x] Add `--require-real-input` guardrails to `scripts.pilot_promote` and `scripts.run_stage2_promotion_bundle` (with regression tests) so check-in commands fail fast when pointed at sample fixture ranked-run JSON.
            - [x] Add promotion-bundle runbook artifact output (`--output-runbook-md`) in `scripts.run_stage2_promotion_bundle` so operators can record canonical promotion + strict check-in commands with emitted artifact paths.
             - [x] Add optional strict post-promotion check-in execution (`--run-check-in`, `--output-check-json`) in `scripts.run_stage2_promotion_bundle` so one command can emit finalists artifacts and a check-in receipt for review.
             - [x] Harden generated promotion runbook commands with shell-escaped artifact paths (including strict check-in receipt paths) so target-GPU operators can execute/check in Stage 2 promotion flows from directories containing spaces without manual command edits.
              - [x] Add real-artifact auto-discovery mode to `scripts.run_stage2_promotion_bundle` (`--input-json auto`, `--input-root`, `--input-json-name`) so promotion/check-in operators can resolve the latest non-sample `pilot_ranked_runs.json` bundle without hand-editing paths.
              - [x] Fix Stage 2 promotion runbook check-in path handling by emitting absolute shell-escaped ranked/finalist paths so strict check-in commands work when `--input-json` is repo-relative or outside `--output-dir`.
              - [x] Add promotion-bundle preflight dry-run mode (`--dry-run`) so operators can resolve/record canonical ranked-input, finalists-output, optional check-receipt, and runbook paths before executing real target-GPU promotion/check-in runs.
              - [x] Harden promotion-bundle auto-discovery candidate filtering to reject payload-marked sample JSON and malformed ranked artifacts with explicit rejection diagnostics, so operators can safely resolve real ranked outputs before promotion.
              - [ ] Execute promotion bundle runner on real pilot output JSON and check in emitted Stage 2 finalist artifacts.
                - [x] Add machine-readable promotion-bundle receipt output (`--output-bundle-json`) in `scripts.run_stage2_promotion_bundle` so real execution/check-in reviews can capture a canonical invocation summary plus SHA256 digests for emitted finalists artifacts.
                - [ ] Run promotion bundle runner on real pilot output JSON with `--output-bundle-json` and check in emitted finalists + receipt artifacts.
                   - [x] Add offline checker support for validating Stage 2 promotion bundle receipts (`--bundle-json`) against ranked/finalists artifacts and receipt SHA256 metadata, with regression coverage.
                   - [x] Add optional dry-run runbook emission in `scripts.run_stage2_promotion_bundle` (`--dry-run-write-runbook`) so target-GPU operators can persist canonical promotion/check-in commands before real execution.
                   - [x] Add review-friendly promotion evidence markdown output in `scripts.run_stage2_promotion_bundle` (`--output-evidence-md`) so real Stage 2 promotion/check-in runs can emit a canonical summary alongside finalists and machine-readable receipts.
                   - [x] Add promotion-bundle preflight validation mode in `scripts.run_stage2_promotion_bundle` (`--preflight`, `--output-preflight-json`) so operators can validate ranked-input/finalist-selection constraints and persist machine-readable readiness evidence before real target-GPU execution.
                   - [x] Add resolved bundle-command artifact output in `scripts.run_stage2_promotion_bundle` (`--output-bundle-command-sh`) so target-GPU operators can persist the exact real execution command (with fully-resolved paths/options) before running promotion/check-in.
                   - [ ] Execute promotion bundle runner on real pilot output JSON with `--output-bundle-json` and check in emitted finalists + receipt artifacts.
                       - [x] Add blocked-run markdown receipt output (`--output-blocked-md`) in `scripts.run_stage2_promotion_bundle` so failed real-input executions still emit check-in-friendly blocker diagnostics with the exact invocation and planned artifact paths.
                       - [x] Add machine-readable auto-discovery receipt output (`--output-discovery-json`) in `scripts.run_stage2_promotion_bundle` so `--input-json auto` promotion attempts persist resolved-input/rejection diagnostics for audit on both success and blocked runs.
                       - [x] Capture the current promotion-bundle execution blocker by running `python -m scripts.run_stage2_promotion_bundle --input-json auto --input-root artifacts/pilot --output-dir artifacts/pilot --require-real-input --output-bundle-json artifacts/pilot/stage2_promotion_bundle_20260228.json --output-evidence-md artifacts/pilot/stage2_promotion_evidence_20260228.md --output-blocked-md artifacts/pilot/stage2_promotion_blocked_20260228.md --output-discovery-json artifacts/pilot/stage2_promotion_discovery_20260228.json`, and check in the emitted blocker/discovery artifacts showing no real `pilot_ranked_runs.json` bundle is available yet.
- [ ] Run 1-2B token training for these only.
  - [x] Add Stage 2 long-run planner (`python -m scripts.plan_stage2_long_runs --finalists-json ...`) that emits canonical 1B/2B `scripts.base_train` commands plus plan/runbook artifacts from promoted finalists.
  - [x] Add single-command Stage 2 long-run bundle runner (`python -m scripts.run_stage2_long_run_bundle --finalists-json ... --output-dir ...`) with preflight/dry-run receipts plus blocked markdown output so target-GPU operators can capture canonical plan/runbook command artifacts before execution.
  - [ ] Execute generated long-run commands on target GPU(s) and check in resulting training/eval artifacts.
- [ ] Compare convergence and final quality against baseline.
  - [x] Add offline Stage 2 comparison helper (`python -m scripts.compare_stage2_long_runs --input-json ...`) that validates long-run metrics and emits baseline-relative JSON/markdown reports for review.
  - [ ] Run the comparison helper on real Stage 2 training artifacts and check in the emitted baseline comparison report.

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
