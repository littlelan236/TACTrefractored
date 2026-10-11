# EasySteer backend implementation and validation

Validated during the 2026-10-10 work session through SSH in
`/home/wangtianyi/tactRefractor`. Reports and tests below belong to this project;
no other project's serving environment was upgraded or overwritten.

## Implemented

- Independent vLLM 0.29.0 / torch 2.13.0+cu130 environment and pinned fork overlay.
- Native Python and HTTP `steering` API, explicit False baseline, one VectorSpec
  for ADD/CAP/GATE, and unified project backend routing/configuration.
- Model-bound axis validation shared with registry, explicit semantic-site
  conversion, `TactBandPayload` wire/hash/model-shape/FP32 handling.
- Decoder complete-state and final-norm targets through `tact_states`.
- CAP/GATE shared band kernel and fused additive/band graph dispatch; ADD retains
  direct lowering/math. All TACT tables stay FP32; native direction dtype behavior
  remains unchanged. TACT ADD uses a separate FP32 resident-cache entry if needed.
- Config slot sharing/clear/rollback uses original native machinery; row zero,
  padding, zero correction, and NONE remain unchanged.
- Generated close all-occurrences selection; exact singleton TACT clauses now
  avoid host token-ID reads. Other selectors retain the native resolver.
- Actual replay counters begin on the first real batch, excluding warmup/profile.
  Optional GPU diagnostics are disabled in production performance measurements.
- Client capability exports and checked installation/source manifests.

## Test evidence

| Check | Result | Artifact |
|---|---|---|
| Full upstream CPU suite after extension/selector integration | 880 passed, 14 CUDA/optional skips; 15 subtests | `logs/easysteer-cpu-final.log` |
| Final payload/dtype/cache/selector regressions | 57 passed | `logs/easysteer-last-runtime.log` |
| Project tests including existing registry/framework contracts | 25 passed | `logs/easysteer-project-final.log` |
| GPU kernel graph capture/replay, FP32/BF16, H=8/2048/5120, residual/no residual | 12 passed | `logs/easysteer-kernel.log` |
| Tiny Qwen2 model, baseline/ADD/CAP/GATE/mixed, cache enabled | passed | `reports/easysteer/tiny-smoke.json` |
| Optimized selector and actual request replay | passed; token-ID D2H calls/bytes=0 | `reports/easysteer/tiny-optimized.json` |
| Qwen3.6-35B-A3B, ordinary TP=4, final norm + decoder | passed on all 4 workers | `reports/easysteer/qwen35-smoke.json` |
| Qwen3.6-27B, ordinary TP=2, final norm + decoder | passed on both workers | `reports/easysteer/qwen27-smoke.json` |
| Original native HTTP completions API | baseline/CAP/GATE succeed; CAP scale=0.5 returns 400 | `reports/easysteer/http-smoke.json` |
| Unified managed backend + original chat API, 27B TP=2 | passed including `graph_requirement=required` | `reports/easysteer/managed-chat/verification.json` |
| Ruff on modified runtime/backend paths | passed | final runtime check log |

The full suite was run before the last isolated TACT ADD resident-cache override;
the final 57-test regression set covers that change and preserves original
direction materialization behavior. Project tests did not rerun SWE-bench tasks;
the user's runner/container/harness code was not changed by this backend work.

Mechanism fixtures force four generated close tokens per request. Only the first
three are consumed by later forwards; two requests therefore produce exactly six
matches at each selected site. Prompt close occurrences do not add matches.
GATE=0 adds matches but no modifications; subsequent baseline adds neither.
Normal and mixed requests use the same engine, with true FULL and PIECEWISE
replay observed. These checks use labelled synthetic axes and controlled generation,
not fitted research axes or SWE resolve-rate evidence.

## End-to-end performance

Measurements use the same fork, 35B checkpoint, BF16, TP=4, GPU allocation,
generation length, and batch geometry. Prefix caching and GPU diagnostics are
disabled. Each condition has one separate admission/warmup sample and three
steady samples of 128 generated tokens per request. The native and steering
engines are measured separately; another user's workload on GPUs 2/3 was left
running. This is an initial engineering measurement, not a statistically strong
production p95 estimate.

Artifacts:

- `performance-native.json`: original fork engine without steering enabled.
- `performance-steering.json`: initial native host token selector.
- `performance-optimized.json`: singleton TACT GPU token matching.

Ordinary-token workload, throughput reduction against the same declared engine's
`steering=False`, after optimization:

| Batch | Prompt tokens | ADD | CAP | GATE |
|---:|---:|---:|---:|---:|
| 1 | 16 | 4.28% | 4.04% | 4.51% |
| 1 | 512 | 4.64% | 4.82% | 4.91% |
| 4 | 16 | 3.71% | 3.75% | 3.72% |
| 4 | 512 | 3.30% | 3.51% | 3.39% |
| 8 | 16 | 2.29% | 3.41% | 2.59% |
| 8 | 512 | 3.17% | 3.23% | 3.38% |

All these initial sparse/ordinary cases meet the plan's 5% throughput target.
The initial host-selector implementation had approximately 5.0–6.3% single-request
overhead. GPU matching removed the token-ID D2H dependency for the common TACT
clause. The disabled declared engine differed from native by approximately
-2.8% to +0.1% in the initial ordinary comparison, within normal measurement noise;
negative overhead is not evidence of a real speedup.

Every-token close stress is reported separately: approximately 3.2–5.6% overhead,
with the highest values at batch 1/long prompt. The 5% sparse-load target is not
claimed for this stress workload. Formal p50/p95 and lower-level GPU kernel-launch
profiles across longer runs remain a performance-study expansion; the present
deliverable includes reproducible benchmark code and raw timing samples.

## Remaining boundaries

- Only audited Qwen3_5 dense/MoE are accepted by the project launcher; 9B is not
  tested because no local checkpoint was available.
- PP/DP, sequence/context/feature-sharded hidden layouts, multimodal inputs, LoRA,
  speculative decoding, async scheduling, and hot axis swaps are not certified.
- Mixed configs, short/long prompt, slots, native cache on/off, model geometry,
  baseline recovery, and final norm are exercised. Capacity overflow and exhaustive
  cache/preemption combinations use existing native policy but are not claimed as
  new full-scale stress certification.
- Request-end evidence contains aggregate per-site counters and a recent sample,
  not a complete per-request/per-token activation dump. GPU diagnostics can
  perturb timings and should remain disabled in performance runs.
- The upstream multiprocessing shutdown sometimes emits semaphore/shared-memory
  warnings and uses its force-kill fallback; test-owned GPU processes are checked
  after completion. No unrelated GPU process is stopped.
- No fitted-axis SWE task experiment was run in this work. Backend correctness
  does not establish an improvement in resolve rate.
