# Native vLLM registry backend

This package constructs Qwen3.5 dense/MoE models with a TACT backbone and an
explicit V1 worker/runner adapter. It does not modify installed vLLM files and
does not implement an HTTP server. Constructor/device initialization portions
are adapted from the audited vLLM 0.30.0 source; the remaining model, loading,
attention, MoE, cache, parser and server behavior is inherited.

Install into the project's serving environment:

```bash
uv pip install --python .venvs/serving/bin/python --no-deps -e packages/tact-vllm
```

Select `serving.backend=registry_vllm` through the common TACT configuration.
Enabled steering requires `steering.axes` to point to a complete model-bound
AxisBundle directory. NONE can run without axes; for a matched NONE/steered
comparison, select the same bundle in both runs.

The backend checks exact upstream source hashes and currently requires native
vLLM 0.30.0 V1, synchronous scheduling, text-only input, ordinary TP, and no
prefix caching, PP/DP/EP/EPLB/DBO, quantization, LoRA, speculative decoding or
sequence/context parallelism. Preempted/resumed requests are rejected. These
limits are checked before using unsupported execution paths.

Every request has an immutable explicit `vllm_xargs` plan. The runner stages the
current flattened request geometry after native batch reordering, observes the
first accepted generated think-close position, and verifies the actual GPU token
ID. The model reads stable buffers during capture/replay and applies the same
mask to every selected site. Intermediate sites update the residual component;
final norm updates normalized hidden states.

Worker JSONL evidence records custom model construction, graph dispatch/replay,
first-close positions, site counts, and summed before/threshold/after projection
and delta norm values. Evidence adds overhead; it is enabled in this initial
validation implementation. FULL replay requirement is checked after actual
requests, never from server readiness alone.

Synthetic axes under `registry-synthetic-mechanism` are test fixtures and must
not be used to claim TACT effectiveness. GPU tests and the real native-server
mechanism script are in `tests/`.
