# Native vLLM registry backend

This package constructs Qwen3.5 dense/MoE models with a TACT backbone and an
explicit V1 worker/runner adapter. It does not modify installed vLLM files and
does not implement an HTTP server. Constructor/device initialization portions
are adapted from the audited vLLM 0.30.0 source; the remaining model, loading,
attention, MoE, cache, parser and server behavior is inherited.

With `serving.enable_prefix_caching=true`, the Registry launcher also selects
the package's `TactScheduler`. This engine-local composition wraps the audited
vLLM cache coordinator; no installed vLLM file is edited. Its `clean_history_v1`
policy keeps the running request's full state but blocks shared prefix
publication after the first generated `</think>` when decoder-output steering
is active. Prompt/history close tokens are ignored, and the next request
recomputes the uncached tail without steering before it is shared. The policy
supports the audited Qwen3.5 hybrid attention/Mamba layout with
`mamba_cache_mode=align`; transfer, streaming, preemption/resumption and
speculative modes are rejected.

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
