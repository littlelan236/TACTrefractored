# Registry TACT backend

Implemented for this project's **vLLM 0.30.0 V1** runtime. It uses the native
OpenAI-compatible server and the same rollout/environment/harness pipeline as
ordinary baseline runs. No installed vLLM file is edited.

## Run

Install the project-local model package into the serving environment:

```bash
uv pip install --python .venvs/serving/bin/python --no-deps -e packages/tact-vllm
```

Use the existing single experiment configuration and explicit overrides:

```bash
.venvs/runner/bin/python -m tact \
  --set serving.backend=registry_vllm \
  --set steering.axes=artifacts/axes/YOUR_FITTED_BUNDLE \
  --set steering.strategy=cap \
  --set steering.k=2 \
  config check

.venvs/runner/bin/python -m tact \
  --set serving.backend=registry_vllm \
  --set steering.axes=artifacts/axes/YOUR_FITTED_BUNDLE \
  --set steering.strategy=cap \
  run --run-id registry-cap
```

For matched baseline use the same backend and bundle with
`steering.strategy=none`. Strategies are `none`, `cap`, `gated`, `add`;
parameters use the existing `steering.k`, `steering.lambda`, and
`steering.additive_scale`. The bundle must come from the selected checkpoint's
research pipeline. A different checkpoint needs its own bundle.

`serving.graph_requirement=auto` reports actual dispatch behavior;
`required` requires observed FULL replay after real requests, and fails the run
if none occurred. Short prefills can remain eager/piecewise. Readiness alone
does not prove replay. `eager` is available for mechanism comparisons.

## Implementation

- `packages/tact-vllm/src/tact_vllm/models/qwen3_5.py`: explicit dense/MoE
  conditional-generation → custom CausalLM → custom backbone construction.
  Original parameter names, native decoder layers, weight loaders and MoE/cache
  protocols remain in use.
- `runtime/ops.py`: FP32 dual-axis projection and CAP/GATED/ADD math; intermediate
  states read `hidden.float() + residual.float()` and update residual; final norm
  updates normalized hidden. Inactive/zero-delta rows preserve original values.
- `runtime/buffers.py`: nonpersistent stable GPU axis, request, mask and evidence
  buffers allocated before compilation/capture.
- `adapters/worker.py`: version-audited device initialization that explicitly
  constructs the custom GPU model runner once.
- `adapters/model_runner.py`: stages geometry after native batch reordering,
  verifies real token IDs, handles slot epochs and warmup, and records evidence.
- `src/tact/backends/registry_vllm.py`: native server launch and request field
  translation. Stock LitellmModel sends `vllm_xargs`; no capture model is added.

Each inference request freezes its plan. The runner locks the first accepted
generated `</think>` absolute position, excludes prompt/history/padding, and
supplies one shared decision mask to all selected sites. A generated close that
ends the request without another forward has no subsequent action state to
steer. Evidence distinguishes token generation from a processed decision state.

Site semantics are the project's `hf-hidden-v1`: decoder outputs 0..L-2 plus
final norm, not the last decoder's unnormalized output. Layer counts, hidden
sizes and token IDs are derived from the checkpoint/bundle/tokenizer.

## Supported scope

Audited Qwen3_5 dense/MoE, text-only, single-token think-close, synchronous
scheduling, ordinary tensor parallelism. Continuous batching uses independent
request slots. Prefix caching, PP/DP/EP/EPLB/DBO, context/sequence parallelism,
quantization, speculative decoding and LoRA are explicitly rejected. Resumed or
preempted requests are rejected until recomputation semantics are validated.
9B has not been tested because no corresponding checkpoint is present.

The package checks the exact audited vLLM source hashes. Upgrading/reinstalling
vLLM requires a new source review and adapter verification. The copied device
initialization/constructors are local package code, not runtime monkeypatches.

## Evidence and tests

`runs/<run>/serving/service.json` identifies the service and its evidence session.
Worker JSONL files under that session record model types, stable buffer addresses,
actual graph dispatch, FULL replay, decision positions, per-site counts, and
before/threshold/after projections plus delta norms. A decision snapshot is
persisted immediately so final requests are observable even without another
scheduler tick. Session directories prevent resume/relaunch mixing old evidence.

This initial implementation writes dispatch evidence and synchronizes at the
one decision boundary. It prioritizes correctness evidence; throughput has not
been optimized or benchmarked.

Run project tests with the runner environment. Package tests use the serving
environment (pytest may be supplied from the runner environment):

```bash
.venvs/runner/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=4 \
  PYTHONPATH="$PWD/.venvs/runner/lib/python3.12/site-packages" \
  .venvs/serving/bin/python -m pytest -q packages/tact-vllm/tests
```

`tests/make_synthetic_bundle.py` and `tests/server_smoke.py` provide explicit
synthetic mechanism fixtures. Synthetic bundles are not fitted drift axes and
cannot establish research effectiveness. The smoke forces repeated generated
close tokens, checks the first-close position, both decoder/final-norm sites,
NONE/CAP/GATED/ADD parameter changes, projection math, concurrent mixed requests
and actual FULL replay.

Validation results and paths are recorded in `REGISTRY-VALIDATION.md`.

## Optional tensor consumer verification

See [consumer check](STEERING-CONSUMER-CHECK.md) and [actual GPU results](STEERING-CONSUMER-RESULT-20261011.md). Disabled by default.
