# EasySteer TACT backend

The project uses the pinned EasySteer 0.29.0 fork with its original Python and
OpenAI-compatible APIs. No custom HTTP server, registry-model plugin, capture
model wrapper, or alternate rollout runner is required.

The source trees are physical copies under `vendor/`. The dedicated runtime is
`.venvs/easysteer`; installing/upgrading a stock vLLM wheel replaces the Python
overlay, so run `scripts/install_easysteer_overlay.py` and
`scripts/verify_easysteer_runtime.py` afterwards. The installer records source
hashes and upstream revisions in `vendor/easysteer-runtime.json`; managed serving
checks the modified files against that manifest.

## Configuration

Use the existing `configs/experiment.yaml` and resolver. For example:

```bash
.venvs/runner/bin/python -m tact \
  --set serving.backend=easysteer \
  --set steering.trigger=all_generated_think_close \
  --set steering.strategy=cap \
  --set steering.axes=artifacts/axes/YOUR_FITTED_BUNDLE \
  config show
```

The same overrides work with the normal rollout/pipeline commands. `none`
explicitly sends `steering: false`, including when the engine has a default.
For `add`, `cap`, and `gated`, the backend sends one native `VectorSpec.data`
payload in the OpenAI `steering` field. The axis manifest must match the actual
checkpoint and tensors; synthetic mechanism fixtures are not fitted research
axes and must not be used to claim TACT benchmark improvements.

`serving.graph_requirement=required` checks actual replay records from every TP
worker after real requests. Readiness and `steer_graph_mode=in_graph` alone do not
pass that check. `serving.easysteer_diagnostics=true` additionally captures
aggregate GPU match/change counters, recent projection values, and correction
norms. It is opt-in because request-end copies and extra atomics change measured
performance. Production has only native replay counters and request-end evidence.

## Semantic differences from registry

The implemented backend uses the following semantics:

- Every newly generated `</think>` that is subsequently consumed by forward is
  selected. Prompt/history occurrences are excluded. This is deliberately
  `all_generated_think_close`, not registry's first-occurrence behavior.
- Decoder projection reads `(hidden + residual)` in model dtype, then computes
  centered projections and correction in FP32. Correction is written through
  the hidden branch; carried residual remains unchanged.
- `final_norm` reads only normalized output. Auxiliary norm outputs are retained
  and never treated as another residual.
- GATE permits finite lambda >= 0, including values above 1. CAP requires scale
  exactly 1. ADD uses the native direct/additive math and scale once.
- Parameters, centers, thresholds, and TACT tables remain FP32. Unselected and
  zero-correction rows remain unchanged.

The TACT component inventory uses site IDs `0..N-1` for decoder output and `N`
for final norm. The project's `hf-hidden-v1` bundle has N rows: its last row is
final norm, so that row maps to site N, not decoder N-1. The adapter does not
fit axes, move directions between layers, normalize directions, or choose new
layers.

## Implementation

- `src/tact/backends/easysteer.py`: validated launch/request/evidence adapter.
- `src/tact/backends/axis_bundle.py`: shared model-bound axis validation.
- `easysteer.tact`: explicit NPZ-to-payload conversion helper.
- `TactBandPayload`: native wire/hash/layers/shape/materialization contract.
- `tact_add`, `tact_cap`, `tact_gate`: native algorithms and capability entries.
- `tact_states`: separate discovery/controller and final norm output adapter.
- `tact_band`: persistent FP32 tables and fused additive/band Triton launch.

One config slot carries one algorithm. Changing method/threshold/lambda creates
a new effective configuration; native fingerprints/refcounts manage sharing and
reuse. Slot installation clears both family tables before writing; release and
rollback clear their contents without replacing GPU storage.

The native selector contract is reused. TACT's common single generated-token
clause now stages host-known phase/slot geometry in bulk and matches the actual
token IDs on the GPU, avoiding token-ID D2H. Complex clauses and non-TACT algorithms
retain the native general resolver, which performs one cached token-ID D2H read
per batch step, not one read per layer. Opt-in
`TACT_EASYSTEER_PROFILE=1` records counts/bytes/host time; it does not change the
public `ApplySpec` union/exclusion semantics. No claim is made that token
selection itself is inside the captured graph.

## Reproducible validation

```bash
.venvs/easysteer/bin/python -m pytest -q vendor/EasySteer/tests/cpu
CUDA_VISIBLE_DEVICES=0 .venvs/easysteer/bin/python -m pytest -q \
  vendor/EasySteer/tests/cpu/test_tact_gpu.py
.venvs/runner/bin/python -m pytest -q tests
```

CPU-only runs skip CUDA tests. Some upstream training tests need the optional
`accelerate` extra; production serving does not depend on it. The upstream test
fixtures were updated for the added graph family and the wheel-overlay source
layout. Run from project root with `PYTHONPATH=$PWD/vendor/EasySteer` when tests
import the upstream experiment helpers.

Historical model/HTTP smoke and serving-performance drivers were removed
from `scripts/` during the 2026-10-14 cleanup. The dated results remain in
[EASYSTEER-VALIDATION.md](EASYSTEER-VALIDATION.md) and the corresponding reports;
they are historical evidence, not current executable script instructions.
The package backend and pinned vendor regression tests remain available through
the commands above. Use [the pipeline guide](BASELINE-WORKFLOW.md) for actual
baseline and fitted-axis CAP/GATE experiments.

A generated think-close applies only if consumed by a subsequent forward;
the final sampled token is not processed again after termination.

## Support boundaries

The project launcher validates the audited Qwen3_5 dense and MoE architectures,
text-only input, single-token think boundary, V2 runner, ordinary TP, PP=DP=1,
and disabled sequence parallelism. Other model families can use the fork extension
only after component/output-layout validation; the project does not silently
accept unknown architectures. Async scheduling, speculative decoding, LoRA,
multimodal input, PP/DP, hot axis replacement, and unverified sharded layouts are
not enabled by the adapter. A multi-token think boundary is rejected.

Performance results, validation coverage, limitations, and artifact locations
are recorded separately in `EASYSTEER-VALIDATION.md`.

## Optional token diagnostics

See [detailed logging](EASYSTEER-DIAGNOSTICS.md) and the [actual fitted-axis diagnosis](EASYSTEER-DIAGNOSIS-20261011.md). Detailed logging is off by default.

## Optional tensor consumer verification

See [consumer check](STEERING-CONSUMER-CHECK.md) and [actual GPU results](STEERING-CONSUMER-RESULT-20261011.md). Disabled by default.
