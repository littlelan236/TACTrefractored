# Registry validation record

Validation date: 2026-10-10 (Asia/Shanghai). Runtime: project-local vLLM
0.30.0, PyTorch 2.13.0, Qwen3.5 checkpoints, V1 model runner.

## Completed

- Project tests: **18 passed**.
- Registry package runtime tests: **5 passed**.
- 35B-A3B synthetic-axis eager server smoke: passed.
- 35B-A3B synthetic-axis CUDA Graph server smoke: passed; **96** FULL replay
  dispatches observed.
- 27B dense synthetic-axis CUDA Graph server smoke: passed; **48** FULL replay
  dispatches observed.
- 35B-A3B TP=4 model loading: all 4 workers loaded the custom registry model,
  with custom TACT backbone evidence on every rank.
- 27B TP=2 model loading and graph smoke: passed.
- End-to-end one-task SWE runner smoke: rollout completed, registry service
  evidence recorded custom backbone on 4 workers and **1,116** FULL replay
  dispatches. The task ended `LimitsExceeded` under the deliberate two-step
  budget; harness reported `MissingPatch`. This validates pipeline wiring and
  failure accounting, not task quality.

## Mechanism checks

The server smoke sends NONE, CAP (`k=0`, `k=2`), GATED and ADD requests, then
repeats mixed requests concurrently. It verifies:

- plan fields arrive through native `vllm_xargs`;
- all selected decoder/final-norm sites share the decision mask;
- prompt/history close tokens do not trigger;
- the first generated close has one logical site execution;
- NONE leaves activation unchanged;
- projection values follow the FP32 reference formula;
- post-capture parameter changes affect replay without changing buffer addresses;
- padded/inactive rows remain unchanged;
- all TP workers see the same verified custom backbone and FULL replay.

The fixtures use orthogonal unit axes `[e0, e1]`, zero centers and unit sigma.
They are explicitly synthetic mechanism inputs and are not TACT fitted axes.

## Scope limitations

No claim is made yet for TACT effectiveness, resolve-rate improvement,
throughput, preemption/resumption, EasySteer, PP/DP/EP, prefix caching,
speculative decoding, quantization, LoRA, or 9B. A real experiment requires a
complete model-bound fitted AxisBundle and a matched NONE/steered cohort.
