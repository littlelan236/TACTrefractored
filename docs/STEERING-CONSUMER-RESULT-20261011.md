# Observed steering tensor handoff — 2026-10-11

Actual GPU validation on Qwen3.6-35B-A3B, TP=2, GPUs 4/5. Both native Registry
vLLM and EasySteer used the same fitted axes and baseline-derived pydicom-1048 /
pyvista-4648 token prefixes. Each backend served twelve short requests covering
NONE, CAP k=2, forced repeated close, CAP k=0, natural close/action, mixed
concurrent CAP/NONE and recovery. Actual FULL graph replay was observed.

| Backend | Full-vector comparisons | Next layer input norm | Logits processor input | Maximum hidden/residual error |
|---|---:|---:|---:|---:|
| Registry | 120 | 108 | 12 | 0 / 0 |
| EasySteer | 160 | 144 | 16 | 0 / 0 |

Every produced selected row was observed once at its actual consumer. Every
2048-component hidden vector and decoder residual vector matched exactly.
None of the checked consumers had mismatched components. NONE, history-close
controls and recovery had no selected producer/consumer records. Shared-slot
concurrent requests remained attributed correctly.

The 40 extra EasySteer records are expected: its all-generated-close policy
consumed three states in the four-close test, while Registry consumed only the
first. Neither backend steered old prompt closes.

CAP k=0 explicitly changed final norm, preventing an identity-only check:

| Backend | Request | Token / position | Actual final-norm modification norm | Logits-input mismatches |
|---|---|---|---:|---:|
| Registry | cmpl-9491e7fd8545eee1-0-b29edec4 | 248069 / 1602 | 5.979913 | 0 |
| EasySteer | cmpl-878085cf1cc8fa53-0-85ea1cab | 248069 / 1602 | 5.779500 | 0 |

These are checks at the actual `LogitsProcessor.forward` entrance after the
runner selected sampling rows, and at the next decoder's `input_layernorm`
entrance before in-place add/norm. They establish the observed tensor delivery,
not that later normalized projections remain unchanged or that the resulting
patch is correct. Full vectors were compared on GPU, but only comparison metrics
were persisted; this is not a raw activation archive.

Unit tests deliberately changed one consumer vector component and verified
nonzero mismatched-element counts. CUDA capture/replay tests changed input
contents after capture and detected a mismatch on replay. Logits index routing
and padded warmup lengths were covered. Registry CPU runtime tests and unified
config tests also passed. One initial EasySteer warmup exposed a diagnostic
padding bug; it was fixed, regression-tested and both backends were rerun.

An initial Registry logger duplicated stale records on scheduler-only ticks;
the logger now drains only when model-forward sequence advances. Results above
come exclusively from the clean second validation run.

Evidence on server:
`reports/steering-consumer-check-20261011-v2/consumer-diagnosis.json`, plus each
backend's serving evidence directory. Diagnostic services were shut down after
completion. Other GPU workloads were left untouched.

Scope remains this checkpoint, supported TP/synchronous text serving and tested
requests. No universal guarantee is claimed for arbitrary architectures,
speculative decode, PP/DP, LoRA, feature sharding or untested preemption paths.
