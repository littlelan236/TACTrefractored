# Actual fitted-axis EasySteer diagnostic — 2026-10-11

Validated on project-local Qwen3.6-35B-A3B, TP=2, GPUs 4/5, native EasySteer
HTTP server with actual FULL replay, original fitted TACT axes and CAP k=2.
Inputs were the exact baseline-derived pydicom-1048 and pyvista-4648 first-step
token prefixes. Two forced-close examples isolate the boundary; another request
naturally generated close and action tokens. No SWE containers were launched.

## Results

- Nine HTTP requests completed; both workers attested real graph replay.
- Two single-consumed-close CAP requests each produced ten site records per
  worker. NONE/history-close controls and final NONE recovery produced zero
  token/site records.
- Four forced generated closes produced three consumed close states, hence
  three records per selected site per worker. This confirms the configured
  all-generated-close semantics without touching prompt closes.
- Natural generation produced one generated close and ten records per worker.
- Total: 120 token/site records, 12 with actual hidden-branch modifications.
  Every recorded token ID was 248069 and in the generated phase; no records
  were dropped. CAP projection error against ideal clamp was at most 0.000573.

Example from the pyvista-4648 prefix:

```text
request: cmpl-84917493224de33f-0-907d1449
mode: FULL
token_id: 248069 (</think>)
absolute_position: 1834; generation_position: 0
site: decoder_output block 35
before: [2.291112185, -1.957344532]
threshold: [2.767550707, 1.435091257]
after: [2.290924072, -1.435528398]
ideal CAP after: [2.291112185, -1.435091257]
actual hidden writeback norm: 0.522165179
ideal correction norm: 0.522253275
```

The OA projection was outside its band and was pulled to the negative boundary;
OT was already inside and stayed nearly unchanged. The logged actual norm is
computed after BF16 hidden-branch writeback, not merely from the ideal delta.
Small discrepancies are the backend's finite-precision writeback effect.

GPU regression tests verify detailed logging preserves the exact original
output, uses mutable fixed-address buffers under graph replay, and clears idle
rows. CPU regressions verify correct request identity when slots are shared,
phase/slot mismatch rejection, disabled-path behavior and bounded-log dropping.
Configuration/consumer regressions cover default-off and explicit enabling.

Raw evidence and machine-readable `diagnosis.json` are in the server's
`reports/easysteer-detailed-diagnosis-20261011/`. Detailed logs contain scalars
and identity metadata, not full 2048-dimensional vectors. This validates
steering execution and CAP mathematics for these requests; it does not establish
SWE resolve-rate improvement or certify all concurrency/preemption combinations.
Diagnostic overhead is excluded from performance claims.
