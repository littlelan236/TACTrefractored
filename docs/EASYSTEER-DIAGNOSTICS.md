# Optional detailed EasySteer diagnostics

Enable this in the unified config, or with `--set`:

```yaml
serving:
  easysteer_detail_log: true
  easysteer_detail_log_limit: 20000
```

The default is false. `easysteer_diagnostics` controls the existing aggregate
diagnostics independently; detailed logs do not require that flag.

Managed server evidence contains `token-details-<worker-pid>.jsonl`. Each record
identifies request ID, engine step, configuration slot, token ID, absolute and
generated position, semantic site/module, algorithm/payload identity, graph
mode, before/threshold/after projections, ideal correction norm and actual BF16
hidden-branch modification norm. It also reports whether the branch changed.

Values come from the actual captured steering kernel. After-projection uses
the actual modified output plus residual under this backend's model-dtype
addition convention. The ideal clamp projection is separate. Full hidden
vectors are not stored.

Persistent GPU buffers are allocated before graph capture and drained after
real forwards. Dummy runs, padding and unselected rows produce no token records.
Shared config slots are mapped to the real batch request. Validity is reset
before each forward, preventing stale data.

The per-worker output limit counts dropped records; final verification exposes
record/drop counts. Once reached, coverage is incomplete. Detailed logging adds
GPU/CPU synchronization and projection/norm work, so disable it for performance
measurements. Disabled kernels compile out the detailed branch and allocate no
detail buffers. First/all-close semantics and steering numerics are unchanged.

Example:

```bash
.venvs/runner/bin/tact \
  --set serving.backend=easysteer \
  --set steering.trigger=all_generated_think_close \
  --set steering.strategy=cap \
  --set steering.axes=artifacts/axes/YOUR_BUNDLE \
  --set serving.easysteer_detail_log=true \
  run --run-id easy-diagnostic
```
