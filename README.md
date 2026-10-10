# tactRefractor

Independent SWE-bench-only TACT reproduction project. The current implementation contains the unified configuration, task container factory, rollout runner, official harness adapter, trajectory-based research pipeline, and native vLLM baseline backend.

Steering backends are intentionally unavailable in this stage. Configurations requesting steering, registry vLLM, or EasySteer fail before GPU or Docker work begins.

## Quick checks

```bash
.venvs/runner/bin/tact config check
.venvs/runner/bin/tact assets-check --deep
.venvs/runner/bin/tact environment-check --run-id env-check
.venvs/runner/bin/tact run --run-id baseline-smoke
```

All task Docker containers are created through `tact.environments.ContainerFactory`. Rollout and harness use the same image identity, dependency policy, wheelhouse, bootstrap script, locale, and timeout resolver. Each task attempt and environment probe is retained under `runs/<run>/tasks/<id>/attempts/<attempt>/`.

The default `.traj.json` is the authoritative rollout artifact. A separate capture model is not used. Stored responses, reasoning, tool calls, usage and provenance are consumed by `TrajectoryReader` and decision-point extraction.

Model-specific hidden-state axes must be generated for the exact checkpoint, tokenizer, template, layer count and hidden size. An axis bundle cannot be used with a different model fingerprint.
