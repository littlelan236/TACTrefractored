# Implementation status

Completed in the first refactor slice:

- Independent project tree, physical model/wheel assets, hashes and clean venvs.
- Strict single experiment configuration and immutable resolved config.
- Native vLLM serving process manager with explicit model identity and no steering fields.
- Shared Docker `ContainerFactory`, bootstrap policy, wheelhouse mount, package probe and timeout checks.
- Common mini-swe-agent rollout runner with fresh attempts and explicit infrastructure failures.
- Scoped SWE-bench 4.1.0 harness adapter and shared evaluation bootstrap.
- Trajectory reader, annotation export/import, decision-point reconstruction, dynamic latent extraction, model-bound axis fitting and reports.
- Unit tests for configuration, trajectory usage, retries, environment policy, site semantics and axis fitting.

Not included:

- Registry implementation and current validation are described in
  [REGISTRY-BACKEND.md](REGISTRY-BACKEND.md) and [REGISTRY-VALIDATION.md](REGISTRY-VALIDATION.md).
- No EasySteer integration.
- CUDA Graph registry mechanism tests use synthetic axes; they do not establish
  TACT research effectiveness.
- A one-task registry SWE smoke completed rollout and harness wiring; it was
  deliberately limited to two agent steps and ended `LimitsExceeded`/`MissingPatch`.

The baseline backend still requires a model smoke on an available GPU set before a full SWE rollout. The native serving lock is separate from the runner and analysis locks.
