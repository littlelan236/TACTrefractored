# Shared baseline task coordination

Set `execution.coordination_group` to the same explicit group ID in cooperating run configurations. Default is null: independent scientific experiments continue to run independently.

The current Qwen3.5-9B cohort uses `qwen35-9b-baseline258-20261012`. Forward uses GPUs 4–7, workers/max_num_seqs 12 and prefix caching; reverse uses GPUs 0–3, workers/max_num_seqs 8 and prefix caching. Both retain gpu_memory_utilization=0.5 and unchanged generation/agent budgets.

The coordinator lives in `src/tact/rollout/coordination.py`. Linux flock claims under `state/task-groups/<group>/` protect each task for its full execution. Another process skips locked tasks and advances in its own queue order. If only peer-owned tasks remain, it waits for completion or lock release; this is runner synchronization, not external monitoring. Locks automatically release on process exit. The owner JSON is diagnostic metadata, not the lock authority.

Only terminal agent outcomes are reused: Submitted, LimitsExceeded, ContextWindowExceededError, RepeatedFormatError and TimeExceeded. This includes valid unsuccessful outcomes; they are not silently rerun until successful. Infrastructure failures remain eligible for fresh attempts under the existing retry policy.

Reuse validates the group model fingerprint, generation, agent/templates, steering, environment, timeouts, dtype, context and TP/parser identity. Per-task dataset content and locked image ID must match. Task order, GPU IDs, workers and prefix-cache settings intentionally do not distinguish these cooperating baseline executors. Different seeds/models/steering/images fail closed rather than sharing outcomes.

Selected completed attempt directories are copied physically, with trajectory SHA-256 validation, atomic result writes and a `reused_from` source run/attempt/hash field. Trajectories retain their original config provenance. Reports expose reused_from; do not sum two cohort reports as independent work or claim reused responses were generated under the destination run configuration.

Completed historical outcomes are retained. Existing interrupted attempts are kept and new attempts start fresh; message-only continuation in a new container is not attempted. Repeated planned restarts still consume the configured attempt budget.

With concurrency, order means admission priority: each runner scans its requested order while skipping tasks claimed/completed by its peer. It does not imply serial finish order. These two run directories represent one collaborative dataset, not independent APC comparison samples.

Regression coverage: competing process lock, release on process death, reuse of terminal unsuccessful outcomes, physical-copy provenance, incompatible seed/image rejection and advance past a busy task then import its completion. Full project checks: 52 passed, 3 skipped before launch.
