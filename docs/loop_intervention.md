# Action-cycle intervention

Enable in an experiment configuration:

```yaml
agent:
  loop_intervention:
    enabled: true
    max_period: 5
    min_repeats: 2
    prompt: "Your current actions are stuck in an ineffective loop. Please explore alternative approaches."
```

Defaults are explicitly recorded in configs/experiment.yaml. The feature is off
by default and works with native, registry, and EasySteer rollout backends.
Existing resolved run configs without this field remain off. Use a new run ID
to change the configuration; resumed runs retain immutable configuration.

Each successfully executed assistant tool-action list is one step. Compare
parsed tool names and arguments, ignoring call IDs and reasoning. JSON key order
is normalized; shell command whitespace is preserved because it can change
quoted strings and heredoc semantics. Compare trailing cycles of period 1..5;
two repetitions trigger by default. Shortest matching period wins. No minimum
trajectory length and no requirement to reach a terminal limit.

After observations are recorded, append a user message with the configured
literal prompt. It is visible on the next query and preserved in the trajectory.
Reset the detection window on intervention or unsuccessful steps; subsequent
cycles can trigger additional reminders. No separate API call is added.
Submission/terminal exceptions never receive a reminder. State is per attempt.
Identical actions can be legitimate; this mechanism does not prove ineffectiveness.

Audit: user-message extra.tact_loop_intervention, info.loop_intervention, and
attempt-local loop-interventions.jsonl. Events contain zero-based assistant step
indices, period, repetitions, action signatures, prompt and message index.
Enrichment reconstruction includes the injected user message automatically.
Token matching must use the serving tokenizer as for other runs.

Offline tests use fake models/environments; replay checks use existing saved
trajectories. No live generation is required to validate detection and injection.
Whether the prompt changes model behavior requires a new rollout experiment.

Reminders are user-role messages immediately after tool observations. Qwen3.5
accepts this order directly: no system merging or wire-message transformation.
Original system prompt and historical messages stay unchanged. This reminder has
user-level priority, not system-level priority. Its efficacy requires live rollout.
