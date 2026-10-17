# Registry LDA seed42 queue

Wait for both EasySteer LDA seed42 rollout, harness and report to terminate.
Require successful rollout/report and summary outcomes covering all selected tasks.
Terminal harness grading failures are recorded but do not prevent continuation with full outcome coverage.
Registry: first_generated_think_close, clean_history_v1 prefix caching.
Other experiment parameters match current EasySteer LDA seed42; see arm-validation.json.
CAP GPUs0-3 port8130; GATE GPUs4-7 port8131; TP4; rollout workers12; harness workers12.
Rollouts run concurrently after resource checks; harness jobs serialized.
Run IDs: lda-registry-seed42-eval-20261015-{cap,gated}.
State: lda-queue-status.json; log: lda-queue.log; per-arm logs: lda-{arm}-{stage}.log.
Lock and existing-state checks prevent duplicate launches; do not restart existing state.
No installed vLLM source changes.
