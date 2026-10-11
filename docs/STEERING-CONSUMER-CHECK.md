# Optional producer-to-consumer tensor verification

Configure Registry or EasySteer new runs with:

```yaml
serving:
  steering_consumer_check: true
  steering_consumer_check_rows: 32
  steering_consumer_log_limit: 20000
```

Disabled by default. GPU diagnostic buffers and synchronization add overhead;
use this for correctness checks, not throughput measurements.

Selected token rows snapshot all hidden components and, for decoder sites,
all residual components separately. A real pre-hook on the next decoder's
`input_layernorm` compares actual inputs before norm/add can mutate them.
Final-normalized outputs are compared at `LogitsProcessor.forward` using actual
sampling row indices. These are full-vector comparisons, not checksums.

Each record includes request identity, token position, site, execution mode,
producer/consumer counts, mismatched-component counts, maximum absolute
differences and actual intervention norm. Zero mismatches and one consumer
per produced row establish the observed handoff for that execution.

Snapshot capacity covers the first configured N flat batch rows. A selected
token outside capacity fails explicitly. Normal prompt rows are not selected
by generation-close steering. Registry writes `type=consumer_check` in worker
evidence; EasySteer writes `consumer-checks-<worker-pid>.jsonl`. Log drops,
capacity overflow and missing/mismatched consumers mean incomplete/failed
verification rather than success.

Audit scope: local Qwen3.5 dense/MoE, ordinary TP, PP=DP=1, synchronous text
serving, no speculative decode/LoRA/feature sharding. Existing first/all-close
semantics and steering mathematics are unchanged. This does not certify
unobserved runtime modes, architectures or task success.
