# Qwen27B 参数说明

本工程的 `assets/models/Qwen3.6-27B` 实际 architecture 是
`Qwen3_5ForConditionalGeneration`，文本配置为 64 层、hidden size 5120、
262,144 最大位置长度，权重 dtype 为 bfloat16。

Qwen 官方 Qwen3 推理示例对 thinking mode 使用：

```text
temperature = 0.6
top_p       = 0.95
top_k       = 20
min_p       = 0
```

官方 non-thinking 示例通常使用 `temperature=0.7, top_p=0.8, top_k=20,
min_p=0`。本 TACT SWE-bench 工程保留 thinking mode，因此采用第一组参数。
模型目录自带的 `generation_config.json` 也确认了 `top_p=0.95、top_k=20`；
工程显式设置 temperature/min_p，是为了让服务端行为不依赖 vLLM 对模型
generation config 的默认解释。

使用 Qwen27B profile：

```bash
.venvs/runner/bin/tact \
  --config data/model-profiles/qwen27-official.yaml \
  config check

.venvs/runner/bin/tact \
  --config data/model-profiles/qwen27-official.yaml \
  run --run-id qwen27-baseline
```

当前 profile 的 `max_model_len=262144` 来自 checkpoint 的模型配置；
`max_tokens=81920` 是编码 agent 的单次响应预算，不是 Qwen 官方推荐的
`max_new_tokens` 值。它用于保持 SWE-bench 长轨迹能力，若做单轮模型比较，
可以在新 run 上显式覆盖 `generation.max_tokens`。

`tensor_parallel_size=8`、`max_num_seqs=8`、`max_num_batched_tokens=8192`
和 `gpu_memory_utilization=0.9` 是当前服务器的工程拓扑参数，不是模型卡
规定的值。TP 应根据实际显存验证；切换 TP、GPU 或并发时需单独记录性能和
CUDA Graph 结果，不能把变化归因于采样参数。

参考：

- [Qwen3 官方模型系列说明](https://qwenlm.github.io/blog/qwen3/)
- [Qwen 官方 Hugging Face 组织](https://huggingface.co/Qwen)
- [vLLM Qwen 模型文档](https://docs.vllm.ai/en/latest/models/supported_models.html)

网络受限时，以模型目录内的 `config.json`、`generation_config.json` 和
tokenizer/chat template 为本地可复现依据；每个 run 会把解析后的参数写入
`resolved-config.json`。
