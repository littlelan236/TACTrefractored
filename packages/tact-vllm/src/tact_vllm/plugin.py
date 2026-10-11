def register():
    from tact_vllm.compat import check_version

    check_version()
    from vllm import ModelRegistry

    for name in ("TactQwen3_5ForConditionalGeneration", "TactQwen3_5MoeForConditionalGeneration"):
        ModelRegistry.register_model(name, "tact_vllm.models.qwen3_5:" + name)
