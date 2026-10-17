from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path


def check_version():
    expected = json.loads(Path(__file__).with_name("upstream.json").read_text())
    if importlib.metadata.version("vllm") != expected["version"]:
        raise RuntimeError("TACT registry requires its audited vLLM 0.30.0 runtime")
    root = Path(importlib.util.find_spec("vllm").origin).parent
    for relative, digest in expected["files"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"TACT audited vLLM source mismatch: {relative}")


def check_runtime(cfg):
    check_version()
    parallel = cfg.parallel_config
    errors = []
    if cfg.use_v2_model_runner:
        errors.append("V2 runner")
    if cfg.scheduler_config.async_scheduling:
        errors.append("async scheduling")
    if parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1:
        errors.append("PP/DP")
    if parallel.enable_dbo or parallel.enable_expert_parallel or parallel.enable_eplb:
        errors.append("DBO/EP/EPLB")
    if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
        errors.append("context parallelism")
    if cfg.speculative_config or cfg.lora_config or cfg.quant_config:
        errors.append("speculative/LoRA/quantization")
    if cfg.cache_config.enable_prefix_caching:
        from tact_vllm.runtime.cache_policy import CLEAN_HISTORY_SCHEDULER

        if cfg.scheduler_config.scheduler_cls != CLEAN_HISTORY_SCHEDULER:
            errors.append("prefix caching without the TACT clean-history scheduler")
        if cfg.cache_config.mamba_cache_mode != "align":
            errors.append("prefix caching without mamba_cache_mode=align")
        if cfg.kv_transfer_config is not None or cfg.ec_transfer_config is not None:
            errors.append("KV/EC transfer with clean-history caching")
    if cfg.compilation_config.pass_config.enable_sp:
        errors.append("sequence parallelism")
    if not cfg.model_config.multimodal_config.language_model_only:
        errors.append("multimodal inputs")
    if errors:
        raise ValueError("TACT registry unsupported execution: " + ", ".join(errors))
