"""Registry-only synchronous scheduler with clean cross-request prefix reuse."""
from __future__ import annotations

import json
import os
from pathlib import Path

from vllm.v1.core.sched.scheduler import Scheduler

from tact_vllm.adapters.kv_cache import CleanHistoryCoordinator
from tact_vllm.compat import check_runtime
from tact_vllm.runtime.buffers import load_manifest
from tact_vllm.runtime.cache_policy import CLEAN_HISTORY_POLICY, CleanHistoryPolicy


class TactScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        check_runtime(self.vllm_config)
        manifest = load_manifest()
        if manifest.get("cache_policy") != CLEAN_HISTORY_POLICY:
            raise ValueError("Registry prefix caching needs the clean-history manifest")
        if self.connector is not None or self.ec_connector is not None:
            raise ValueError("Registry clean-history caching does not support KV/EC transfer")
        directory = Path(os.environ["TACT_REGISTRY_EVIDENCE_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        self._cache_log = (directory / "scheduler.jsonl").open("a", buffering=1)
        self.tact_cache_policy = CleanHistoryPolicy(manifest, self._write_cache)
        native = self.kv_cache_manager.coordinator
        self.kv_cache_manager.coordinator = CleanHistoryCoordinator(native, self.tact_cache_policy)
        self._write_cache({"type": "cache_loaded", "policy": CLEAN_HISTORY_POLICY,
                           "coordinator": type(native).__name__,
                           "scheduler_block_size": native.scheduler_block_size,
                           "group_block_sizes": list(native.group_block_sizes)})

    def _write_cache(self, event):
        self._cache_log.write(json.dumps(event, allow_nan=False) + "\n")

    def add_request(self, request):
        if request.resumable or request.request_id in self.requests:
            raise ValueError("Registry clean-history caching does not support streaming input")
        self.tact_cache_policy.observe(request)
        try:
            return super().add_request(request)
        except BaseException:
            self.tact_cache_policy.entries.pop(request.request_id, None)
            raise

    def schedule(self, throttle_prefills: bool = False):
        # cache_blocks inside allocate_slots observes accepted outputs before
        # the GPU consumes the generated close token. No GPU readback is needed.
        result = super().schedule(throttle_prefills)
        for request in result.scheduled_new_reqs:
            label = (request.sampling_params.extra_args or {}).get("cache_probe_label")
            self._write_cache({"type": "cache_admitted", "request_id": request.req_id,
                               "label": label,
                               "cached_tokens": request.num_computed_tokens})
        return result

    def _free_blocks(self, request):
        super()._free_blocks(request)
        self.tact_cache_policy.release(request)
