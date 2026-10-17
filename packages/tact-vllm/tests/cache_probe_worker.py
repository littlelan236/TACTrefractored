"""Test-only worker: capture actual per-layer caches after a completed prefill.

Only loaded by clean_history_smoke.py; production launch never selects this worker.
"""
import os
from pathlib import Path

import numpy as np
import torch
from vllm.v1.kv_cache_interface import MambaSpec

from tact_vllm.adapters.model_runner import TactGPUModelRunner
from tact_vllm.adapters.worker import TactWorker


class CacheProbeRunner(TactGPUModelRunner):
    def _model_forward(self, *args, **kwargs):
        result = super()._model_forward(*args, **kwargs)
        if self._tact_dummy or self._tact_scheduler is None:
            return result
        offset = 0
        for request_id in self.input_batch.req_ids:
            state = self.requests[request_id]
            row = self.input_batch.req_id_to_index[request_id]
            start = int(self.input_batch.num_computed_tokens_cpu[row])
            count = self._tact_scheduler.num_scheduled_tokens[request_id]
            end = start + count
            label = state.sampling_params.extra_args.get("cache_probe_label")
            if label and end == state.num_prompt_tokens:
                rank = self.vllm_config.parallel_config.rank
                arrays = {"last_hidden": result[offset + count - 1].detach().float().cpu().numpy()}
                context = self.compilation_config.static_forward_context
                for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
                    table = self.input_batch.block_table[gid]
                    size = table.block_size
                    ids = table.get_numpy_array()[row]
                    for layer_name in group.layer_names:
                        layer = context[layer_name]
                        cache = layer.kv_cache
                        if isinstance(group.kv_cache_spec, MambaSpec):
                            # Conv/SSM views have physical state block as dimension 0.
                            block = int(ids[(end - 1) // size])
                            for i, component in enumerate(cache):
                                arrays[f"{layer_name}/state{i}"] = (
                                    component[block].detach().float().cpu().numpy())
                        else:
                            if isinstance(cache, (tuple, list)):
                                assert len(cache) == 1
                                cache = cache[0]
                            num_blocks = self.kv_cache_config.num_blocks
                            axes = [i for i, d in enumerate(cache.shape) if d == num_blocks]
                            assert len(axes) == 1, (layer_name, cache.shape, num_blocks)
                            block_axis = axes[0]
                            # Include the allocated tail block. The comparison uses
                            # the same token prefix on both runs, so unwritten slots
                            # in a partial block are identical and harmless.
                            block_ids = torch.tensor(ids[:end // size], device=cache.device,
                                                     dtype=torch.long)
                            if end % size:
                                block_ids = torch.tensor(ids[: (end + size - 1) // size],
                                                          device=cache.device, dtype=torch.long)
                            arrays[f"{layer_name}/kv"] = cache.index_select(
                                block_axis, block_ids).movedim(block_axis, 0).detach().float().cpu().numpy()
                directory = Path(os.environ["TACT_CACHE_PROBE_DIR"])
                directory.mkdir(parents=True, exist_ok=True)
                np.savez(directory / f"{label}-rank{rank}.npz", **arrays)
            offset += count
        return result


class CacheProbeWorker(TactWorker):
    def init_device(self):
        super().init_device()
        # init_device precedes load_model/cache initialization. Replace only this
        # test worker's uninitialized runner, never a vLLM module or global class.
        self.model_runner = CacheProbeRunner(self.vllm_config, self.device)
