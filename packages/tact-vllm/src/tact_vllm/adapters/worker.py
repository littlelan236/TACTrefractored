# SPDX-License-Identifier: Apache-2.0
# init_device adapted from vLLM 0.30.0. Other worker behavior remains inherited.
from vllm.v1.worker import gpu_worker as native

from tact_vllm.compat import check_runtime


class TactWorker(native.Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        check_runtime(self.vllm_config)

    def init_device(self):
        if self.device_config.device_type == "cuda":
            native.os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            parallel_config = self.parallel_config
            if (
                parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and (parallel_config.nnodes_within_dp == 1)
            ):
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_index
                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size
                    * self.parallel_config.tensor_parallel_size
                )
                self.local_rank += dp_local_rank * tp_pp_world_size
            assigned_physical_gpu_ids = parallel_config.assigned_physical_gpu_ids
            if assigned_physical_gpu_ids is not None:
                from vllm.platforms.interface import set_assigned_physical_gpu_ids

                set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)
                assert self.local_rank < len(assigned_physical_gpu_ids), (
                    f"local_rank {self.local_rank} is out of bounds for assigned_physical_gpu_ids {assigned_physical_gpu_ids}"
                )
                if parallel_config.distributed_executor_backend not in ("ray", "external_launcher"):
                    assert self.parallel_config.local_world_size <= len(
                        assigned_physical_gpu_ids
                    ), (
                        f"local_world_size ({self.parallel_config.local_world_size}) exceeds assigned_physical_gpu_ids count ({len(assigned_physical_gpu_ids)})"
                    )
            else:
                assert self.local_rank < native.torch.accelerator.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of bounds for {native.torch.accelerator.device_count()} devices."
                )
            visible_device_index = native.current_platform.logical_device_id_to_visible_device_id(
                self.local_rank
            )
            self.device = native.torch.device(f"cuda:{visible_device_index}")
            native.torch.accelerator.set_device_index(self.device)
            native.current_platform.check_if_supports_dtype(self.model_config.dtype)
            native.init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                native.current_platform.dist_backend,
            )
            if self.use_v2_model_runner:
                native.logger.info_once("Using V2 Model Runner")
            native.set_random_seed(self.model_config.seed)
            native.gc.collect()
            native.torch.accelerator.empty_cache()
            self.init_snapshot = init_snapshot = native.MemorySnapshot(device=self.device)
            self.requested_memory = native.request_memory(init_snapshot, self.cache_config)
            native.logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            native.logger.debug(
                "worker requested memory: %sGiB", native.format_gib(self.requested_memory)
            )
        else:
            raise RuntimeError(f"Unsupported device type: {self.device_config.device}")
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        native.init_workspace_manager(
            self.device,
            num_ubatches,
            native._num_workspace_lanes(self.vllm_config, self.use_v2_model_runner),
        )
        if self.use_v2_model_runner:
            if self.vllm_config.is_mm_encoder_only:
                from vllm.v1.worker.mm_encoder_model_runner import (
                    MMEncoderModelRunner as GPUModelRunnerV2,
                )
            else:
                from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
            self.model_runner: native.GPUModelRunner = GPUModelRunnerV2(
                self.vllm_config, self.device
            )
        else:
            from tact_vllm.adapters.model_runner import TactGPUModelRunner as GPUModelRunnerV1

            self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)
        if self.rank == 0:
            native.report_usage_stats(self.vllm_config)
