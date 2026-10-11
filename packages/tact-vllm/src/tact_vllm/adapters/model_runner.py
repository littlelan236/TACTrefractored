from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.forward_context import get_forward_context
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from tact_vllm.runtime.buffers import TactRuntimeBuffers, load_manifest
from tact_vllm.runtime.ledger import RequestLedger
from tact_vllm.runtime.plans import parse_plan


class TactGPUModelRunner(GPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tact_manifest = load_manifest()
        self.tact_ledger = RequestLedger(self.tact_manifest["request_capacity"])
        self.tact_runtime = None
        self._tact_scheduler = None
        self._tact_dummy = False
        self._tact_log = None
        self._tact_sequence = 0
        capacity = self.tact_manifest["token_capacity"]
        self._tact_rows = torch.zeros(capacity, dtype=torch.int32, pin_memory=True)
        self._tact_eligible = torch.zeros(capacity, dtype=torch.bool, pin_memory=True)

    def _write(self, value):
        if self._tact_log is None:
            directory = Path(os.environ["TACT_REGISTRY_EVIDENCE_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            rank = self.vllm_config.parallel_config.rank
            self._tact_log = (directory / f"worker-{rank}.jsonl").open("a", buffering=1)
        self._tact_log.write(json.dumps(value, allow_nan=False) + "\n")

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        runtimes = [m for m in self.get_model().modules() if isinstance(m, TactRuntimeBuffers)]
        if len(runtimes) != 1:
            raise RuntimeError("Registry model did not construct exactly one TACT backbone")
        self.tact_runtime = runtimes[0]
        self._write(
            {
                "type": "loaded",
                "model": type(self.get_model()).__name__,
                "backbones": [
                    type(m).__name__
                    for m in self.get_model().modules()
                    if type(m).__name__ == "TactQwen3_5Model"
                ],
                "bundle_id": self.tact_manifest["bundle_id"],
                "runner": "v030-v1",
                "buffer_addresses": {n: b.data_ptr() for n, b in self.tact_runtime.named_buffers()},
            }
        )
        return result

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        runtime = self.tact_runtime
        for request_id in scheduler_output.finished_req_ids:
            if request_id in self.tact_ledger.entries:
                entry = self.tact_ledger.entries[request_id]
                counts = runtime.counts[entry.slot].cpu().tolist()
                sums = runtime.evidence[entry.slot].cpu().tolist()
                self._write(
                    {
                        "type": "finished",
                        "request_id": request_id,
                        "slot": entry.slot,
                        "epoch": entry.epoch,
                        "strategy": entry.plan.strategy,
                        "first_close_position": entry.first_close,
                        "site_counts": counts,
                        "evidence_sums": sums,
                    }
                )
                runtime.reset_slot(entry.slot)
                self.tact_ledger.release(request_id)
        resumed = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        if resumed:
            raise RuntimeError(
                "TACT preemption/resumption is not validated; refusing resumed requests"
            )
        self._tact_scheduler = scheduler_output
        try:
            return super().execute_model(scheduler_output, intermediate_tensors)
        finally:
            self._tact_scheduler = None

    def _stage(self, n):
        runtime, scheduled = self.tact_runtime, self._tact_scheduler
        if n > runtime.token_capacity:
            raise RuntimeError("TACT graph bucket exceeds fixed token buffer capacity")
        self._tact_rows.zero_()
        self._tact_eligible.zero_()
        offset = 0
        staged = []
        for request_id in self.input_batch.req_ids:
            state = self.requests[request_id]
            count = scheduled.num_scheduled_tokens[request_id]
            plan = parse_plan(state.sampling_params.extra_args, self.tact_manifest)
            new = request_id not in self.tact_ledger.entries
            entry = self.tact_ledger.admit(request_id, plan)
            if new:
                runtime.set_plan(entry.slot, plan)
            entry.observe(
                state.output_token_ids, state.num_prompt_tokens, self.tact_manifest["close_id"]
            )
            row = self.input_batch.req_id_to_index[request_id]
            start = int(self.input_batch.num_computed_tokens_cpu[row])
            self._tact_rows[offset : offset + count] = entry.slot
            eligible = np.asarray(entry.eligible_positions(start, count), dtype=np.bool_)
            self._tact_eligible[offset : offset + count].copy_(torch.from_numpy(eligible))
            staged.append(
                {
                    "request_id": request_id,
                    "slot": entry.slot,
                    "epoch": entry.epoch,
                    "start": start,
                    "count": count,
                    "first_close": entry.first_close,
                    "strategy": plan.strategy,
                    "k": plan.k,
                    "beta": plan.beta,
                    "additive_scale": plan.additive_scale,
                }
            )
            offset += count
        if offset != scheduled.total_num_scheduled_tokens or offset > n:
            raise RuntimeError("TACT flattened batch geometry mismatch")
        runtime.request_rows[:n].copy_(self._tact_rows[:n], non_blocking=False)
        runtime.eligible[:n].copy_(self._tact_eligible[:n], non_blocking=False)
        runtime.token_ids[:n].copy_(self.input_ids.gpu[:n])
        runtime.fill_decision_mask()
        return staged

    def _model_forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **model_kwargs,
    ):
        runtime = self.tact_runtime
        if runtime is None:
            raise RuntimeError("TACT runtime must be bound before profile/capture")
        if self._tact_dummy or self._tact_scheduler is None:
            runtime.disable()
            return super()._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
        n = positions.shape[-1]
        staged = self._stage(n)
        context = get_forward_context()
        mode = context.cudagraph_runtime_mode.name
        descriptor = context.batch_descriptor
        wrappers = [
            w
            for w in CUDAGraphWrapper._all_instances
            if w.runtime_mode.name == mode
            and descriptor in w.concrete_cudagraph_entries
            and w.concrete_cudagraph_entries[descriptor].cudagraph is not None
        ]
        result = super()._model_forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
        # A final request may never receive another scheduler tick. Persist its
        # one decision event now; synchronization is limited to that boundary.
        for entry in staged:
            close = entry["first_close"]
            if close is not None and entry["start"] <= close < entry["start"] + entry["count"]:
                slot = entry["slot"]
                self._write({"type": "decision", **entry, "mode": mode,
                             "site_counts": runtime.counts[slot].cpu().tolist(),
                             "evidence_sums": runtime.evidence[slot].cpu().tolist()})
        self._tact_sequence += 1
        self._write(
            {
                "type": "dispatch",
                "sequence": self._tact_sequence,
                "mode": mode,
                "bucket": n,
                "full_replay": mode == "FULL" and bool(wrappers),
                "requests": staged,
            }
        )
        return result

    def _dummy_run(self, *args, **kwargs):
        previous = self._tact_dummy
        self._tact_dummy = True
        if self.tact_runtime is not None:
            self.tact_runtime.disable()
        try:
            return super()._dummy_run(*args, **kwargs)
        finally:
            self._tact_dummy = previous

    def shutdown(self):
        if self._tact_log is not None:
            self._tact_log.close()
            self._tact_log = None
        return super().shutdown()
