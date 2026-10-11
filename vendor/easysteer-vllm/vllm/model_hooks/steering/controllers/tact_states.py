# SPDX-License-Identifier: Apache-2.0
"""FP32 tables on decoder residual states and final normalized outputs."""

import os

import torch

from vllm.model_hooks.components.registry import TACT_STATES

from .hidden_states import HiddenStatesController


class TactStatesController(HiddenStatesController):
    component_id = TACT_STATES

    def init_graph_buffers(self, families, masks, **kwargs):
        kwargs["dtype"] = torch.float32
        super().init_graph_buffers(families, masks, **kwargs)
        self.tact_evidence = (
            torch.zeros(7, device=kwargs["device"], dtype=torch.float32)
            if os.environ.get("TACT_EASYSTEER_DIAGNOSTICS")
            else None
        )

        self.tact_details = (
            torch.zeros((kwargs["token_rows"].numel(), 11),
                        device=kwargs["device"], dtype=torch.float32)
            if os.environ.get("TACT_EASYSTEER_DETAIL_LOG") == "1"
            else None
        )

        self.consumer_probe = None
        if os.environ.get("TACT_CONSUMER_CHECK") == "1":
            from vllm.model_hooks.steering.consumer_probe import ConsumerProbe
            self.consumer_probe = ConsumerProbe(
                kwargs["hidden_size"], int(os.environ["TACT_CONSUMER_CHECK_ROWS"]),
                kwargs["device"])

    def record_consumer_output(self, hidden, residual, original):
        if self.consumer_probe is not None:
            n = hidden.shape[0]
            active = (self.graph_mask[:n] != 0) & (self.graph_token_rows[:n] != 0)
            self.consumer_probe.produce(hidden, residual, active,
                (hidden.float() - original.float()).norm(dim=-1))

    def set_graph_row(self, row, algorithm, payload, scale, normalize=False):
        if normalize:
            raise ValueError("TACT does not renormalize outputs")
        self.clear_graph_row(row)
        super().set_graph_row(row, algorithm, payload, scale, normalize=False)
