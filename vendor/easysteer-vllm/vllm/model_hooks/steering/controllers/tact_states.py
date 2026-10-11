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

    def set_graph_row(self, row, algorithm, payload, scale, normalize=False):
        if normalize:
            raise ValueError("TACT does not renormalize outputs")
        self.clear_graph_row(row)
        super().set_graph_row(row, algorithm, payload, scale, normalize=False)
