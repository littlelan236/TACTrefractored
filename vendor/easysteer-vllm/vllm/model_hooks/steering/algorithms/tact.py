# SPDX-License-Identifier: Apache-2.0
"""TACT dual-axis bands with direct, delta-form hidden-branch writes."""

import math

import torch

from .base import BaseSteerVectorAlgorithm
from .direct import DirectAlgorithm
from .registry import register_algorithm


class TactDeltaWrites:
    def _batch_transform_tensor(
        self, hidden_states, positions_tensor, params, residual=None
    ):
        selected = hidden_states.index_select(0, positions_tensor)
        complete = (
            selected + residual.index_select(0, positions_tensor)
            if residual is not None
            else selected
        )
        delta = self._delta(complete, params).to(selected.dtype)
        updated = torch.where(delta != 0, selected + delta, selected)
        hidden_states.index_copy_(0, positions_tensor, updated)
        return hidden_states


@register_algorithm("tact_add")
class TactAddAlgorithm(TactDeltaWrites, DirectAlgorithm):
    """Native direct math/lowering and additive kernels on TACT states."""

    def _delta(self, hidden, params):
        return params.expand_as(hidden)


class TactBandAlgorithm(TactDeltaWrites, BaseSteerVectorAlgorithm):
    graph_family = "tact_band"
    is_cap = False

    @classmethod
    def graph_lower(cls, payload, scale):
        if not math.isfinite(scale) or scale < 0 or (cls.is_cap and scale != 1):
            raise ValueError("Invalid TACT band scale")
        beta = 1.0 if cls.is_cap else scale
        return {
            "V_OT": payload["v_ot"],
            "V_OA": payload["v_oa"],
            "MU": payload["mu"],
            "TAU_OT": payload["tau_ot"],
            "TAU_OA": payload["tau_oa"],
            "BETA": payload["tau_ot"].new_tensor(beta),
        }

    def _delta(self, hidden, params):
        beta = 1.0 if self.is_cap else params.get("scale_factor", 1.0)
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("Invalid TACT beta")
        centered = hidden.float() - params["mu"]
        p_ot = (centered * params["v_ot"]).sum(-1, keepdim=True)
        p_oa = (centered * params["v_oa"]).sum(-1, keepdim=True)
        e_ot = p_ot - p_ot.clamp(-params["tau_ot"], params["tau_ot"])
        e_oa = p_oa - p_oa.clamp(-params["tau_oa"], params["tau_oa"])
        return -beta * (e_ot * params["v_ot"] + e_oa * params["v_oa"])

    def _transform(self, hidden_state, params):
        delta = self._delta(hidden_state, params).to(hidden_state.dtype)
        return torch.where(delta != 0, hidden_state + delta, hidden_state)


@register_algorithm("tact_cap")
class TactCapAlgorithm(TactBandAlgorithm):
    is_cap = True


@register_algorithm("tact_gate")
class TactGateAlgorithm(TactBandAlgorithm):
    pass
