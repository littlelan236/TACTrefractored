from __future__ import annotations

import torch


def dual_axis_delta(state, axes, center, sigma, k, beta, additive):
    projection = (state.float() - center) @ axes.T
    threshold = k[:, None] * sigma
    excess = projection - torch.clamp(projection, -threshold, threshold)
    delta = -beta[:, None] * (excess @ axes) + additive[:, None] * axes.sum(0)
    return delta, projection, threshold


def apply_site(hidden, residual, runtime, site: int):
    n = hidden.shape[0]
    rows = runtime.request_rows[:n].long()
    active = runtime.decision_mask[:n] & runtime.site_enabled[rows, site]
    state = hidden.float() if residual is None else hidden.float() + residual.float()
    delta, before, tau = dual_axis_delta(
        state,
        runtime.axes[site],
        runtime.center[site],
        runtime.sigma[site],
        runtime.k[rows],
        runtime.beta[rows],
        runtime.additive[rows],
    )
    update = active & torch.any(delta != 0, dim=-1)
    original = hidden if residual is None else residual
    modified = (original.float() + delta).to(original.dtype)
    result = torch.where(update[:, None], modified, original)
    after_state = result.float() if residual is None else hidden.float() + result.float()
    after = (after_state - runtime.center[site]) @ runtime.axes[site].T
    values = torch.cat((before, tau, after, torch.linalg.vector_norm(delta, dim=-1)[:, None]), 1)
    values = torch.where(active[:, None], values, torch.zeros_like(values))
    runtime.evidence[:, site].index_add_(0, rows, values)
    runtime.counts[:, site].index_add_(0, rows, active.to(torch.int32))
    if runtime.consumer_check:
        next_hidden = result if residual is None else hidden
        next_residual = None if residual is None else result
        actual_norm = (result.float() - original.float()).norm(dim=-1)
        runtime.consumer_probes[site].produce(
            next_hidden, next_residual, active, actual_norm)
    return result
