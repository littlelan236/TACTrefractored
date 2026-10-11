# SPDX-License-Identifier: Apache-2.0
"""One captured launch for additive and dual-axis band configurations."""

import torch
from torch.library import wrap_triton

from vllm.triton_utils import tl, triton


@triton.jit
def _tact_band_kernel(
    Hidden,
    Residual,
    Add,
    OT,
    OA,
    Mu,
    TauOT,
    TauOA,
    Beta,
    Mask,
    Rows,
    Out,
    Evidence,
    Details,
    RECORD: tl.constexpr,
    DETAIL: tl.constexpr,
    H: tl.constexpr,
    HS0: tl.constexpr,
    HS1: tl.constexpr,
    RS0: tl.constexpr,
    RS1: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_ADD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < H
    dtype = Hidden.dtype.element_ty
    hidden = tl.load(Hidden + token * HS0 + cols * HS1, valid, other=0).to(tl.float32)
    out = hidden
    row = tl.load(Rows + token)
    selected = tl.load(Mask + token)
    if DETAIL:
        tl.store(Details + token * 11, 0.0)
    if (selected != 0) & (row != 0):
        beta = tl.load(Beta + row)
        p_ot, p_oa, tau_ot, tau_oa = 0.0, 0.0, 0.0, 0.0
        ideal_magnitude = 0.0
        if beta != 0:
            x = hidden
            if HAS_RESIDUAL:
                residual = tl.load(
                    Residual + token * RS0 + cols * RS1, valid, other=0
                ).to(tl.float32)
                x = (hidden + residual).to(dtype).to(tl.float32)
            ot = tl.load(OT + row * H + cols, valid, other=0)
            oa = tl.load(OA + row * H + cols, valid, other=0)
            mu = tl.load(Mu + row * H + cols, valid, other=0)
            centered = tl.where(valid, x - mu, 0.0)
            p_ot = tl.sum(centered * ot, 0)
            p_oa = tl.sum(centered * oa, 0)
            tau_ot = tl.load(TauOT + row)
            tau_oa = tl.load(TauOA + row)
            e_ot = p_ot - tl.minimum(tl.maximum(p_ot, -tau_ot), tau_ot)
            e_oa = p_oa - tl.minimum(tl.maximum(p_oa, -tau_oa), tau_oa)
            ideal_delta = -beta * (e_ot * ot + e_oa * oa)
            if DETAIL:
                ideal_magnitude = tl.sqrt(tl.sum(ideal_delta * ideal_delta, 0))
            delta = ideal_delta.to(dtype).to(tl.float32)
            out = tl.where(
                delta != 0, (hidden + delta).to(dtype).to(tl.float32), hidden
            )
        elif HAS_ADD:
            if DETAIL:
                add_value = tl.load(Add + row * H + cols, valid, other=0)
                ideal_magnitude = tl.sqrt(tl.sum(add_value * add_value, 0))
            delta = (
                tl.load(Add + row * H + cols, valid, other=0).to(dtype).to(tl.float32)
            )
            out = tl.where(
                delta != 0, (hidden + delta).to(dtype).to(tl.float32), hidden
            )
        if DETAIL:
            difference = out - hidden
            actual_norm = tl.sqrt(tl.sum(difference * difference, 0))
            after_ot, after_oa = 0.0, 0.0
            if beta != 0:
                after_x = out
                if HAS_RESIDUAL:
                    after_residual = tl.load(
                        Residual + token * RS0 + cols * RS1,
                        valid, other=0).to(tl.float32)
                    after_x = (out + after_residual).to(dtype).to(tl.float32)
                detail_mu = tl.load(Mu + row * H + cols, valid, other=0)
                detail_ot = tl.load(OT + row * H + cols, valid, other=0)
                detail_oa = tl.load(OA + row * H + cols, valid, other=0)
                after_centered = tl.where(valid, after_x - detail_mu, 0.0)
                after_ot = tl.sum(after_centered * detail_ot, 0)
                after_oa = tl.sum(after_centered * detail_oa, 0)
            tl.store(Details + token * 11, 1.0)
            tl.store(Details + token * 11 + 1, row)
            tl.store(Details + token * 11 + 2, p_ot)
            tl.store(Details + token * 11 + 3, p_oa)
            tl.store(Details + token * 11 + 4, tau_ot)
            tl.store(Details + token * 11 + 5, tau_oa)
            tl.store(Details + token * 11 + 6, after_ot)
            tl.store(Details + token * 11 + 7, after_oa)
            tl.store(Details + token * 11 + 8, actual_norm)
            tl.store(Details + token * 11 + 9, ideal_magnitude)
            tl.store(Details + token * 11 + 10, beta)
        if RECORD:
            tl.atomic_add(Evidence, 1.0)
            difference = out - hidden
            magnitude = tl.sum(tl.where(valid, difference * difference, 0.0), 0)
            tl.atomic_add(Evidence + 1, tl.cast(magnitude > 0, tl.float32))
            tl.store(Evidence + 2, tl.sqrt(magnitude))
            if beta != 0:
                tl.store(Evidence + 3, p_ot)
                tl.store(Evidence + 4, p_oa)
                tl.store(Evidence + 5, tau_ot)
                tl.store(Evidence + 6, tau_oa)
    tl.store(Out + token * H + cols, out, valid)


def apply_tact_families(
    tables, mask, rows, hidden, residual, evidence=None, details=None
):
    if "tact_band" not in tables:
        from vllm.model_hooks.steering.graph.kernels import apply_decoder_families

        return apply_decoder_families(tables, mask, None, None, rows, hidden, residual)
    band = tables["tact_band"]
    if hidden.is_cuda:
        n, h = hidden.shape
        out = torch.empty_like(hidden, memory_format=torch.contiguous_format)
        additive = tables.get("additive", {}).get("V", band["V_OT"])
        wrap_triton(_tact_band_kernel)[(n,)](
            hidden,
            residual,
            additive,
            band["V_OT"],
            band["V_OA"],
            band["MU"],
            band["TAU_OT"],
            band["TAU_OA"],
            band["BETA"],
            mask,
            rows,
            out,
            evidence,
            details,
            RECORD=evidence is not None,
            DETAIL=details is not None,
            H=h,
            HS0=hidden.stride(0),
            HS1=hidden.stride(1),
            RS0=0 if residual is None else residual.stride(0),
            RS1=0 if residual is None else residual.stride(1),
            HAS_RESIDUAL=residual is not None,
            HAS_ADD="additive" in tables,
            BLOCK=triton.next_power_of_2(h),
            num_warps=4 if h <= 2048 else 8,
            enable_fp_fusion=False,
        )
        return out
    n = hidden.shape[0]
    rt = rows[:n]
    x = (hidden + residual).float() if residual is not None else hidden.float()
    centered = x - band["MU"][rt]
    ot, oa = band["V_OT"][rt], band["V_OA"][rt]
    p_ot = (centered * ot).sum(-1)
    p_oa = (centered * oa).sum(-1)
    e_ot = p_ot - p_ot.clamp(-band["TAU_OT"][rt], band["TAU_OT"][rt])
    e_oa = p_oa - p_oa.clamp(-band["TAU_OA"][rt], band["TAU_OA"][rt])
    beta = band["BETA"][rt, None]
    delta = -beta * (e_ot[:, None] * ot + e_oa[:, None] * oa)
    if "additive" in tables:
        delta = torch.where(beta != 0, delta, tables["additive"]["V"][rt])
    delta = delta.to(hidden.dtype)
    enabled = (mask[:n, None] != 0) & (rt[:, None] != 0) & (delta != 0)
    output = torch.where(enabled, hidden + delta, hidden)
    if details is not None:
        active = (mask[:n] != 0) & (rt != 0)
        after = (output + residual).float() if residual is not None else output.float()
        values = torch.stack((active.float(), rt.float(), p_ot, p_oa,
            band["TAU_OT"][rt], band["TAU_OA"][rt],
            ((after - band["MU"][rt]) * ot).sum(-1),
            ((after - band["MU"][rt]) * oa).sum(-1),
            (output.float() - hidden.float()).norm(dim=-1),
            (delta.float()).norm(dim=-1), beta[:, 0]), dim=-1)
        details[:n].copy_(values)
    return output
