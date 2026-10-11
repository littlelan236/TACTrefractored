import numpy as np
import pytest
import torch

from tact_vllm.runtime.buffers import TactRuntimeBuffers
from tact_vllm.runtime.ledger import RequestLedger
from tact_vllm.runtime.ops import apply_site, dual_axis_delta
from tact_vllm.runtime.plans import parse_plan


def manifest():
    return {
        "schema_version": 1,
        "bundle_id": "test",
        "token_capacity": 8,
        "request_capacity": 2,
        "hidden_size": 3,
        "axes_path": None,
        "sites": [{"kind": "decoder_output", "block_index": 0}],
        "close_id": 9,
    }


def fields(strategy="cap"):
    return {
        "tact_schema": 1,
        "tact_bundle_id": "test",
        "tact_strategy": strategy,
        "tact_k": 1,
        "tact_lambda": 0.5,
        "tact_additive_scale": -1,
        "tact_site_ids": [0],
    }


def test_plan_rejects_silent_fallback_or_invalid_parameters():
    for extra in (
        {},
        {**fields(), "tact_k": float("nan")},
        {**fields(), "tact_site_ids": [1]},
        {**fields(), "tact_bundle_id": "other"},
    ):
        with pytest.raises(ValueError):
            parse_plan(extra, manifest())


def test_first_generated_close_survives_batch_reorder_and_new_slot_epoch():
    ledger = RequestLedger(2)
    entry = ledger.admit("first", parse_plan(fields(), manifest()))
    entry.observe([2, 9, 4, 9], 10, 9)
    assert entry.first_close == 11
    assert entry.eligible_positions(10, 4) == [False, True, False, False]
    assert entry.eligible_positions(0, 10) == [False] * 10
    assert entry.eligible_positions(11, 1) == [True]  # same logical position on recompute
    old = ledger.release("first")
    new = ledger.admit("second", parse_plan(fields("none"), manifest()))
    assert new.slot == old.slot and new.epoch == old.epoch + 1
    assert new.first_close is None


def test_cap_gated_and_add_match_independent_scalar_reference():
    axes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    state = torch.tensor([[4.0, -3.0, 5.0], [0.5, 0.5, 3.0]])
    center = torch.tensor([1.0, 0.0, 0.0])
    sigma = torch.tensor([1.0, 2.0])
    for beta, additive in [(1.0, 0.0), (0.5, 0.0), (0.0, -1.0), (0.0, 0.0)]:
        delta, _, _ = dual_axis_delta(
            state,
            axes,
            center,
            sigma,
            torch.ones(2),
            torch.full((2,), beta),
            torch.full((2,), additive),
        )
        expected = []
        for row in state.numpy():
            out = np.zeros(3)
            for v, scale in zip(axes.numpy(), sigma.numpy()):
                p = float((row - center.numpy()) @ v)
                out += -beta * (p - max(-scale, min(scale, p))) * v + additive * v
            expected.append(out)
        np.testing.assert_allclose(delta.numpy(), expected)


def test_residual_reads_complete_state_and_preserves_inactive_rows():
    runtime = TactRuntimeBuffers(manifest())
    runtime.axes[0].copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    runtime.sigma[0].fill_(1)
    runtime.set_plan(1, parse_plan(fields(), manifest()))
    runtime.request_rows[:2] = 1
    runtime.decision_mask[0] = True
    hidden = torch.tensor([[3.0, -4.0, 0.0], [3.0, 3.0, 0.0]], dtype=torch.bfloat16)
    residual = torch.tensor([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]], dtype=torch.bfloat16)
    result = apply_site(hidden, residual, runtime, 0)
    torch.testing.assert_close(
        hidden[0].float() + result[0].float(), torch.tensor([1.0, -1.0, 2.0])
    )
    assert torch.equal(result[1], residual[1])
    assert runtime.counts[1, 0] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_capture_replay_reads_changed_parameters_and_padding_mask():
    runtime = TactRuntimeBuffers(manifest(), device="cuda")
    runtime.axes[0].copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda"))
    runtime.sigma[0].fill_(1)
    runtime.set_plan(1, parse_plan(fields(), manifest()))
    runtime.request_rows[:2] = 1
    hidden = torch.tensor([[4.0, -3.0, 2.0], [7.0, 8.0, 9.0]], device="cuda")
    address = runtime.beta.data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            apply_site(hidden, None, runtime, 0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = apply_site(hidden, None, runtime, 0)
    runtime.decision_mask[0] = True
    for beta, expected in [
        (1.0, [1.0, -1.0, 2.0]),
        (0.5, [2.5, -2.0, 2.0]),
        (0.0, [4.0, -3.0, 2.0]),
    ]:
        runtime.beta[1] = beta
        graph.replay()
        torch.testing.assert_close(output[0].cpu(), torch.tensor(expected))
        assert torch.equal(output[1], hidden[1])
    runtime.disable()
    graph.replay()
    assert torch.equal(output, hidden)
    assert runtime.beta.data_ptr() == address
