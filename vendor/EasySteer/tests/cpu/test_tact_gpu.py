"""GPU graph replay guards: captured at zero mask, mutable slots, exact idle rows."""

import pytest
import torch
from vllm.model_hooks.steering.controllers.tact_states import TactStatesController
from vllm.model_hooks.steering.graph.kernels import apply_decoder_families

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("width", [8, 2048, 5120])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_residual", [False, True])
def test_capture_empty_then_replay_band_add_gate_and_mask_changes(width, dtype, with_residual):
    c = TactStatesController()
    mask = torch.zeros(4, device="cuda")
    rows = torch.zeros(4, device="cuda", dtype=torch.long)
    c.init_graph_buffers(
        {"additive", "tact_band"},
        {"transform": mask},
        capacity=4,
        hidden_size=width,
        max_rank=2,
        dtype=dtype,
        device="cuda",
        token_rows=rows,
    )
    h = torch.zeros((4, width), device="cuda", dtype=dtype)
    h[:, :2] = torch.tensor([2.0, -3.0], device="cuda", dtype=dtype)
    residual = torch.full_like(h, 0.25) if with_residual else None
    ot = torch.zeros(width, device="cuda")
    ot[0] = 1
    oa = torch.zeros(width, device="cuda")
    oa[1] = 1
    data = {
        "v_ot": ot,
        "v_oa": oa,
        "mu": torch.zeros_like(ot),
        "tau_ot": ot.new_tensor(0.5),
        "tau_oa": ot.new_tensor(1.0),
    }

    def run():
        return apply_decoder_families(
            c.graph_tables, mask, None, c.normalize_flag, rows, h, residual
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    baseline = h.clone()
    addresses = [t.data_ptr() for t in c.graph_row_tensors()]
    for algorithm, scale in [
        ("tact_cap", 1),
        ("tact_gate", 0),
        ("tact_gate", 2),
        ("tact_add", -0.5),
        ("tact_cap", 1),
    ]:
        c.set_graph_row(
            1, algorithm, torch.ones_like(ot) if algorithm == "tact_add" else data, scale
        )
        rows.copy_(torch.tensor([1, 1, 0, 1], device="cuda"))
        mask.copy_(torch.tensor([1.0, 1.0, 1.0, 0.0], device="cuda"))
        graph.replay()
        actual = output.clone()
        x = (baseline + residual).float() if residual is not None else baseline.float()
        if algorithm == "tact_add":
            delta = torch.full_like(x, scale)
        else:
            beta = 1 if algorithm == "tact_cap" else scale
            delta = torch.zeros_like(x)
            delta[:, 0] = -beta * (x[:, 0] - x[:, 0].clamp(-0.5, 0.5))
            delta[:, 1] = -beta * (x[:, 1] - x[:, 1].clamp(-1, 1))
        expected = baseline.clone()
        expected[:2] = baseline[:2] + delta[:2].to(dtype)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        assert torch.equal(actual[2:].view(torch.uint8), baseline[2:].view(torch.uint8))
        assert addresses == [t.data_ptr() for t in c.graph_row_tensors()]
        mask.zero_()
        graph.replay()
        assert torch.equal(output.view(torch.uint8), baseline.view(torch.uint8))
    c.clear_graph_row(1)
    mask.fill_(1)
    graph.replay()
    assert torch.equal(output.view(torch.uint8), baseline.view(torch.uint8))
