"""Guard TACT payload identity, precision, writes, and graph slot reuse."""

import copy

import numpy as np
import pytest
import torch
from easysteer.vectors import to_json_payload
from vllm.model_hooks.components.registry import FinalNormOutputAdapter
from vllm.model_hooks.steering.algorithms import get_algorithm
from vllm.model_hooks.steering.api import to_engine_request
from vllm.model_hooks.steering.controllers.tact_states import TactStatesController
from vllm.model_hooks.steering.graph.kernels import apply_decoder_families
from vllm.model_hooks.steering.payloads import from_wire, materialize
from vllm.steer_vectors import ApplySpec, SteeringSpec, TactBandPayload, VectorSpec


def band(width=8):
    ot, oa = np.eye(width, dtype=np.float32)[:2]
    fields = {
        "v_ot": ot,
        "v_oa": oa,
        "mu": np.zeros(width, np.float32),
        "tau_ot": np.float32(0.5),
        "tau_oa": np.float32(1),
    }
    manifest = {
        "schema_version": 1,
        "layer_count": 2,
        "hidden_size": width,
        "sites": {
            "0": {"kind": "decoder_output", "block_index": 0},
            "2": {"kind": "final_norm", "block_index": None},
        },
    }
    return TactBandPayload({0: fields, 2: fields}, manifest, k=2)


def controller(device="cpu", width=8):
    c = TactStatesController()
    rows = torch.tensor([1, 2, 0, 1], device=device)
    mask = torch.tensor([1, 1, 1, 0], device=device)
    c.init_graph_buffers(
        {"additive", "tact_band"},
        {"transform": mask},
        capacity=2,
        hidden_size=width,
        max_rank=2,
        dtype=torch.bfloat16,
        device=device,
        token_rows=rows,
    )
    return c


def test_wire_roundtrip_identity_and_fp32_materialization():
    payload = band()
    wire = to_json_payload(payload)
    assert from_wire(wire).to_wire()["sha256"] == payload.to_wire()["sha256"]
    tensors = materialize(wire, "cpu", torch.bfloat16, [2])
    assert set(tensors) == {2}
    assert all(t.dtype == torch.float32 for t in tensors[2].values())
    corrupted = copy.deepcopy(wire)
    corrupted["extra"]["k"] = 3
    with pytest.raises(ValueError, match="sha256"):
        from_wire(corrupted)


def test_incomplete_or_unknown_wire_fields_are_rejected_before_materialization():
    from vllm.model_hooks.steering.payloads import validate_wire
    wire = band().to_wire()
    del wire["tensors"]["mu.layer.0"]
    with pytest.raises(ValueError, match="complete"):
        validate_wire(wire)
    wire = band().to_wire()
    wire["tensors"]["unexpected"] = wire["tensors"]["mu.layer.0"]
    with pytest.raises(ValueError, match="field name"):
        validate_wire(wire)


@pytest.mark.parametrize(
    "algorithm,scale",
    [("tact_cap", 1), ("tact_gate", 0), ("tact_gate", 0.5), ("tact_gate", 1), ("tact_gate", 2)],
)
def test_band_math_against_manual_projection_and_graph(algorithm, scale):
    data = materialize(band().to_wire(), "cpu", torch.bfloat16, None)[0]
    x = torch.tensor([[2.0, -3.0, 0, 0, 0, 0, 0, 0], [0.25, 0.5, 0, 0, 0, 0, 0, 0]])
    expected = x.clone()
    beta = 1 if algorithm == "tact_cap" else scale
    expected[0, 0] -= beta * 1.5
    expected[0, 1] += beta * 2
    algo = get_algorithm(algorithm)()
    algo.set_payload(data, scale)
    torch.testing.assert_close(algo._transform(x, algo.payload), expected)
    c = controller()
    c.set_graph_row(1, algorithm, data, scale)
    c.set_graph_row(2, algorithm, data, scale)
    hidden = torch.cat([x, x])
    out = apply_decoder_families(
        c.graph_tables, c.graph_mask, None, c.normalize_flag, c.graph_token_rows, hidden, None
    )
    torch.testing.assert_close(out[:2], expected)
    assert torch.equal(out[2:], hidden[2:])


def test_slot_clear_switch_add_band_and_zero_delta_preserve_bits():
    c = controller()
    data = materialize(band().to_wire(), "cpu", torch.bfloat16, None)[0]
    c.set_graph_row(1, "tact_cap", data, 1)
    c.set_graph_row(1, "tact_add", torch.ones(8), -0.5)
    assert c.graph_tables["tact_band"]["BETA"][1] == 0
    assert c.graph_tables["additive"]["V"].dtype == torch.float32
    c.clear_graph_row(1)
    assert all(torch.count_nonzero(t[1]) == 0 for t in c.graph_row_tensors())
    hidden = torch.tensor([[0.0, -0.0, 1, -1, 2, 3, 4, 5]] * 4, dtype=torch.bfloat16)
    residual = torch.full_like(hidden, 64)
    c.set_graph_row(1, "tact_gate", data, 0)
    out = apply_decoder_families(
        c.graph_tables, c.graph_mask, None, c.normalize_flag, c.graph_token_rows, hidden, residual
    )
    assert torch.equal(out.view(torch.int16), hidden.view(torch.int16))
    algo = get_algorithm("tact_gate")()
    algo.set_payload(data, 0)
    original = hidden.clone()
    algo._batch_transform_tensor(hidden, torch.tensor([0]), algo.payload, residual)
    assert torch.equal(hidden.view(torch.int16), original.view(torch.int16))


def test_final_norm_auxiliary_is_never_added_as_residual():
    hidden, aux = torch.ones(2, 8), torch.full((2, 8), 10.0)
    values, residual, extras, fmt = FinalNormOutputAdapter.read_output((hidden, aux))
    assert residual is None and values is hidden
    rebuilt = FinalNormOutputAdapter.write_output(values * 2, residual, extras, fmt, (hidden, aux))
    assert rebuilt[1] is aux


@pytest.mark.parametrize(
    "algorithm,scale", [("tact_cap", 0.5), ("tact_gate", -1), ("tact_gate", float("inf"))]
)
def test_invalid_scale_rejected_by_public_api(algorithm, scale):
    with pytest.raises(ValueError):
        VectorSpec(
            algorithm=algorithm, data=band(), scale=scale, apply=ApplySpec(generation_tokens=[123])
        )


def test_request_identity_changes_with_scale_and_native_selection():
    def request(scale):
        return to_engine_request(
            SteeringSpec(
                vectors=[
                    VectorSpec(
                        algorithm="tact_gate",
                        data=band(),
                        scale=scale,
                        apply=ApplySpec(generation_tokens=[123]),
                    )
                ]
            )
        )

    from vllm.model_hooks.steering.request import config_fingerprint

    assert config_fingerprint(request(0.5)) != config_fingerprint(request(1))


def test_tact_add_precision_override_does_not_affect_native_direction_cache():
    from types import SimpleNamespace
    from vllm.model_hooks.steering.payload_cache import PayloadCache
    from vllm.steer_vectors import DirectionVector
    cache = PayloadCache("cpu", SimpleNamespace(max_steer_vectors=4,
                                               adapter_dtype=torch.bfloat16))
    wire = DirectionVector({0: np.array([1.001, 2.003], np.float32)}).to_wire()
    native = cache.get(wire)
    tact = cache.get(wire, dtype_override=torch.float32)
    assert native[0].dtype == torch.bfloat16
    assert tact[0].dtype == torch.float32
    assert torch.equal(tact[0], torch.tensor([1.001, 2.003]))
