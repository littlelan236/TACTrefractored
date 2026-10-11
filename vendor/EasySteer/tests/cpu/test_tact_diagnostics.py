"""Detailed evidence must identify the real request, not a shared config owner."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_hooks.steering.diagnostics import encode_detail, write_step_details


def fixture():
    vector = SimpleNamespace(algorithm="tact_cap", payload={"sha256": "axis"}, scale=1)
    manager = SimpleNamespace(
        _tact_detail_step=3, slot_for_request=lambda request: 0,
        graph_batch_entries=lambda: {0: (1, SimpleNamespace(vectors=[vector]), [])})
    geometry = SimpleNamespace(
        query_start_loc_cpu=np.array([0, 1, 2]), req_ids=["a", "b"],
        num_computed=torch.tensor([10, 17]), num_prompt=torch.tensor([10, 15]),
        token_ids_cpu=lambda: np.array([248069, 248069]))
    controller = SimpleNamespace(layer_id=40, _op_key="model.norm::tact")
    values = np.array([1, 1, 5, -3, 2, 2, 2.01, -2.01, 3.16, 3.16, 1])
    return manager, geometry, controller, values


def test_shared_slot_still_identifies_each_request():
    manager, geometry, controller, values = fixture()
    record = encode_detail(manager, geometry, controller, 1, values, "FULL")
    assert record["request_id"] == "b" and record["slot"] == 0
    assert record["absolute_position"] == 17
    assert record["generation_position"] == 2
    assert record["token_id"] == 248069
    assert record["projection_expected"] == [2, -2]
    assert record["changed"] and record["site_kind"] == "final_norm"


def test_diagnostics_fail_if_routing_or_phase_is_wrong():
    manager, geometry, controller, values = fixture()
    values[1] = 2
    with pytest.raises(RuntimeError, match="slot mismatch"):
        encode_detail(manager, geometry, controller, 1, values, "FULL")
    values[1] = 1
    geometry.num_computed[1] = 14
    with pytest.raises(RuntimeError, match="prompt token"):
        encode_detail(manager, geometry, controller, 1, values, "FULL")


def test_disabled_diagnostics_do_not_touch_manager(monkeypatch):
    monkeypatch.delenv("TACT_EASYSTEER_DETAIL_LOG", raising=False)
    write_step_details(None, None, None)


def test_limit_counts_dropped_records_without_misattributing_requests(monkeypatch, tmp_path):
    import json

    manager, geometry, controller, values = fixture()
    controller.tact_details = torch.tensor(np.stack([values, values]))
    manager._controller_manager = SimpleNamespace(controllers={"site": controller})
    manager._graph_full = True
    monkeypatch.setenv("TACT_EASYSTEER_DETAIL_LOG", "1")
    monkeypatch.setenv("TACT_EASYSTEER_DETAIL_LOG_LIMIT", "1")
    monkeypatch.setenv("TACT_EASYSTEER_EVIDENCE_DIR", str(tmp_path))
    write_step_details(manager, geometry, "FULL")
    write_step_details(manager, geometry, "FULL")
    assert manager._tact_detail_records == 1
    assert manager._tact_detail_dropped == 3
    rows = [json.loads(line) for path in tmp_path.glob("token-details-*.jsonl")
            for line in path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["request_id"] == "a"
