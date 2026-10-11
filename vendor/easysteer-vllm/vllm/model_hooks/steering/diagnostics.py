# SPDX-License-Identifier: Apache-2.0
"""Opt-in, post-forward diagnostics; never called inside a captured graph."""

import json
import os
from pathlib import Path

import numpy as np


def encode_detail(manager, geometry, controller, flat_position, values, mode):
    owner = int(np.searchsorted(
        geometry.query_start_loc_cpu[1:], flat_position, side="right"))
    request_id = geometry.req_ids[owner]
    slot = manager.slot_for_request(request_id)
    if slot is None or int(values[1]) != slot + 1:
        raise RuntimeError("TACT diagnostic request/config slot mismatch")
    entry = manager.graph_batch_entries()[slot]
    vector = entry[1].vectors[0]
    absolute = int(geometry.num_computed[owner]) + (
        flat_position - int(geometry.query_start_loc_cpu[owner]))
    prompt = int(geometry.num_prompt[owner])
    if absolute < prompt:
        raise RuntimeError("TACT generation selector changed a prompt token")
    before, threshold = values[2:4].tolist(), values[4:6].tolist()
    beta = float(values[10])
    band_projection = vector.algorithm in {"tact_cap", "tact_gate"} and beta != 0
    expected = ([p - beta * (p - max(-tau, min(tau, p)))
                 for p, tau in zip(before, threshold)]
                if band_projection else None)
    return {
        "type": "token_site", "schema_version": 1, "pid": os.getpid(),
        "request_id": request_id, "slot": slot,
        "algorithm": vector.algorithm, "payload_hash": vector.payload.get("sha256"),
        "scale": float(vector.scale), "mode": mode,
        "engine_step": manager._tact_detail_step,
        "flat_position": int(flat_position), "absolute_position": absolute,
        "generation_position": absolute - prompt,
        "token_id": int(geometry.token_ids_cpu()[flat_position]),
        "site_id": controller.layer_id, "module": controller._op_key,
        "site_kind": "final_norm" if ".norm::" in controller._op_key
        else "decoder_output",
        "projection_before": before if band_projection else None,
        "threshold": threshold if band_projection else None,
        "projection_after": values[6:8].tolist() if band_projection else None,
        "projection_expected": expected,
        "actual_hidden_delta_norm": float(values[8]),
        "ideal_delta_norm": float(values[9]), "beta": beta,
        "changed": bool(values[8] > 0),
        "scope": "Actual captured-kernel scalars; full hidden vectors not stored",
    }


def write_step_details(manager, geometry, mode):
    if os.environ.get("TACT_EASYSTEER_DETAIL_LOG") != "1":
        return
    if not manager._graph_full:
        raise RuntimeError("Detailed TACT diagnostics require in_graph mode")
    manager._tact_detail_step = getattr(manager, "_tact_detail_step", 0) + 1
    manager._tact_detail_records = getattr(manager, "_tact_detail_records", 0)
    manager._tact_detail_dropped = getattr(manager, "_tact_detail_dropped", 0)
    limit = int(os.environ.get("TACT_EASYSTEER_DETAIL_LOG_LIMIT", "20000"))
    n = int(geometry.query_start_loc_cpu[-1])
    records = []
    if manager._controller_manager is not None:
        for controller in manager._controller_manager.controllers.values():
            buffer = getattr(controller, "tact_details", None)
            if buffer is None:
                continue
            # Diagnostics deliberately synchronize after a real forward.
            values = buffer[:n].cpu().numpy()
            for position in np.flatnonzero(values[:, 0] != 0):
                if manager._tact_detail_records >= limit:
                    manager._tact_detail_dropped += 1
                    continue
                records.append(encode_detail(
                    manager, geometry, controller,
                    int(position), values[position], mode))
                manager._tact_detail_records += 1
    if not records:
        return
    directory = Path(os.environ["TACT_EASYSTEER_EVIDENCE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"token-details-{os.getpid()}.jsonl").open("a") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
