# SPDX-License-Identifier: Apache-2.0
"""Write bounded request/token identity for actual consumer tensor comparisons."""

import json
import os
from pathlib import Path

import numpy as np


def write_consumer_checks(manager):
    geometry = manager._consumer_geometry
    if geometry is None:
        return
    limit = int(os.environ.get("TACT_CONSUMER_LOG_LIMIT", "20000"))
    records = []
    for controller in manager._controller_manager.controllers.values():
        probe = getattr(controller, "consumer_probe", None)
        if probe is None:
            continue
        if int(probe.overflow.cpu().item()):
            raise RuntimeError("Consumer check row capacity exceeded")
        values = probe.stats.cpu().numpy()
        for flat in np.flatnonzero(values[:, 0]):
            if manager._consumer_records >= limit:
                manager._consumer_dropped += 1
                continue
            owner = int(np.searchsorted(
                geometry.query_start_loc_cpu[1:], flat, side="right"))
            if owner >= len(geometry.req_ids):
                raise RuntimeError("Consumer check included padding")
            value = values[flat]
            records.append({
                "type": "consumer_check", "pid": os.getpid(),
                "request_id": geometry.req_ids[owner],
                "token_id": int(geometry.token_ids_cpu()[flat]),
                "absolute_position": int(geometry.num_computed[owner]) +
                    int(flat) - int(geometry.query_start_loc_cpu[owner]),
                "flat_position": int(flat), "site_id": controller.layer_id,
                "site_kind": controller.site_kind, "module": controller._op_key,
                "consumer": (
                    "logits_processor_input" if controller.site_kind == "final_norm"
                    else "next_input_layernorm"
                ),
                "mode": manager._consumer_mode,
                "producer_count": float(value[0]), "consumer_count": float(value[1]),
                "hidden_max_abs_diff": float(value[2]),
                "residual_max_abs_diff": float(value[3]),
                "hidden_mismatched_elements": int(value[4]),
                "residual_mismatched_elements": int(value[5]),
                "actual_delta_norm": float(value[6]),
                "hidden_size": probe.expected_hidden.shape[1],
                "all_elements_equal": bool(value[1] == 1 and value[4] == value[5] == 0),
            })
            manager._consumer_records += 1
    if manager._consumer_dropped:
        raise RuntimeError("Consumer log limit reached; verification incomplete")
    if not records:
        return
    directory = Path(os.environ["TACT_EASYSTEER_EVIDENCE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"consumer-checks-{os.getpid()}.jsonl").open("a") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
