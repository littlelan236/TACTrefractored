from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_manifest():
    path = Path(os.environ["TACT_REGISTRY_MANIFEST"])
    manifest = json.loads(path.read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported TACT runtime manifest")
    return manifest


class TactRuntimeBuffers(nn.Module):
    def __init__(self, manifest: dict, *, device=None):
        super().__init__()
        self.manifest = manifest
        self.sites = manifest["sites"]
        self.decoder_sites = {
            s["block_index"]: i for i, s in enumerate(self.sites) if s["kind"] == "decoder_output"
        }
        self.final_site = next(
            (i for i, s in enumerate(self.sites) if s["kind"] == "final_norm"), None
        )
        t, r, d = manifest["token_capacity"], manifest["request_capacity"], manifest["hidden_size"]
        self.token_capacity, self.request_capacity = t, r
        n_sites = len(self.sites)
        axes = np.zeros((n_sites, 2, d), np.float32)
        center = np.zeros((n_sites, d), np.float32)
        sigma = np.zeros((n_sites, 2), np.float32)
        if manifest["axes_path"]:
            path = Path(manifest["axes_path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["axes_sha256"]:
                raise ValueError("TACT axes content changed")
            with np.load(path, allow_pickle=False) as z:
                indices = manifest["selected_indices"]
                axes[:, 0] = z["v_ot"][indices]
                axes[:, 1] = z["v_oa_perp"][indices]
                center[:] = z["mu_cal"][indices]
                sigma[:, 0] = z["sigma_ot"][indices]
                sigma[:, 1] = z["sigma_oa"][indices]
        for name, value in (("axes", axes), ("center", center), ("sigma", sigma)):
            self.register_buffer(name, torch.tensor(value, device=device), persistent=False)
        for name, shape, dtype in (
            ("token_ids", (t,), torch.int32),
            ("eligible", (t,), torch.bool),
            ("decision_mask", (t,), torch.bool),
            ("request_rows", (t,), torch.int32),
            ("enabled", (r + 1,), torch.bool),
            ("k", (r + 1,), torch.float32),
            ("beta", (r + 1,), torch.float32),
            ("additive", (r + 1,), torch.float32),
            ("site_enabled", (r + 1, n_sites), torch.bool),
            ("evidence", (r + 1, n_sites, 7), torch.float32),
            ("counts", (r + 1, n_sites), torch.int32),
        ):
            self.register_buffer(
                name, torch.zeros(shape, device=device, dtype=dtype), persistent=False
            )

    def disable(self):
        self.eligible.zero_()
        self.decision_mask.zero_()
        self.request_rows.zero_()

    def reset_slot(self, slot):
        self.enabled[slot] = False
        self.site_enabled[slot].zero_()
        self.evidence[slot].zero_()
        self.counts[slot].zero_()

    def set_plan(self, slot, plan):
        self.reset_slot(slot)
        self.enabled[slot] = plan.enabled
        self.k[slot] = plan.k
        self.beta[slot] = plan.beta
        self.additive[slot] = plan.additive_scale
        for site in plan.sites:
            self.site_enabled[slot, site] = True

    def fill_decision_mask(self):
        self.decision_mask.copy_(
            self.eligible
            & (self.token_ids == self.manifest["close_id"])
            & self.enabled[self.request_rows.long()]
        )
