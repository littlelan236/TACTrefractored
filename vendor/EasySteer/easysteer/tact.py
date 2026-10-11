"""Convert existing model-bound TACT axes to native single-vector specs."""

import json
import math
from pathlib import Path

import numpy as np


def from_axes(path, *, layer_count, hidden_size, selected_indices, sites, k=2.0):
    """Load explicit HF-hidden-v1 indices; never infer an NPZ layout."""
    from vllm.steer_vectors import DirectionVector, TactBandPayload

    if not math.isfinite(k) or k < 0:
        raise ValueError("k must be finite and nonnegative")
    if (
        not selected_indices
        or len(set(selected_indices)) != len(selected_indices)
        or len(sites) != len(selected_indices)
    ):
        raise ValueError("TACT requires unique selected indices and matching sites")
    layers, directions, inventory = {}, {}, {}
    with np.load(path, allow_pickle=False) as z:
        for name in ("v_ot", "v_oa_perp", "mu_cal", "sigma_ot", "sigma_oa"):
            shape = (
                (layer_count, hidden_size)
                if name in {"v_ot", "v_oa_perp", "mu_cal"}
                else (layer_count,)
            )
            if z[name].shape != shape or not np.isfinite(z[name]).all():
                raise ValueError(f"Invalid TACT array {name}")
        for index, site in zip(selected_indices, sites):
            if type(index) is not int or not 0 <= index < layer_count:
                raise ValueError("Invalid selected axis index")
            expected = (
                {"kind": "final_norm", "block_index": None}
                if index == layer_count - 1
                else {"kind": "decoder_output", "block_index": index}
            )
            if site != expected:
                raise ValueError("HF-hidden-v1 selected site/index mismatch")
            site_id = layer_count if site["kind"] == "final_norm" else site["block_index"]
            if z["sigma_ot"][index] < 0 or z["sigma_oa"][index] < 0:
                raise ValueError("TACT sigma must be nonnegative")
            layers[site_id] = {
                "v_ot": z["v_ot"][index].copy(),
                "v_oa": z["v_oa_perp"][index].copy(),
                "mu": z["mu_cal"][index].copy(),
                "tau_ot": np.float32(k * z["sigma_ot"][index]),
                "tau_oa": np.float32(k * z["sigma_oa"][index]),
            }
            directions[site_id] = layers[site_id]["v_ot"] + layers[site_id]["v_oa"]
            inventory[str(site_id)] = site
    manifest = {
        "schema_version": 1,
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "sites": inventory,
    }
    band = TactBandPayload(layers, manifest, k=k, source=Path(path).name)
    return DirectionVector(directions), band


def spec_from_bundle(directory, *, strategy, k, strength, close_id):
    """Return a native SteeringSpec; baseline is explicitly False."""
    if strategy == "none":
        return False
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    model = manifest["model"]
    if model["site_schema"] != "hf-hidden-v1":
        raise ValueError("Only explicit hf-hidden-v1 bundles are supported")
    direct, band = from_axes(
        Path(directory) / "axes.npz",
        layer_count=model["layer_count"],
        hidden_size=model["hidden_size"],
        selected_indices=manifest["selected_indices"],
        sites=manifest["selected_sites"],
        k=k,
    )
    algorithm = {"add": "tact_add", "cap": "tact_cap", "gated": "tact_gate"}[strategy]
    return SteeringSpec(
        vectors=[
            VectorSpec(
                algorithm=algorithm,
                data=direct if strategy == "add" else band,
                scale=1.0 if strategy == "cap" else strength,
                normalize=False,
                apply=ApplySpec(generation_tokens=[close_id]),
            )
        ]
    )
