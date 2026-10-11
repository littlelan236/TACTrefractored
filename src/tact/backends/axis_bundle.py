"""Model identity and axis tensors shared by serving backends."""

from pathlib import Path

from tact.artifacts import read_json, sha256
from tact.contracts import validate_axis_manifest


def validate_axis_bundle(cfg):
    desc, steering = cfg["model_descriptor"], cfg["steering"]
    bundle = None
    if steering["axes"]:
        import numpy as np

        directory = Path(steering["axes"])
        bundle = read_json(directory / "manifest.json")
        validate_axis_manifest(bundle, desc)
        if not bundle.get("complete") or sha256(directory / "axes.npz") != bundle["axes_sha256"]:
            raise ValueError("Incomplete or changed AxisBundle")
        indices = bundle["selected_indices"]
        if not isinstance(indices, list) or any(
            type(i) is not int or i < 0 or i >= desc["layer_count"] for i in indices
        ):
            raise ValueError("AxisBundle has invalid selected indices")
        if len(indices) != len(set(indices)) or bundle["selected_sites"] != [
            desc["sites"][i] for i in indices
        ]:
            raise ValueError("AxisBundle selected indices/sites disagree")
        with np.load(directory / "axes.npz", allow_pickle=False) as arrays:
            for key in ("v_ot", "v_oa_perp", "mu_cal", "sigma_ot", "sigma_oa"):
                expected = (
                    (desc["layer_count"], desc["hidden_size"])
                    if key in {"v_ot", "v_oa_perp", "mu_cal"}
                    else (desc["layer_count"],)
                )
                if arrays[key].shape != expected or not np.isfinite(arrays[key]).all():
                    raise ValueError(f"Invalid AxisBundle tensor: {key}")
            for key in ("sigma_ot", "sigma_oa"):
                if np.any(arrays[key][indices] < 0):
                    raise ValueError("AxisBundle sigma must be nonnegative")
            ot, oa = arrays["v_ot"][indices], arrays["v_oa_perp"][indices]
            if (
                not np.allclose(np.linalg.norm(ot, axis=1), 1, atol=1e-4)
                or not np.allclose(np.linalg.norm(oa, axis=1), 1, atol=1e-4)
                or not np.allclose(np.sum(ot * oa, axis=1), 0, atol=1e-4)
            ):
                raise ValueError("AxisBundle directions must be orthonormal")
    return bundle
