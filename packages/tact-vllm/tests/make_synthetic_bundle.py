"""Create explicitly synthetic, checkpoint-bound mechanism fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tact.artifacts import sha256, write_json
from tact.config import resolve_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="assets/models/Qwen3.6-35B-A3B")
    parser.add_argument("--bundle-id", default="registry-synthetic-mechanism")
    args = parser.parse_args()
    cfg = resolve_config(overrides=["model.path=" + args.model_path])
    desc = cfg["model_descriptor"]
    layers, width = desc["layer_count"], desc["hidden_size"]
    directory = Path(cfg["project_root"]) / "artifacts/axes" / args.bundle_id
    if directory.exists():
        raise FileExistsError("Use a new fixture bundle ID; existing artifacts are immutable")
    directory.mkdir(parents=True)
    ot = np.zeros((layers, width), dtype=np.float32)
    oa = ot.copy()
    ot[:, 0], oa[:, 1] = 1, 1
    selected = [layers - 2, layers - 1]
    np.savez_compressed(
        directory / "axes.npz",
        v_ot=ot,
        v_oa_perp=oa,
        mu_cal=np.zeros_like(ot),
        sigma_ot=np.ones(layers, dtype=np.float32),
        sigma_oa=np.ones(layers, dtype=np.float32),
        selected_indices=np.asarray(selected, dtype=np.int32),
    )
    write_json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "bundle_id": args.bundle_id,
            "model": desc,
            "selected_indices": selected,
            "selected_sites": [desc["sites"][i] for i in selected],
            "complete": True,
            "synthetic": True,
            "purpose": "Mechanism fixture only; not fitted TACT axes",
            "axes_sha256": sha256(directory / "axes.npz"),
        },
    )
    print(directory)


if __name__ == "__main__":
    main()
