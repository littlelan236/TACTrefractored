from __future__ import annotations

import gc
from pathlib import Path

from tact.artifacts import read_json, sha256, write_json


def extract(cfg: dict, directory: Path):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    manifest = read_json(directory / "manifest.json")
    if manifest["model"]["model_fingerprint"] != cfg["model_descriptor"]["model_fingerprint"]:
        raise ValueError("Latent manifest checkpoint mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_config = AutoConfig.from_pretrained(cfg["model"]["path"], local_files_only=True)
    dtype = getattr(torch, cfg["research"]["extraction_dtype"])
    architecture = (model_config.architectures or [""])[0]
    cls = AutoModelForImageTextToText if "ConditionalGeneration" in architecture else AutoModelForCausalLM
    outer = cls.from_pretrained(cfg["model"]["path"], local_files_only=True, torch_dtype=dtype,
                               device_map="auto", trust_remote_code=cfg["model"]["trust_remote_code"])
    candidates = [outer, getattr(outer, "model", None)]
    candidates += [getattr(x, "language_model", None) for x in list(candidates) if x is not None]
    candidates += [getattr(x, "model", None) for x in list(candidates) if x is not None]
    model = next((m for m in candidates if m is not None and hasattr(m, "layers")
                  and hasattr(m, "embed_tokens")), None)
    if model is None:
        raise ValueError("No supported text decoder found; add an explicit architecture adapter")
    model.eval()
    desc = cfg["model_descriptor"]
    device = model.embed_tokens.weight.device
    chunk = cfg["research"]["extraction_chunk_size"]
    for task in manifest["tasks"]:
        iid = task["instance_id"]
        source = Path(cfg["project_root"]) / task["source_traj"]
        if sha256(source) != task["source_sha256"]:
            raise ValueError(f"Source changed: {iid}")
        path = directory / f"{iid}.decisions.json"
        if sha256(path) != task["decisions_sha256"]:
            raise ValueError(f"Decision input changed: {iid}")
        output = directory / f"{iid}.npz"
        if output.exists():
            continue
        states = []
        with torch.inference_mode():
            for decision in read_json(path):
                ids = decision["prefix_token_ids"]
                cache, result = None, None
                for start in range(0, len(ids), chunk):
                    inputs = torch.tensor([ids[start:start + chunk]], dtype=torch.long, device=device)
                    final = start + chunk >= len(ids)
                    result = model(input_ids=inputs, past_key_values=cache, use_cache=True,
                                   output_hidden_states=final)
                    cache = result.past_key_values
                hidden = np.stack([h[0, -1].float().cpu().numpy() for h in result.hidden_states[1:]])
                if hidden.shape != (desc["layer_count"], desc["hidden_size"]) or not np.isfinite(hidden).all():
                    raise ValueError(f"Invalid model hidden-state convention for {iid}")
                states.append(hidden.astype(np.float16))
                del cache, result
        if not states:
            raise ValueError(f"No decision states: {iid}")
        with output.with_suffix(".tmp").open("wb") as f:
            np.savez_compressed(f, hidden=np.stack(states), step=np.array(task["steps"], dtype=np.int32))
        output.with_suffix(".tmp").replace(output)
        write_json(directory / f"{iid}.extraction.json", {"model": desc, "dtype": str(dtype),
            "chunk_size": chunk, "tf32": False, "npz_sha256": sha256(output),
            "reconstructed_checkpoint_states": True})
        gc.collect()
        print(f"Extracted {iid}: {len(states)} decision states", flush=True)
    write_json(directory / "verification.json", {"complete": True, "model": desc,
        "files": {r["instance_id"]: sha256(directory / (r["instance_id"] + ".npz"))
                  for r in manifest["tasks"]}})
