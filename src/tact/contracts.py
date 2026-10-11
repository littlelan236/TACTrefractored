from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tact.artifacts import hash_json, sha256


@dataclass(frozen=True)
class SiteRef:
    kind: str
    block_index: int | None = None

    def __post_init__(self):
        if self.kind not in {"decoder_output", "final_norm"}:
            raise ValueError(f"Unsupported site kind: {self.kind}")
        if self.kind == "decoder_output" and (
            type(self.block_index) is not int or self.block_index < 0
        ):
            raise ValueError("Decoder site requires a non-negative block_index")
        if self.kind == "final_norm" and self.block_index is not None:
            raise ValueError("Final norm has no block index")


def hf_sites(layer_count: int) -> list[dict]:
    # HF's last output_hidden_states entry is after the final norm.
    return [asdict(SiteRef("decoder_output", i)) for i in range(layer_count - 1)] + [
        asdict(SiteRef("final_norm"))
    ]


def model_descriptor(path: Path) -> dict:
    cfg = json.loads((path / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    layers, width = text.get("num_hidden_layers"), text.get("hidden_size")
    if type(layers) is not int or layers < 1 or type(width) is not int or width < 1:
        raise ValueError("Model config does not describe a supported text decoder")
    files = {}
    for name in ["config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json",
                 "merges.txt", "chat_template.jinja", "generation_config.json"]:
        if (path / name).is_file():
            files[name] = sha256(path / name)
    identity_path = path / "tact-identity.json"
    if not identity_path.is_file():
        raise ValueError("Import the model and create its weight identity manifest first")
    identity = json.loads(identity_path.read_text())
    for item in identity["files"]:
        relative = Path(item["destination"]).relative_to(Path("assets/models") / path.name)
        file = path / relative
        if not file.resolve().is_relative_to(path.resolve()) or file.is_symlink():
            raise ValueError(f"Model identity path escapes physical model directory: {file}")
        if not file.is_file() or file.stat().st_size != item["bytes"]:
            raise ValueError(f"Model identity size mismatch: {file}")
    return {
        "model_fingerprint": hash_json({"identity": identity, "metadata": files}),
        "architectures": cfg.get("architectures", []), "model_type": cfg.get("model_type"),
        "layer_count": layers, "hidden_size": width, "metadata_hashes": files,
        "site_schema": "hf-hidden-v1", "sites": hf_sites(layers),
        "layer_types": text.get("layer_types"),
    }


def validate_axis_manifest(manifest: dict, descriptor: dict) -> None:
    for key in ["model_fingerprint", "layer_count", "hidden_size", "site_schema"]:
        if manifest.get("model", {}).get(key) != descriptor[key]:
            raise ValueError(f"Axis bundle model mismatch: {key}")
    if not manifest.get("selected_sites"):
        raise ValueError("Axis bundle must select sites")
    for site in manifest["selected_sites"]:
        if site not in descriptor["sites"]:
            raise ValueError(f"Axis bundle site not supported by model: {site}")
