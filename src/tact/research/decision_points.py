from __future__ import annotations

import copy
import json
from pathlib import Path

from tact.artifacts import identifier, now, read_json, selected_trajectory, sha256, write_json
from tact.config import select_instances
from tact.research.trajectories import TrajectoryReader


def tools_schema() -> list[dict]:
    # Physical copy of the stock mini-swe-agent tool contract; tracked by hash.
    return [{"type": "function", "function": {
        "name": "bash", "description": "Execute a bash command",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"]}}}]


def template_messages(prefix: list[dict]) -> list[dict]:
    prepared = []
    for message in prefix:
        m = {k: copy.deepcopy(v) for k, v in message.items()
             if k in {"role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name"}}
        for call in m.get("tool_calls") or []:
            args = call["function"].get("arguments")
            if isinstance(args, str):
                call["function"]["arguments"] = json.loads(args)
        prepared.append(m)
    return prepared


def build_decisions(cfg: dict, run: Path, set_id: str) -> Path:
    from transformers import AutoTokenizer

    root = Path(cfg["project_root"])
    output = root / "artifacts/latents" / identifier(set_id)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["path"], local_files_only=True,
                                        trust_remote_code=cfg["model"]["trust_remote_code"])
    close = tok.encode("</think>", add_special_tokens=False)
    if not close or tok.decode(close) != "</think>":
        raise ValueError("Tokenizer has no exact think-close sequence")
    tasks = []
    rows, partitions = select_instances(cfg)
    membership = {i: name for name, ids in partitions.items() for i in ids}
    exclusions = set(read_json(Path(cfg["research"]["fit_exclusions"]))["instance_ids"])
    templates = {k: cfg["generation"][k] for k in ["enable_thinking", "preserve_thinking"]}
    for row in rows:
        iid = row["instance_id"]
        if membership[iid] not in {"axis-train", "axis-val"} or iid in exclusions:
            continue
        source = selected_trajectory(run, iid)
        if source is None:
            raise ValueError(f"Missing trajectory: {iid}")
        reader = TrajectoryReader(source)
        decisions, unavailable = [], []
        for step in reader.steps():
            response = step["response"]
            if not isinstance(response, dict) or not step["reasoning"].strip():
                unavailable.append({"step": step["step"], "reason": "Missing response/reasoning"})
                continue
            choice = response.get("choices", [{}])[0]
            if choice.get("finish_reason") == "length" and not step["tool_calls"] and not step["content"]:
                unavailable.append({"step": step["step"], "reason": "Incomplete reasoning"})
                continue
            request = template_messages(reader.prepared_prefix(step["message_index"]))
            prompt = tok.apply_chat_template(request, tools=tools_schema(), tokenize=True,
                add_generation_prompt=True, **templates)
            reported = (response.get("usage") or {}).get("prompt_tokens")
            if reported != len(prompt):
                raise ValueError(f"Prompt length mismatch {iid}/{step['step']}: {len(prompt)} != {reported}")
            server_ids = response.get("prompt_token_ids")
            if server_ids is not None and server_ids != prompt:
                raise ValueError(f"Prompt token ID mismatch {iid}/{step['step']}")
            prefix = prompt + tok.encode(step["reasoning"], add_special_tokens=False) + close
            decisions.append({"step": step["step"], "message_index": step["message_index"],
                              "prefix_token_ids": prefix, "decision_index": len(prefix) - 1,
                              "prompt_tokens": len(prompt), "response_id": response.get("id"),
                              "evidence": "server_token_ids" if server_ids else "template_hash_and_token_count"})
        write_json(output / f"{iid}.decisions.json", decisions)
        tasks.append({"instance_id": iid, "partition": membership[iid],
                      "source_traj": str(source.relative_to(root)), "source_sha256": sha256(source),
                      "steps": [d["step"] for d in decisions], "unavailable": unavailable,
                      "decisions_sha256": sha256(output / f"{iid}.decisions.json")})
    if not tasks:
        raise ValueError("No train/validation trajectories selected")
    write_json(output / "manifest.json", {"schema_version": 1, "latent_set": set_id,
        "run_id": run.name, "model": cfg["model_descriptor"], "template_kwargs": templates,
        "token_boundary": {"text": "</think>", "token_ids": close, "site": "close_token_as_input"},
        "created_at": now(), "tasks": tasks})
    return output
