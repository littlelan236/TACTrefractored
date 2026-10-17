#!/usr/bin/env python3
"""Compare saved semantic trajectories; never infer original sampled token IDs.

Run from project root with .venvs/runner/bin/python. See --help.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(run, task):
    base = run / "tasks" / task
    result = base / "result.json"
    if not result.exists():
        return None
    attempt = load(result)["attempt_id"]
    files = sorted((base / "attempts" / attempt).glob("*.traj.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one selected trajectory: {base}/{attempt}")
    return files[0]


def fields(message):
    # Keep exact argument strings. Ignore IDs, timestamps, costs, duplicated
    # provider fields and response metadata which are not generated semantics.
    result = [("role", message.get("role", ""))]
    if message.get("role") == "assistant":
        reasoning = message.get("reasoning_content")
        if reasoning is None:
            reasoning = (message.get("provider_specific_fields") or {}).get("reasoning")
        result.append(("reasoning_content", reasoning or ""))
    result.append(("content", message.get("content") or ""))
    for i, call in enumerate(message.get("tool_calls") or []):
        for key in ("name", "arguments"):
            value = call.get("function", {}).get(key, "")
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            result.append((f"tool_calls[{i}].{key}", value))
    return result


def common_length(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def diff_config(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(a.keys() | b.keys()):
            out.extend(diff_config(a.get(key), b.get(key), f"{prefix}.{key}".strip(".")))
        return out
    return [] if a == b else [{"path": prefix, "registry": a, "easysteer": b}]


def compare(left, right, tok, dest):
    lm, rm = load(left)["messages"], load(right)["messages"]
    common = []
    turn = 0
    for index in range(max(len(lm), len(rm))):
        a = lm[index] if index < len(lm) else None
        b = rm[index] if index < len(rm) else None
        if (a or b).get("role") == "assistant":
            turn += 1
        af, bf = fields(a) if a else [], fields(b) if b else []
        if af == bf:
            common.append(dict(af))
            continue
        shared_fields = []
        for j in range(max(len(af), len(bf))):
            ak, av = af[j] if j < len(af) else ("<absent>", "")
            bk, bv = bf[j] if j < len(bf) else ("<absent>", "")
            if (ak, av) == (bk, bv):
                shared_fields.append({"field": ak, "text": av})
                continue
            # Encode each saved field independently: NOT chat-template or raw output.
            ai = tok.encode(av, add_special_tokens=False).ids
            bi = tok.encode(bv, add_special_tokens=False).ids
            n = common_length(ai, bi)
            nc = common_length(av, bv)
            offsets = tok.encode(av, add_special_tokens=False).offsets
            prefix = av[:offsets[n - 1][1]] if n else ""
            row = {
                "status": "different", "message_index_0based": index,
                "assistant_turn_1based": turn, "role": (a or b).get("role"),
                "registry_field": ak, "easysteer_field": bk,
                "same_field_name": ak == bk,
                "token_basis": "saved_field_reencoded_NOT_original_generation",
                "common_field_tokens": n,
                "first_different_field_token_1based": n + 1,
                "common_field_characters": nc,
                "registry_token_id": ai[n] if n < len(ai) else None,
                "easysteer_token_id": bi[n] if n < len(bi) else None,
                "registry_next_tokens": ai[n:n + 32],
                "easysteer_next_tokens": bi[n:n + 32],
                "registry_next_text": tok.decode(ai[n:n + 64], skip_special_tokens=False),
                "easysteer_next_text": tok.decode(bi[n:n + 64], skip_special_tokens=False),
                "common_prefix_tail": av[max(0, nc - 600):nc],
                "registry_character_context": av[max(0, nc - 200):nc + 600],
                "easysteer_character_context": bv[max(0, nc - 200):nc + 600],
                "prior_complete_common_messages": len(common),
                "first_turn_reasoning_divergence": turn == 1 and ak == bk == "reasoning_content",
                "raw_output_token_ids_available_at_divergence": [
                    bool((((m or {}).get("extra") or {}).get("response") or {})
                         .get("choices", [{}])[0].get("provider_specific_fields", {}).get("token_ids"))
                    for m in (a, b)],
            }
            dest.mkdir(parents=True)
            save(dest / "common-history.json", {
                "complete_messages": common,
                "equal_fields_in_divergent_message": shared_fields,
                "note": "Semantic fields, not original request serialization. Random tool IDs omitted."})
            (dest / "common-field-prefix.txt").write_text(av[:nc], encoding="utf-8")
            (dest / "common-token-prefix.txt").write_text(prefix, encoding="utf-8")
            save(dest / "common-field-token-ids.json", ai[:n])
            save(dest / "first-difference.json", row)
            return row
    return {"status": "identical_semantic_trajectory", "messages": len(lm)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry-run", type=Path, required=True)
    p.add_argument("--easysteer-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New directory; refuses overwrite")
    p.add_argument("--task", action="append", help="Compare only this instance ID; repeatable")
    args = p.parse_args()
    rc, ec = [load(r / "resolved-config.json") for r in (args.registry_run, args.easysteer_run)]
    for key in ("strategy", "axes", "k", "lambda", "additive_scale", "sites"):
        if rc["steering"].get(key) != ec["steering"].get(key):
            p.error(f"Steering differs at {key}")
    if rc["model_descriptor"]["metadata_hashes"]["tokenizer.json"] != ec["model_descriptor"]["metadata_hashes"]["tokenizer.json"]:
        p.error("Tokenizer identities differ")
    from tokenizers import Tokenizer
    tokenizer_path = Path(rc["model"]["path"]) / "tokenizer.json"
    if digest(tokenizer_path) != rc["model_descriptor"]["metadata_hashes"]["tokenizer.json"]:
        p.error("Tokenizer file hash does not match run descriptor")
    tok = Tokenizer.from_file(str(tokenizer_path))
    args.output.mkdir(parents=True, exist_ok=False)
    selected_ids = []
    for run in (args.registry_run, args.easysteer_run):
        selection = load(run / "selection.json")
        selected_ids.append(selection)
    tasks = sorted(set(x.name for r in (args.registry_run, args.easysteer_run)
                       for x in (r / "tasks").iterdir() if x.is_dir()))
    if args.task:
        tasks = sorted(set(args.task))
    config_keys = ("model", "generation", "steering", "serving", "agent", "templates", "execution", "timeouts", "environment", "benchmark")
    differences = diff_config({k: rc.get(k) for k in config_keys}, {k: ec.get(k) for k in config_keys})
    rows = []
    for task in tasks:
        left, right = selected(args.registry_run, task), selected(args.easysteer_run, task)
        if left is None or right is None:
            row = {"status": "missing_selected_trajectory", "registry_available": left is not None,
                   "easysteer_available": right is not None}
        else:
            row = compare(left, right, tok, args.output / task)
            row.update({"registry_trajectory": str(left), "easysteer_trajectory": str(right),
                        "registry_sha256": digest(left), "easysteer_sha256": digest(right)})
        rows.append({"instance_id": task, **row})
    save(args.output / "results.json", rows)
    columns = ["instance_id", "status", "message_index_0based", "assistant_turn_1based", "role", "registry_field", "common_field_tokens", "first_different_field_token_1based", "registry_token_id", "easysteer_token_id", "first_turn_reasoning_divergence", "common_prefix_tail", "registry_next_text", "easysteer_next_text"]
    with (args.output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "registry_run": str(args.registry_run), "easysteer_run": str(args.easysteer_run),
        "registry_seed": rc["generation"]["seed"], "easysteer_seed": ec["generation"]["seed"],
        "selection_equal": selected_ids[0] == selected_ids[1],
        "tokenizer_sha256": digest(tokenizer_path), "tasks": len(rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "first_turn_reasoning_divergences": sum(r.get("first_turn_reasoning_divergence", False) for r in rows),
        "divergent_fields": dict(Counter(r.get("registry_field") for r in rows if r["status"] == "different")),
        "configuration_differences": differences,
        "limitations": [
            "Token indices are 1-based within separately reencoded saved fields, not original sampled tokens or a concatenated trajectory.",
            "Parsed responses omit think/tool delimiters and stop tokens; exact raw generation cannot be recovered from these artifacts.",
            "Common character prefix can extend past common token prefix because BPE token boundaries depend on the suffix.",
            "First chronological semantic difference includes tool output (timestamps/paths may differ); random tool-call IDs and metadata are ignored.",
            "A first-turn reasoning difference is in saved reasoning before the first parsed think-close; this alone does not identify a numerical or sampling root cause.",
        ],
    }
    save(args.output / "manifest.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("configuration_differences", "limitations")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
