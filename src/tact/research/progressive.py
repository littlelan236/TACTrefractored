"""Nested historical label subsets: refit axes and plot exactly the selected rows."""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from tact.artifacts import hash_json, identifier, read_json, sha256, write_json
from tact.contracts import model_descriptor
from tact.research.axes import CLASSES, fit_pair, macro_auc, save_projections
from tact.research.decision_points import template_messages, tools_schema
from tact.research.plots import plot_bundle
from tact.research.trajectories import TrajectoryReader

SPLITS = ("axis-train", "axis-val")
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
METHODS = ("tact_mean_difference", "lda_contrast")


def nested_selections(tasks: list[dict], seed: int) -> list[dict]:
    """Select whole trajectories with a seeded order and class-contrast anchors."""
    import numpy as np

    rng = np.random.default_rng(seed)
    order = {}
    for split in SPLITS:
        group = sorted((t for t in tasks if t["partition"] == split),
                       key=lambda t: t["instance_id"])
        if len(group) < 5:
            raise ValueError(f"Five increasing subsets require at least five trajectories: {split}")
        shuffled = [group[i] for i in rng.permutation(len(group))]
        anchors = []
        for label in CLASSES[:2]:
            candidate = next((t for t in shuffled if t["counts"].get(label, 0)
                              and t["counts"].get(CLASSES[2], 0)), None)
            if candidate is None:
                raise ValueError(f"No per-problem {label}/CAL contrast in {split}")
            if candidate["instance_id"] not in anchors:
                anchors.append(candidate["instance_id"])
        order[split] = anchors + [t["instance_id"] for t in shuffled
                                 if t["instance_id"] not in anchors]
        if math.ceil(len(group) * FRACTIONS[0]) < len(anchors):
            raise ValueError("First subset too small to cover both per-problem contrasts")
    by_id = {t["instance_id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Duplicate trajectory or train/val overlap")
    result = []
    previous_counts = {s: 0 for s in SPLITS}
    for index, fraction in enumerate(FRACTIONS, 1):
        selected = {s: order[s][:math.ceil(len(order[s]) * fraction)] for s in SPLITS}
        summary = {}
        for split, ids in selected.items():
            counts = Counter()
            for iid in ids:
                counts.update(by_id[iid]["counts"])
            if len(ids) <= previous_counts[split]:
                raise ValueError("Trajectory counts must strictly increase at each stage")
            previous_counts[split] = len(ids)
            summary[split] = {"trajectories": len(ids), "steps": sum(counts.values()),
                              "classes": dict(counts)}
        result.append({"stage": index, "fraction": fraction, "selected_ids": selected,
                       "counts": summary, "selection_seed": seed,
                       "selection_unit": "whole_trajectory", "class_anchor_policy": True})
    return result


def prepare(root: Path, source: Path, output_id: str, seed: int) -> Path:
    """Physically import historical labels/trajectories and validate CPU decision prefixes."""
    import json
    import shutil

    from transformers import AutoTokenizer

    output = root / "reports/pointclouds" / identifier(output_id)
    if (output / "manifest.json").exists():
        raise FileExistsError(output)
    preserved = read_json(source / "MANIFEST.json")
    records = {r["path"]: r for r in preserved["files"]}
    model = root / "assets/models/Qwen3.6-35B-A3B"
    descriptor = model_descriptor(model)
    excluded = set(read_json(root / "data/swebench/fit-exclusions.json")["instance_ids"])
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    tasks, omitted = [], []
    label_files = sorted((source / "annotation/labels").rglob("*.jsonl"))
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    close = tok.encode("</think>", add_special_tokens=False)
    if not close or tok.decode(close) != "</think>":
        raise ValueError("No exact think-close token sequence")
    for file in label_files:
        iid, split = file.stem, file.parent.name
        if split not in SPLITS:
            raise ValueError(f"Unknown historical partition: {file}")
        if iid in excluded:
            omitted.append({"instance_id": iid, "partition": split,
                            "reason": "reviewed whole-trajectory fit exclusion"})
            continue
        trajectory = source / "baseline-trajectory" / iid / f"{iid}.traj.json"
        for item in (file, trajectory):
            record = records[str(item.relative_to(source))]
            if sha256(item) != record["sha256"]:
                raise ValueError(f"Preserved source changed: {item}")
        reader = TrajectoryReader(trajectory)
        stored = reader.data["info"]["config"]["model"]
        if stored["model_name"] != "openai/Qwen3.6-35B-A3B":
            raise ValueError(f"Historical checkpoint mismatch: {iid}")
        templates = stored["model_kwargs"]["extra_body"]["chat_template_kwargs"]
        rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        if (len({r["step"] for r in rows}) != len(rows)
                or any(r["instance_id"] != iid or r["label"] not in CLASSES for r in rows)):
            raise ValueError(f"Invalid historical labels: {iid}")
        kept = sorted((r for r in rows if r["confidence"] in {"high", "medium"}
                       and r["step"] > 0), key=lambda r: r["step"])
        steps = {r["step"]: r for r in reader.steps()}
        decisions = []
        for label in kept:
            step = steps[label["step"]]
            if not step["reasoning"].strip() or not isinstance(step["response"], dict):
                raise ValueError(f"Labeled decision missing reasoning/response: {iid}/{label['step']}")
            response = step["response"]
            if (response["choices"][0].get("finish_reason") == "length"
                    and not step["content"] and not step["tool_calls"]):
                raise ValueError(f"Labeled reasoning incomplete: {iid}/{label['step']}")
            prefix = template_messages(reader.prepared_prefix(step["message_index"]))
            prompt_text = tok.apply_chat_template(prefix, tools=tools_schema(), tokenize=False,
                                                 add_generation_prompt=True, **templates)
            prompt = tok.encode(prompt_text, add_special_tokens=False)
            if len(prompt) != response["usage"]["prompt_tokens"]:
                raise ValueError(f"Prompt reconstruction mismatch: {iid}/{label['step']}")
            if response.get("prompt_token_ids") is not None and response["prompt_token_ids"] != prompt:
                raise ValueError(f"Prompt IDs mismatch: {iid}/{label['step']}")
            ids = prompt + tok.encode(step["reasoning"], add_special_tokens=False) + close
            decisions.append({"step": label["step"], "prefix_token_ids": ids,
                              "decision_index": len(ids) - 1})
        if not kept:
            raise ValueError(f"No eligible labels: {iid}")
        dest = inputs / split / iid
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(trajectory, dest / "trajectory.json")
        shutil.copyfile(file, dest / "labels.jsonl")
        write_json(dest / "decisions.json", decisions)
        tasks.append({"instance_id": iid, "partition": split,
                      "steps": [r["step"] for r in kept], "labels": [r["label"] for r in kept],
                      "counts": dict(Counter(r["label"] for r in kept)),
                      "trajectory": str((dest / "trajectory.json").relative_to(output)),
                      "trajectory_sha256": sha256(dest / "trajectory.json"),
                      "labels_path": str((dest / "labels.jsonl").relative_to(output)),
                      "labels_sha256": sha256(dest / "labels.jsonl"),
                      "decisions": str((dest / "decisions.json").relative_to(output)),
                      "decisions_sha256": sha256(dest / "decisions.json"),
                      "label_rows_total": len(rows), "dropped_label_rows": len(rows) - len(kept)})
        print(f"Prepared {split} {iid}: {len(kept)} labels", flush=True)
    stages = nested_selections(tasks, seed)
    manifest = {"schema_version": 1, "output_id": output_id, "model": descriptor,
                "model_path": str(model.relative_to(root)), "historical_only": True,
                "source_manifest_sha256": sha256(source / "MANIFEST.json"),
                "historical_partition_source": "preserved annotation labels directories",
                "policy": {"confidence": ["high", "medium"], "exclude_step_zero": True,
                           "exclusions_sha256": sha256(root / "data/swebench/fit-exclusions.json")},
                "research": {"pca_components": 100, "ridge": 0.001, "top_k_sites": 10},
                "tasks": tasks, "omitted": omitted, "stages": stages}
    write_json(output / "manifest.json", manifest)
    write_json(output / "status.json", {"stage": "prepared", "requires_hidden_extraction": True})
    return output


def extract_states(root: Path, output: Path, memory_gib: float = 28,
                   weights_gib: float = 20):
    """Extract each eligible state once; reuse only bit-identical causal prefixes."""
    import gc

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    manifest = read_json(output / "manifest.json")
    model_path = root / manifest["model_path"]
    if model_descriptor(model_path) != manifest["model"]:
        raise ValueError("Model changed since preparation")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
    cls = (AutoModelForImageTextToText if "ConditionalGeneration" in cfg.architectures[0]
           else AutoModelForCausalLM)
    max_memory, limits = {}, {}
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        cap = min(memory_gib, free / 2**30 - 6)
        weights = min(weights_gib, cap - 6)
        if weights < 1:
            raise ValueError(f"Insufficient free shared GPU memory: GPU {i}")
        torch.cuda.set_per_process_memory_fraction(cap * 2**30 / total, i)
        max_memory[i] = int(weights * 2**30)
        limits[str(i)] = {"allocator_cap_gib": cap, "weights_budget_gib": weights,
                          "free_gib_before": free / 2**30, "safety_gib": 6}
    write_json(output / "extraction-runtime.json", {
        "gpu_limits": limits, "forward_dtype": "float32", "chunk_size": 256,
        "changes_to_running_vllm": False})
    outer = cls.from_pretrained(model_path, local_files_only=True, dtype=torch.float32,
                                device_map="auto", max_memory=max_memory)
    write_json(output / "extraction-device-map.json", outer.hf_device_map)
    candidates = [outer, getattr(outer, "model", None)]
    candidates += [getattr(m, "language_model", None) for m in list(candidates) if m is not None]
    candidates += [getattr(m, "model", None) for m in list(candidates) if m is not None]
    model = next((m for m in candidates if m is not None and hasattr(m, "layers")
                  and hasattr(m, "embed_tokens")), None)
    if model is None:
        raise ValueError("Cannot identify text decoder")
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise ValueError("Extraction needs sufficient GPU memory; CPU/disk offload not enabled")
    model.eval()
    device = model.embed_tokens.weight.device
    latents = output / "latents"
    latents.mkdir(exist_ok=True)
    d = manifest["model"]
    files = {}
    cache_check = None
    for task in manifest["tasks"]:
        iid = task["instance_id"]
        file = latents / f"{iid}.npz"
        record = latents / f"{iid}.json"
        decisions_file = output / task["decisions"]
        if sha256(decisions_file) != task["decisions_sha256"]:
            raise ValueError(f"Decision input changed: {iid}")
        if file.exists() and record.exists():
            metadata = read_json(record)
            if (metadata["sha256"] != sha256(file)
                    or metadata["decisions_sha256"] != task["decisions_sha256"]):
                raise ValueError(f"Existing latent identity changed: {iid}")
            files[iid] = metadata["sha256"]
            continue
        cache, previous, hidden_rows = None, [], []
        with torch.inference_mode():
            for decision in read_json(decisions_file):
                ids = decision["prefix_token_ids"]
                start = len(previous) if previous and ids[:len(previous)] == previous else 0
                if start == 0:
                    cache = None
                for pos in range(start, len(ids), 256):
                    end = min(pos + 256, len(ids))
                    inputs = torch.tensor([ids[pos:end]], dtype=torch.long, device=device)
                    result = model(input_ids=inputs, past_key_values=cache, use_cache=True,
                                   output_hidden_states=end == len(ids))
                    cache = result.past_key_values
                hidden = np.stack([h[0, -1].float().cpu().numpy()
                                   for h in result.hidden_states[1:]])
                if hidden.shape != (d["layer_count"], d["hidden_size"]) or not np.isfinite(hidden).all():
                    raise ValueError(f"Invalid hidden states: {iid}")
                hidden_rows.append(hidden.astype(np.float16))
                if cache_check is None and start > 0:
                    fresh_cache = None
                    for pos in range(0, len(ids), 256):
                        end = min(pos + 256, len(ids))
                        inputs = torch.tensor([ids[pos:end]], dtype=torch.long, device=device)
                        fresh = model(input_ids=inputs, past_key_values=fresh_cache, use_cache=True,
                                      output_hidden_states=end == len(ids))
                        fresh_cache = fresh.past_key_values
                    independent = np.stack([h[0, -1].float().cpu().numpy()
                                            for h in fresh.hidden_states[1:]])
                    relative = np.linalg.norm(hidden - independent, axis=1) / np.maximum(
                        np.linalg.norm(independent, axis=1), 1e-12)
                    if relative.max() > 1e-3:
                        raise ValueError(f"Cached-prefix numerical validation failed: {relative.max()}")
                    cache_check = {"instance_id": iid, "step": decision["step"],
                                   "max_relative_l2": float(relative.max()),
                                   "method": "exact-prefix reuse versus independent chunked forward"}
                    write_json(output / "cache-validation.json", cache_check)
                    del fresh, fresh_cache
                previous = ids
                del result
                print(f"Extracted {iid}/{decision['step']} tokens={len(ids)} reused={start}", flush=True)
        with file.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(stream, hidden=np.stack(hidden_rows),
                                step=np.array(task["steps"], dtype=np.int32))
        file.with_suffix(".tmp").replace(file)
        files[iid] = sha256(file)
        write_json(record, {"sha256": files[iid], "decisions_sha256": task["decisions_sha256"],
                            "dtype": "float32 forward / float16 storage", "tf32": False,
                            "cache_policy": "bit-identical prefix only", "chunk_size": 256})
        del cache
        gc.collect()
        torch.cuda.empty_cache()
        write_json(output / "status.json", {"stage": "extracting", "completed_tasks": len(files),
                                             "total_tasks": len(manifest["tasks"])})
    write_json(latents / "verification.json", {"complete": True, "files": files,
                                                "model": manifest["model"]})


def fit_and_plot(output: Path):
    """Refit both methods separately for five nested populations and emit galleries."""
    import numpy as np

    manifest = read_json(output / "manifest.json")
    verification = read_json(output / "latents/verification.json")
    if not verification["complete"] or verification["model"] != manifest["model"]:
        raise ValueError("Latent set not complete or wrong model")
    populations = {s: {} for s in SPLITS}
    for task in manifest["tasks"]:
        iid = task["instance_id"]
        file = output / "latents" / f"{iid}.npz"
        if sha256(file) != verification["files"][iid]:
            raise ValueError(f"Latent changed: {iid}")
        with np.load(file, allow_pickle=False) as z:
            h = z["hidden"]
            if (z["step"].tolist() != task["steps"]
                    or h.shape != (len(task["steps"]), manifest["model"]["layer_count"],
                                   manifest["model"]["hidden_size"]) or not np.isfinite(h).all()):
                raise ValueError(f"Latent shape/step mismatch: {iid}")
            populations[task["partition"]][iid] = {"hidden": h, "steps": np.array(task["steps"]),
                                                    "labels": np.array(task["labels"])}
    for stage in manifest["stages"]:
        stage_dir = output / f"stage-{stage['stage']:02d}"
        stage_dir.mkdir(exist_ok=True)
        parts = {s: {iid: populations[s][iid] for iid in stage["selected_ids"][s]} for s in SPLITS}
        audit = [{"instance_id": iid, "partition": split, "kept_labels": len(data["labels"])}
                 for split, problems in parts.items() for iid, data in sorted(problems.items())]
        label_identity = hash_json({"selected": stage["selected_ids"], "tasks": [
            {k: t[k] for k in ("instance_id", "labels_sha256", "steps", "labels")}
            for t in manifest["tasks"] if t["instance_id"] in {r['instance_id'] for r in audit}]})
        paths = []
        for method in METHODS:
            bundle = stage_dir / method
            bundle.mkdir(exist_ok=True)
            marker = bundle / "manifest.json"
            if marker.exists():
                saved = read_json(marker)
                if (saved["label_manifest_sha256"] != label_identity
                        or saved["latent_manifest_sha256"] != sha256(output / "latents/verification.json")):
                    raise ValueError(f"Existing stage identity differs: {bundle}")
                paths.append(bundle)
                continue
            desc = manifest["model"]
            arrays = {k: np.zeros((desc["layer_count"], desc["hidden_size"]), dtype=np.float32)
                      for k in ("v_ot", "v_oa_perp", "mu_cal")}
            arrays.update({k: np.zeros(desc["layer_count"], dtype=np.float32)
                           for k in ("sigma_ot", "sigma_oa")})
            research = {**manifest["research"], "fit_method": method}
            curves = {}
            for site in range(desc["layer_count"]):
                ot, oa, center, sigmas = fit_pair(parts["axis-train"], site, research)
                for key, value in zip(("v_ot", "v_oa_perp", "mu_cal", "sigma_ot", "sigma_oa"),
                                      (ot, oa, center, *sigmas)):
                    arrays[key][site] = value
                curves[str(site)] = {s: {axis: macro_auc(problems, arrays[key][site],
                    arrays["mu_cal"][site], label, site)
                    for axis, key, label in (("ot", "v_ot", CLASSES[0]),
                                              ("oa", "v_oa_perp", CLASSES[1]))}
                    for s, problems in parts.items()}
                print(f"Stage {stage['stage']} {method} site {site} fitted", flush=True)
            scores = [(int(i), c["axis-val"]["ot"]["mean_auc"]) for i, c in curves.items()]
            selected = [i for i, _ in sorted(scores, key=lambda p: (-p[1], p[0]))[
                :manifest["research"]["top_k_sites"]]]
            arrays["selected_indices"] = np.array(selected, dtype=np.int32)
            np.savez_compressed(bundle / "axes.npz", **arrays)
            metadata = {"schema_version": 1, "bundle_id": f"stage-{stage['stage']:02d}-{method}",
                        "model": desc, "fit_method": method, "research_config": research,
                        "historical_only": True, "selection": stage, "input_audit": audit,
                        "label_manifest_sha256": label_identity,
                        "latent_manifest_sha256": sha256(output / "latents/verification.json"),
                        "curves": curves, "selected_indices": selected,
                        "selected_sites": [desc["sites"][i] for i in selected],
                        "axes_sha256": sha256(bundle / "axes.npz"), "complete": True}
            metadata["projection_files"] = save_projections(bundle, parts, arrays)
            write_json(marker, metadata)
            paths.append(bundle)
        plot_bundle(paths[0], paths[1:])
        write_json(stage_dir / "selection.json", stage)
        write_json(output / "status.json", {"stage": "plotting", "completed_stages": stage["stage"],
                                            "total_stages": 5})
    lines = ["# Five nested historical TACT/LDA point-cloud populations", "",
             "Historical Qwen3.6-35B-A3B; train-only fitting, validation-only peak-site selection.",
             "Whole trajectories; nested 20/40/60/80/100% subsets with class-contrast anchors.",
             "Full population means all historical nonexcluded high/medium labels except step zero.",
             "No current dataset repartition and no eval labels used.", "",
             "| Stage | Fraction | Train trajectories | Train steps | Val trajectories | Val steps |",
             "|---|---:|---:|---:|---:|---:|"]
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Five TACT/LDA subsets</title>',
            '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto}img{width:100%}</style>',
            '<h1>Five nested historical TACT / LDA populations</h1>',
            '<p>Refitted on each selected training subset; only selected points plotted. ',
            'OT red, OA green, CAL purple. <a href="REPORT.md">Report</a></p>']
    for stage in manifest["stages"]:
        counts = stage["counts"]
        lines.append(f"| {stage['stage']} | {stage['fraction']:.0%} | "
                     f"{counts['axis-train']['trajectories']} | {counts['axis-train']['steps']} | "
                     f"{counts['axis-val']['trajectories']} | {counts['axis-val']['steps']} |")
        gallery = f"stage-{stage['stage']:02d}/tact_mean_difference/visualizations"
        page.append(f'<h2>Stage {stage["stage"]} — {stage["fraction"]:.0%}</h2>'
                    f'<a href="{gallery}/index.html">Full TACT/LDA gallery and exports</a>'
                    f'<img src="{gallery}/validation-method-comparison.png" alt="Stage comparison">')
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    (output / "index.html").write_text("".join(page) + "</html>\n")
    write_json(output / "status.json", {"stage": "complete", "complete": True,
                                        "completed_stages": 5, "methods": list(METHODS)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "extract", "plot"))
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--memory-gib", type=float, default=28,
                        help="Per-GPU allocator cap for this extraction process only")
    parser.add_argument("--weights-gib", type=float, default=20,
                        help="Per-GPU model placement budget within the allocator cap")
    args = parser.parse_args()
    from tact.config import root_path
    root = root_path()
    output = root / "reports/pointclouds" / identifier(args.output_id)
    if args.action == "prepare":
        if not args.source:
            parser.error("prepare requires --source")
        print(prepare(root, args.source.resolve(), args.output_id, args.seed))
    elif args.action == "extract":
        extract_states(root, output, args.memory_gib, args.weights_gib)
    else:
        fit_and_plot(output)


if __name__ == "__main__":
    main()
