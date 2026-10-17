"""Remote baseline preparation, two axis fits, and parallel TACT eval arms."""
from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tact.artifacts import hash_json, identifier, now, read_json, sha256, write_json
from tact.config import Experiment, resolve_config


def settings():
    cfg = read_json(ROOT / "baseline/config/resolved-config.json")
    parts = read_json(ROOT / "baseline/config/splits.json")["partitions"]
    cfg = copy.deepcopy(cfg)
    cfg["project_root"] = str(ROOT)
    cfg["benchmark"]["splits"] = str(ROOT / "baseline/config/splits.json")
    cfg["research"]["fit_exclusions"] = str(ROOT / "baseline/config/fit-exclusions-workflow.json")
    # All 191 annotated tasks participate. Exclude step zero only at fit time.
    cfg["selected_ids"] = parts["axis-train"] + parts["axis-val"]
    return cfg, parts


def preflight():
    cfg, parts = settings()
    from tact.contracts import model_descriptor
    if model_descriptor(Path(cfg["model"]["path"])) != cfg["model_descriptor"]:
        raise ValueError("Baseline checkpoint metadata changed")
    if len(parts["axis-train"]) != 127 or len(parts["axis-val"]) != 64:
        raise ValueError("Unexpected baseline partition sizes")
    if set(parts["axis-train"]) & set(parts["axis-val"]) or set(cfg["selected_ids"]) & set(parts["eval"]):
        raise ValueError("Partition overlap")
    counts = {}
    for iid in cfg["selected_ids"]:
        traj = ROOT / "baseline/traj" / (identifier(iid) + ".traj.json")
        ann = ROOT / "baseline/annotation" / (iid + ".jsonl")
        rows = [json.loads(x) for x in ann.read_text().splitlines() if x.strip()]
        n = sum(m.get("role") == "assistant" for m in read_json(traj)["messages"])
        if not rows or [r["step"] for r in rows] != list(range(len(rows))) or len(rows) > n:
            raise ValueError(f"Invalid annotated prefix: {iid}")
        digest = sha256(traj)
        for r in rows:
            if r["instance_id"] != iid or r["source_sha256"] != digest:
                raise ValueError(f"Changed trajectory: {iid}")
            if r["label"] not in {"OVERTHINKING", "OVERACTING", "CALIBRATED"} or r["confidence"] not in {"high", "medium", "low"}:
                raise ValueError(f"Invalid label: {iid}")
        counts[iid] = len(rows)
    print(f"Preflight: {len(counts)} annotated tasks, {sum(counts.values())} labels, {len(parts['eval'])} eval tasks", flush=True)
    return cfg, parts


def enrich(directory: Path):
    """Reconstruct exact request prefixes; reject mismatches rather than invent usage."""
    from transformers import AutoTokenizer
    from tact.research.decision_points import template_messages, tools_schema
    from tact.research.trajectories import TrajectoryReader

    serving_python = ROOT / ".venvs/serving/bin/python"
    # Re-execute this stage in the serving environment so tokenizer behavior exactly matches rollout.
    if os.environ.get("TACT_WORKFLOW_TOKENIZER_ENV") != "serving":
        env = dict(os.environ); env["TACT_WORKFLOW_TOKENIZER_ENV"] = "serving"
        command = [str(serving_python), str(Path(__file__).resolve()), "--stage", "enrichment", "--experiment-id", os.environ.get("TACT_WORKFLOW_EXPERIMENT_ID", "tact-easysteer-eval-20261013")]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return

    cfg, parts = preflight()
    # A clean enrichment run is explicitly requested: no prior decisions or manifests are reused.
    if directory.exists():
        import shutil
        shutil.rmtree(directory)
    labels_dir = ROOT / "baseline/prepared-labels"
    if labels_dir.exists():
        import shutil
        shutil.rmtree(labels_dir)
    for stale in [ROOT / "baseline/enrichment/completed.json", ROOT / "baseline/config/workflow-research.json"]:
        stale.unlink(missing_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    write_json(Path(cfg["research"]["fit_exclusions"]), {"schema_version": 1, "instance_ids": [], "reason": "Use all 191 baseline annotation prefixes"})
    tok = AutoTokenizer.from_pretrained(cfg["model"]["path"], local_files_only=True, trust_remote_code=False)
    close = tok.encode("</think>", add_special_tokens=False)
    if len(close) != 1:
        raise ValueError("EasySteer needs a single-token think close")
    templates = {k: cfg["generation"][k] for k in ["enable_thinking", "preserve_thinking"]}
    membership = {iid: part for part, ids in parts.items() for iid in ids}
    tasks = []
    source_run = read_json(ROOT / "baseline/manifest.json")["run_id"]
    for iid in cfg["selected_ids"]:
        source = ROOT / "baseline/traj" / (iid + ".traj.json")
        annotation = ROOT / "baseline/annotation" / (iid + ".jsonl")
        rows = [json.loads(x) for x in annotation.read_text().splitlines() if x.strip()]
        # No relabeling or extension into the unannotated loop tail.
        reader = TrajectoryReader(source)
        decisions, unavailable, fit_rows = [], [], []
        rows_by_step = {r["step"]: r for r in rows}
        path = directory / (iid + ".decisions.json")
        for step in reader.steps():
            index = step["step"]
            if index not in rows_by_step:
                continue
            row = rows_by_step[index]
            response = step["response"]
            if not isinstance(response, dict) or not step["reasoning"].strip():
                unavailable.append({"step": index, "reason": "No recoverable reasoning/response"})
                continue
            choice = response.get("choices", [{}])[0]
            if choice.get("finish_reason") == "length" and not step["tool_calls"] and not step["content"]:
                unavailable.append({"step": index, "reason": "Unfinished reasoning has no think boundary"})
                continue
            prompt = tok.apply_chat_template(template_messages(reader.prepared_prefix(step["message_index"])), tools=tools_schema(), tokenize=True, return_dict=False, add_generation_prompt=True, **templates)
            reported = (response.get("usage") or {}).get("prompt_tokens")
            server_ids = response.get("prompt_token_ids")
            if reported != len(prompt) or (server_ids is not None and server_ids != prompt):
                raise ValueError(f"Request token reconstruction mismatch {iid}/{index}: reconstructed={len(prompt)} reported={reported}")
            ids = prompt + tok.encode(step["reasoning"], add_special_tokens=False) + close
            decisions.append({"step": index, "message_index": step["message_index"], "prefix_token_ids": ids, "decision_index": len(ids)-1, "prompt_tokens": len(prompt), "response_id": response.get("id"), "evidence": "server_token_ids" if server_ids is not None else "template_hash_and_token_count"})
            if index > 0 and row["confidence"] in cfg["research"]["label_confidence"]:
                fit_rows.append(row)

        if not decisions:
            raise ValueError(f"No usable labeled states: {iid}")
        path = directory / (iid + ".decisions.json")
        write_json(path, decisions)
        label_path = labels_dir / "labels" / (iid + ".jsonl")
        label_path.parent.mkdir(exist_ok=True)
        label_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in fit_rows))
        task = {"instance_id": iid, "partition": membership[iid], "source_traj": str(source.relative_to(ROOT)), "source_sha256": sha256(source), "annotation_source": str(annotation.relative_to(ROOT)), "annotation_sha256": sha256(annotation), "annotation_steps": [r["step"] for r in fit_rows], "steps": [d["step"] for d in decisions], "unavailable": unavailable, "decisions_sha256": sha256(path), "decision_reused": False, "tokenizer_class": type(tok).__name__}
        tasks.append(task)
        print(f"Enriched {iid}: {len(decisions)} states; unavailable={len(unavailable)}", flush=True)
    manifest = {"schema_version": 1, "latent_set": directory.name, "run_id": source_run, "model": cfg["model_descriptor"], "template_kwargs": templates, "token_boundary": {"text": "</think>", "token_ids": close, "site": "close_token_as_input"}, "created_at": now(), "tokenizer_class": type(tok).__name__, "reused_tasks": 0, "tasks": tasks}
    write_json(directory / "manifest.json", manifest)
    write_json(labels_dir / "manifest.json", {"schema_version": 1, "run_id": source_run, "model": cfg["model_descriptor"], "tasks": tasks, "policy": "annotated prefix, step > 0, high/medium, recoverable think boundary"})
    cfg["input_hashes"]["splits"] = sha256(ROOT / "baseline/config/splits.json")
    cfg["config_hash"] = hash_json({k: v for k, v in cfg.items() if k != "config_hash"})
    write_json(ROOT / "baseline/config/workflow-research.json", cfg)
    write_json(ROOT / "baseline/enrichment/completed.json", {"complete": True, "tasks": len(tasks), "states": sum(len(t["steps"]) for t in tasks), "unavailable": [dict(instance_id=t["instance_id"], **x) for t in tasks for x in t["unavailable"]]})


def make_arm(cfg, strategy, gpus, port, bundle, experiment):
    allowed = Experiment.model_fields
    raw = {k: copy.deepcopy(v) for k, v in cfg.items() if k in allowed}
    raw["benchmark"]["selection"] = "eval"
    raw["serving"].update(backend="easysteer", management="managed", gpus=gpus, tensor_parallel_size=4, port=port, max_num_seqs=12, gpu_memory_utilization=0.5, enable_prefix_caching=True, async_scheduling=False, graph_requirement="auto", easysteer_diagnostics=False, easysteer_detail_log=False, steering_consumer_check=False)
    raw["execution"].update(rollout_workers=12, harness_workers=12, coordination_group=None, staged_partitions=False)
    raw["steering"].update(strategy=strategy, axes=str(bundle), k=2.0, **{"lambda": 0.5}, trigger="all_generated_think_close")
    config = experiment / (strategy + ".json")
    write_json(config, raw)
    return config


def steer(experiment_id):
    cfg = read_json(ROOT / "baseline/config/workflow-research.json")
    experiment = ROOT / "experiments" / experiment_id
    experiment.mkdir(parents=True, exist_ok=True)
    bundle = ROOT / "baseline/axes/tact"
    handles, procs, records = [], [], []
    try:
        for strategy, gpus, port in [("cap", [0,1,2,3], 8130), ("gated", [4,5,6,7], 8131)]:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"Port occupied: {port}")
            config = make_arm(cfg, strategy, gpus, port, bundle, experiment)
            # Resolve now so model/bundle/capability failures prevent rollout.
            resolved = resolve_config(config)
            if resolved["model_descriptor"] != cfg["model_descriptor"]:
                raise ValueError("Steering checkpoint differs from extraction checkpoint")
            run_id = identifier(experiment_id + "-" + strategy)
            command = [str(ROOT / ".venvs/runner/bin/python"), "-m", "tact", "--config", str(config), "run", "--run-id", run_id, "--no-evaluate"]
            log = (experiment / (strategy + ".log")).open("a")
            handles.append(log)
            p = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            procs.append(p)
            records.append({"strategy": strategy, "run_id": run_id, "pid": p.pid, "gpus": gpus, "port": port, "config": str(config), "log": str(experiment / (strategy + ".log"))})
        write_json(experiment / "controller.json", {"pid": os.getpid(), "started_at": now(), "arms": records})
        codes = [p.wait() for p in procs]
        for record, code in zip(records, codes):
            record["rollout_exit_code"] = code
        # Score both arms, including missing patches, even if one rollout exits nonzero.
        eval_procs = []
        for record, log in zip(records, handles):
            command = [str(ROOT / ".venvs/runner/bin/python"), "-m", "tact", "evaluate", "--run-id", record["run_id"], "--eval-id", "candidate"]
            eval_procs.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        for record, p, log in zip(records, eval_procs, handles):
            record["harness_exit_code"] = p.wait()
            subprocess.run([str(ROOT / ".venvs/runner/bin/python"), "-m", "tact", "report", "--run-id", record["run_id"]], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(experiment / "completed.json", {"finished_at": now(), "arms": records})
        if any(r["rollout_exit_code"] or r["harness_exit_code"] for r in records):
            raise RuntimeError("One or more steering/evaluation stages failed; see individual run logs")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "enrichment", "hidden", "axes", "steering", "all"], default="all")
    parser.add_argument("--experiment-id", default="tact-easysteer-eval-20261013")
    args = parser.parse_args()
    identifier(args.experiment_id)
    status = ROOT / "baseline/workflow/status.json"
    directory = ROOT / "baseline/hidden"
    plan = ["enrichment", "hidden", "axes", "steering"] if args.stage == "all" else [args.stage]
    try:
        for stage in plan:
            if stage == "steering":
                status = ROOT / "experiments" / args.experiment_id / "status.json"
            print(f"Starting stage: {stage}", flush=True)
            write_json(status, {"stage": stage, "started_at": now(), "pid": os.getpid(), "experiment_id": args.experiment_id})
            if args.stage == "all":
                # Release the extraction process and its CUDA allocator before serving.
                stage_python = str(ROOT / ".venvs/runner/bin/python") if stage == "steering" else sys.executable
                command = [stage_python, str(Path(__file__).resolve()), "--stage", stage, "--experiment-id", args.experiment_id]
                subprocess.run(command, cwd=ROOT, check=True)
                continue
            if stage == "preflight":
                preflight()
            elif stage == "enrichment":
                enrich(directory)
            elif stage == "hidden":
                from tact.research.extract import extract
                cfg = read_json(ROOT / "baseline/config/workflow-research.json")
                extract(cfg, directory)
            elif stage == "axes":
                from tact.research.axes import fit_axes
                cfg = read_json(ROOT / "baseline/config/workflow-research.json")
                for method, name in [("tact_mean_difference", "tact"), ("lda_contrast", "lda")]:
                    current = copy.deepcopy(cfg)
                    current["research"]["fit_method"] = method
                    fit_axes(current, directory, ROOT / "baseline/prepared-labels", name, output_directory=ROOT / "baseline/axes" / name)
            else:
                steer(args.experiment_id)
        write_json(status, {"stage": "complete", "finished_at": now(), "experiment_id": args.experiment_id})
    except BaseException as exc:
        write_json(status, {"stage": "failed", "failed_stage": stage, "error": repr(exc), "at": now()})
        raise


if __name__ == "__main__":
    main()
