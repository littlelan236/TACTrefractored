"""Real native-server mechanism check with forced close tokens and synthetic axes."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import requests
from tact.artifacts import write_json
from tact.backends import process_manager
from tact.backends.registry_vllm import request_fields
from tact.config import resolve_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eager", "auto"], required=True)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--model-path", default="assets/models/Qwen3.6-35B-A3B")
    parser.add_argument("--axes", default="artifacts/axes/registry-synthetic-mechanism")
    parser.add_argument("--gpus", default="[4,5,6,7]")
    parser.add_argument("--tp", type=int, default=4)
    args = parser.parse_args()
    cfg = resolve_config(overrides=[
        "serving.backend=registry_vllm", "steering.axes=" + args.axes,
        "model.path=" + args.model_path,
        "serving.gpus=" + args.gpus, "serving.tensor_parallel_size=" + str(args.tp), "serving.port=18131",
        "serving.max_num_seqs=2", "serving.max_num_batched_tokens=256",
        "serving.gpu_memory_utilization=0.65", "model.max_model_len=4096",
        "generation.max_tokens=16", "serving.startup_seconds=1200",
        "serving.graph_requirement=" + args.mode,
    ])
    args.directory.mkdir(parents=True, exist_ok=True)
    write_json(args.directory / "resolved-config.json", cfg)
    session = requests.Session()
    session.trust_env = False
    outputs = []
    with process_manager(cfg, args.directory / "serving") as manager:
        runtime = json.loads((args.directory / "serving/registry-runtime.json").read_text())
        close = runtime["close_id"]
        url = "http://127.0.0.1:18131/v1/completions"

        def request(strategy, k=0, lam=0.5, additive=1.0):
            plan = request_fields(cfg)
            plan.update(tact_strategy=strategy, tact_k=k, tact_lambda=lam,
                        tact_additive_scale=additive)
            payload = {"model": "tact-model", "prompt": [248068, close, 248068],
                       "max_tokens": 4, "temperature": 0, "ignore_eos": True,
                       "allowed_token_ids": [close], "vllm_xargs": plan}
            response = session.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            outputs.append({"strategy": strategy, "k": k, "lambda": lam,
                            "id": result["id"], "usage": result["usage"]})
            print("HTTP OK", strategy, k, result["id"], flush=True)
            return result

        for strategy, k in [("none", 0), ("cap", 0), ("cap", 2), ("gated", 0),
                            ("add", 0), ("none", 0)]:
            request(strategy, k)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(request, ["cap", "none"]))
        request("none")  # drain previous finished-request accounting
        time.sleep(1)
        manager.validate_run_evidence()
        records = [json.loads(line) for path in manager.evidence_directory.glob("*.jsonl")
                   for line in path.read_text().splitlines()]
        finished = [r for r in records if r["type"] == "finished"]
        for strategy in ("cap", "gated", "add"):
            items = [r for r in finished if r["strategy"] == strategy]
            if not items or any(r["first_close_position"] != 3 or r["site_counts"] != [1, 1]
                                for r in items):
                raise AssertionError(f"Incorrect first-close/site execution: {strategy} {items}")
        for r in finished:
            if r["strategy"] == "none" and r["site_counts"] != [0, 0]:
                raise AssertionError("NONE changed activation")
        replay = [r for r in records if r["type"] == "dispatch" and r["full_replay"]]
        if args.mode == "auto" and not replay:
            raise AssertionError("No actual FULL replay occurred in native model dispatch")
        decisions = [r for r in records if r["type"] == "decision"]
        for r in decisions:
            if r["strategy"] == "none":
                continue
            for evidence in r["evidence_sums"]:
                before, threshold, after = evidence[:2], evidence[2:4], evidence[4:6]
                for p, tau, result in zip(before, threshold, after):
                    expected = (p - r["beta"] * (p - max(-tau, min(tau, p)))
                                + r["additive_scale"])
                    if abs(result - expected) > 0.03 + 0.01 * abs(expected):
                        raise AssertionError(f"Incorrect real model projection: {r}")
        write_json(args.directory / "smoke.json", {"complete": True, "mode": args.mode,
                   "synthetic": True, "outputs": outputs, "finished_records": finished,
                   "decision_records": decisions, "full_replay_dispatches": len(replay)})
    print("SERVER MECHANISM SMOKE PASSED", args.mode, flush=True)


if __name__ == "__main__":
    main()
