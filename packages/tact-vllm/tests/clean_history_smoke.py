"""Real HTTP prefix-cache test and fixed-token KV/conv/SSM reference comparison."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import requests

from tact.artifacts import write_json
from tact.backends.registry_vllm import RegistryProcessManager, request_fields
from tact.config import resolve_config


class ProbeManager(RegistryProcessManager):
    def launch_spec(self):
        args, env = super().launch_spec()
        args[args.index("--worker-cls") + 1] = "cache_probe_worker.CacheProbeWorker"
        env["PYTHONPATH"] = str(Path(self.cfg["project_root"]) / "packages/tact-vllm/tests")
        env["TACT_CACHE_PROBE_DIR"] = str(self.directory.parent / "snapshots")
        return args, env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--mode", choices=["eager", "auto"], default="eager")
    parser.add_argument("--axes", required=True)
    args = parser.parse_args()
    args.directory = args.directory.resolve()
    args.directory.mkdir(parents=True, exist_ok=False)
    cfg = resolve_config(overrides=[
        "serving.backend=registry_vllm", "serving.enable_prefix_caching=true",
        "model.path=assets/models/Qwen3.5-9B", "steering.axes=" + args.axes,
        "serving.gpus=[0]", "serving.tensor_parallel_size=1", "serving.port=18139",
        "serving.max_num_seqs=2", "serving.max_num_batched_tokens=512",
        "serving.gpu_memory_utilization=0.45", "model.max_model_len=4096",
        "generation.max_tokens=16", "serving.startup_seconds=1200",
        "serving.graph_requirement=" + args.mode,
    ])
    write_json(args.directory / "resolved-config.json", cfg)
    session = requests.Session()
    session.trust_env = False
    outputs = []
    with ProbeManager(cfg, args.directory / "serving") as manager:
        runtime = json.loads((manager.directory / "registry-runtime.json").read_text())
        close = runtime["close_id"]

        def request(prompt, strategy="none", label=None, salt="primary", length=1, skip=False):
            plan = request_fields(cfg)
            plan.update(tact_strategy=strategy, tact_k=0, tact_lambda=0.5, tact_additive_scale=5)
            if label:
                plan["cache_probe_label"] = label
            payload = {"model": cfg["model"]["served_name"], "prompt": prompt,
                       "max_tokens": length, "temperature": 0, "ignore_eos": True,
                       "allowed_token_ids": [close], "vllm_xargs": plan,
                       "cache_salt": salt, "skip_reading_prefix_cache": skip}
            response = session.post("http://127.0.0.1:18139/v1/completions", json=payload,
                                    timeout=240)
            response.raise_for_status()
            result = response.json()
            outputs.append({"strategy": strategy, "label": label,
                            "id": result["id"], "usage": result["usage"]})
            print("HTTP OK", strategy, label, result["usage"], flush=True)
            return result

        # The synthetic bundle selects an early decoder site AND final norm.
        # Forced output IDs make the next-turn history identical across strategies.
        comparisons = []
        for strategy in ("add", "cap", "gated"):
            prompt = [100 + len(comparisons)] * 256
            prompt[12] = close  # historical close must never trigger
            salt = "producer-" + strategy
            request(prompt, strategy, salt=salt, length=8)
            history = prompt + [close] * 8 + [110] * 248
            request(history, strategy, label=strategy + "-rebuild", salt=salt)
            request(history, "none", label=strategy + "-reference", salt="reference-" + strategy,
                    skip=True)
            # Rebuilt history is now a clean prefix that a subsequent turn can reuse.
            request(history + [111] * 256, strategy, label=strategy + "-reuse", salt=salt)
            left = np.load(args.directory / "snapshots" / f"{strategy}-rebuild-rank0.npz")
            right = np.load(args.directory / "snapshots" / f"{strategy}-reference-rank0.npz")
            assert left.files == right.files
            maxima = {}
            for key in left.files:
                diff = np.abs(left[key] - right[key])
                maxima[key] = float(diff.max())
                # Prefill/decode kernels and chunk shapes differ; BF16 tolerance.
                np.testing.assert_allclose(left[key], right[key], rtol=0.08, atol=0.08,
                                           err_msg=f"{strategy}: {key}")
            comparisons.append({"strategy": strategy, "max_abs_by_tensor": maxima,
                                "tensors": len(maxima)})
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda s: request([120 if s == "cap" else 121] * 256, s,
                                           salt="concurrent-" + s, length=8), ["cap", "none"]))
        request([122] * 256)  # flush worker finished records
        manager.validate_run_evidence()
        events = [json.loads(line) for p in manager.evidence_directory.glob("*.jsonl")
                  for line in p.read_text().splitlines()]
        admissions = {e.get("label"): e["cached_tokens"] for e in events
                      if e["type"] == "cache_admitted" and e.get("label")}
        for output in outputs:
            label = output["label"] or ""
            if label.endswith("-reuse"):
                assert label in admissions and admissions[label] >= 272, (label, admissions)
        freezes = [e for e in events if e["type"] == "cache_freeze"]
        assert freezes and all(e["max_shared_tokens"] <= e["clean_prefix_end"] for e in freezes)
        decisions = [e for e in events if e["type"] == "decision" and e["strategy"] != "none"]
        for strategy in ("add", "cap", "gated"):
            selected = [e for e in decisions if e["strategy"] == strategy]
            assert selected and any(any(site[-1] > 0 for site in e["evidence_sums"]) for e in selected)
            assert all(e["site_counts"] == [1, 1] for e in selected)
        full_replay = sum(e["type"] == "dispatch" and e["full_replay"] for e in events)
        if args.mode == "auto":
            assert full_replay > 0
        write_json(args.directory / "smoke.json", {"complete": True, "synthetic": True,
                   "mode": args.mode, "outputs": outputs, "comparisons": comparisons,
                   "cache_freezes": freezes, "full_replay_dispatches": full_replay})
    print("CLEAN HISTORY SMOKE PASSED", args.mode, flush=True)


if __name__ == "__main__":
    main()
