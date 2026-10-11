"""Controlled native API smoke with consumed close tokens and worker evidence."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, DirectionVector, SteeringSpec, TactBandPayload, VectorSpec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--max-model-len", type=int, default=256)
    args = p.parse_args()
    config = json.loads((Path(args.model) / "config.json").read_text())
    text = config.get("text_config", config)
    n, h = text["num_hidden_layers"], text["hidden_size"]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.65,
        dtype="bfloat16",
        enable_steer_vector=True,
        steer_algorithms=["tact_add", "tact_cap", "tact_gate"],
        steer_graph_mode="in_graph",
        steer_vector_dtype="float32",
        language_model_only=True,
        enable_prefix_caching=True,
        async_scheduling=False,
        compilation_config={
            "cudagraph_capture_sizes": [1, 2, 4],
            "pass_config": {"enable_sp": False},
        },
        enforce_eager=False,
    )
    tok = llm.get_tokenizer()
    close = tok.encode("</think>", add_special_tokens=False)
    assert len(close) == 1
    # Every output token is close. Prompt also includes close; generation-only must exclude it.
    prompts = [
        {"prompt_token_ids": tok.encode("hello </think> hello", add_special_tokens=False)},
        {
            "prompt_token_ids": tok.encode(
                "hello hello hello </think> hello", add_special_tokens=False
            )
        },
    ]
    ot = np.zeros(h, np.float32)
    ot[0] = 1
    oa = np.zeros(h, np.float32)
    oa[1] = 1
    sites = [n - 2, n]
    site_manifest = {
        "schema_version": 1,
        "layer_count": n,
        "hidden_size": h,
        "sites": {
            str(i): {
                "kind": "final_norm" if i == n else "decoder_output",
                "block_index": None if i == n else i,
            }
            for i in sites
        },
    }
    data = {
        i: {
            "v_ot": ot,
            "v_oa": oa,
            "mu": np.full(h, 4.0, np.float32),
            "tau_ot": np.float32(0),
            "tau_oa": np.float32(0),
        }
        for i in sites
    }
    band = TactBandPayload(data, site_manifest, k=0)
    apply = ApplySpec(generation_tokens=close)

    def spec(method, scale=1):
        return SteeringSpec(
            vectors=[
                VectorSpec(
                    algorithm=method,
                    data=DirectionVector({i: ot + oa for i in sites})
                    if method == "tact_add"
                    else band,
                    scale=scale,
                    normalize=False,
                    apply=apply,
                )
            ]
        )

    sampling = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, allowed_token_ids=close)
    results = []
    for name, steering in [
        ("baseline", False),
        ("cap", spec("tact_cap")),
        ("gate_zero", spec("tact_gate", 0)),
        ("gate", spec("tact_gate", 0.5)),
        ("add", spec("tact_add", -0.5)),
        ("baseline_after", False),
    ]:
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, steering=steering, use_tqdm=False)
        evidence = llm.collective_rpc("get_tact_runtime_evidence")
        assert all(o.outputs[0].token_ids == close * 4 for o in outputs)
        results.append(
            {
                "method": name,
                "seconds": time.perf_counter() - start,
                "evidence": evidence,
                "tokens": [o.outputs[0].token_ids for o in outputs],
            }
        )
    previous = None
    for item in results:
        for rank, worker in enumerate(item["evidence"]):
            values = {site["site_id"]: site["values"] for site in worker["sites"]}
            if previous is None:
                assert all(values[site][0] == 0 for site in sites)
            else:
                old = {site["site_id"]: site["values"] for site in previous[rank]["sites"]}
                expected_matches = 0 if item["method"].startswith("baseline") else 6
                expected_changes = (
                    0 if item["method"] in ("gate_zero", "baseline_after") else expected_matches
                )
                assert all(values[site][0] - old[site][0] == expected_matches for site in sites)
                assert all(values[site][1] - old[site][1] == expected_changes for site in sites)
        previous = item["evidence"]
    mixed = llm.generate(
        prompts * 2,
        sampling,
        steering=[False, spec("tact_add", -0.5), spec("tact_cap"), spec("tact_gate", 2)],
        use_tqdm=False,
    )
    assert len(mixed) == 4
    evidence = llm.collective_rpc("get_tact_runtime_evidence")
    for worker in evidence:
        assert worker["steering_mode"] == "in_graph"
        assert worker["full_replays"] + worker["piecewise_replays"] > 0
        selected = [s for s in worker["sites"] if s["site_id"] in sites]
        assert len(selected) == 2
        assert all(s["values"][0] > 0 and s["values"][1] > 0 for s in selected)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "tp": args.tp,
                "sites": sites,
                "results": results,
                "mixed_evidence": evidence,
            },
            indent=2,
        )
        + "\n"
    )
    print("Controlled close/mixed native API/replay smoke passed:", out)


if __name__ == "__main__":
    main()
