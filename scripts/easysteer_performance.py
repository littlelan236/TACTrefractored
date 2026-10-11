"""End-to-end selector/serving benchmark, including cold payload admission separately."""

import argparse
import json
import statistics
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
    p.add_argument("--native", action="store_true")
    args = p.parse_args()
    config = json.loads((Path(args.model) / "config.json").read_text())
    text = config.get("text_config", config)
    n, h = text["num_hidden_layers"], text["hidden_size"]
    options = (
        {}
        if args.native
        else {
            "enable_steer_vector": True,
            "steer_algorithms": ["tact_add", "tact_cap", "tact_gate"],
            "steer_graph_mode": "in_graph",
            "steer_vector_dtype": "float32",
        }
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.65,
        dtype="bfloat16",
        language_model_only=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        compilation_config={
            "cudagraph_capture_sizes": [1, 2, 4, 8],
            "pass_config": {"enable_sp": False},
        },
        **options,
    )
    tok = llm.get_tokenizer()
    close = tok.encode("</think>", add_special_tokens=False)
    assert len(close) == 1
    ot = np.zeros(h, np.float32)
    oa = ot.copy()
    ot[0] = 1
    oa[1] = 1
    sites = list(range(max(0, n - 10), n - 1)) + [n]
    band = TactBandPayload(
        {
            i: {
                "v_ot": ot,
                "v_oa": oa,
                "mu": np.zeros(h, np.float32),
                "tau_ot": np.float32(1),
                "tau_oa": np.float32(1),
            }
            for i in sites
        },
        {
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
        },
    )
    strategies = {"none": False}
    if not args.native:
        for name, algo, scale in [
            ("add", "tact_add", -0.5),
            ("cap", "tact_cap", 1),
            ("gate", "tact_gate", 0.5),
        ]:
            strategies[name] = SteeringSpec(
                vectors=[
                    VectorSpec(
                        algorithm=algo,
                        data=DirectionVector({i: ot + oa for i in sites})
                        if name == "add"
                        else band,
                        scale=scale,
                        normalize=False,
                        apply=ApplySpec(generation_tokens=close),
                    )
                ]
            )
    results = []
    token = tok.encode("hello", add_special_tokens=False)[0]
    for batch in [1, 4, 8]:
        for prompt_len in [16, 512]:
            prompts = [{"prompt_token_ids": [token] * prompt_len} for _ in range(batch)]
            for workload in ["ordinary", "close_stress"]:
                sample = SamplingParams(
                    temperature=0,
                    max_tokens=128,
                    ignore_eos=True,
                    allowed_token_ids=close if workload == "close_stress" else [token],
                )
                for name, steering in strategies.items():
                    kwargs = {} if args.native else {"steering": steering}
                    start = time.perf_counter()
                    llm.generate(prompts, sample, use_tqdm=False, **kwargs)
                    cold = time.perf_counter() - start
                    samples = []
                    for _ in range(3):
                        start = time.perf_counter()
                        outputs = llm.generate(prompts, sample, use_tqdm=False, **kwargs)
                        samples.append(time.perf_counter() - start)
                    count = sum(len(o.outputs[0].token_ids) for o in outputs)
                    results.append(
                        {
                            "batch": batch,
                            "prompt_len": prompt_len,
                            "workload": workload,
                            "method": name,
                            "cold_seconds": cold,
                            "seconds": samples,
                            "tokens": count,
                            "median_tokens_per_second": count / statistics.median(samples),
                        }
                    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "tp": args.tp,
                "native": args.native,
                "diagnostics": False,
                "sites": sites,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print("End-to-end performance matrix saved:", out)


if __name__ == "__main__":
    main()
