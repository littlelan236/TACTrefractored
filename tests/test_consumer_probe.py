"""Guard real vector comparison, independent branches, routing and replay."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relative", [
    "packages/tact-vllm/src/tact_vllm/runtime/consumer_probe.py",
    "vendor/easysteer-vllm/vllm/model_hooks/steering/consumer_probe.py",
])
def test_real_consumer_corruption_and_row_selection(relative):
    spec = importlib.util.spec_from_file_location("consumer_probe", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probe = module.ConsumerProbe(8, 4, "cpu")
    hidden = torch.arange(32, dtype=torch.float32).reshape(4, 8).bfloat16()
    residual = torch.ones_like(hidden)
    active = torch.tensor([False, True, False, True])
    probe.produce(hidden, residual, active, torch.ones(4))
    actual = hidden.clone()
    actual[1, 5] += 1
    probe.consume(actual, residual)
    assert probe.stats[1, 1] == 1
    assert probe.stats[1, 2] == 1 and probe.stats[1, 4] == 1
    assert probe.stats[3, 2] == 0 and probe.stats[3, 4] == 0
    assert probe.stats[0, 1] == probe.stats[2, 1] == 0
    probe.reset()
    probe.produce(hidden, None, active, torch.ones(4))
    indices = torch.tensor([3, 1])
    probe.consume(hidden[indices], None, indices)
    assert probe.stats[:, 1].tolist() == [0, 1, 0, 1]
    assert probe.stats[:, 4:6].sum() == 0
    # Logits warmup may retain padded hidden rows beyond actual sample indices.
    probe.reset()
    probe.produce(hidden, None, active, torch.ones(4))
    probe.consume(torch.cat([hidden[indices], hidden[:2]]), None, indices)
    assert probe.stats[:, 1].tolist() == [0, 1, 0, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_capture_replay_compares_changed_values_not_only_capture_sample():
    spec = importlib.util.spec_from_file_location(
        "consumer_probe", ROOT / "packages/tact-vllm/src/tact_vllm/runtime/consumer_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probe = module.ConsumerProbe(8, 4, "cuda")
    source = torch.ones((4, 8), device="cuda", dtype=torch.bfloat16)
    consumed = source.clone()
    active = torch.tensor([True, False, True, False], device="cuda")

    def run():
        probe.produce(source, None, active, torch.ones(4, device="cuda"))
        probe.consume(consumed, None)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    probe.reset()
    graph.replay()
    assert probe.stats[:, 4].sum().item() == 0
    probe.reset()
    consumed[2, 4] += 1
    graph.replay()
    assert probe.stats[2, 4].item() == 1
    assert probe.stats[2, 2].item() == 1
    assert probe.stats[0, 4].item() == 0
