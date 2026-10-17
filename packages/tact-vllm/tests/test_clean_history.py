"""Publication invariants and the actual audited native hybrid cache managers."""
from types import SimpleNamespace

import pytest
import torch

from tact_vllm.adapters.kv_cache import CleanHistoryCoordinator
from tact_vllm.runtime.cache_policy import CleanHistoryPolicy


def manifest():
    return {"bundle_id": "cache-test", "close_id": 9,
            "sites": [{"kind": "decoder_output", "block_index": 0},
                      {"kind": "final_norm", "block_index": None}]}


def fields(strategy="cap", sites=None):
    return {"tact_schema": 1, "tact_bundle_id": "cache-test", "tact_strategy": strategy,
            "tact_k": 2, "tact_lambda": 0.5, "tact_additive_scale": -0.5,
            "tact_site_ids": [0, 1] if sites is None else sites}


def fake_request(name="request", strategy="cap", sites=None):
    return SimpleNamespace(request_id=name, num_prompt_tokens=8,
                           prompt_token_ids=[9] * 8, output_token_ids=[],
                           sampling_params=SimpleNamespace(extra_args=fields(strategy, sites)))


@pytest.mark.parametrize("strategy", ["cap", "gated", "add"])
@pytest.mark.parametrize("offset", [0, 1, 3, 4])
def test_close_is_excluded_even_at_block_boundaries(strategy, offset):
    policy = CleanHistoryPolicy(manifest())
    request = fake_request(strategy=strategy)
    assert policy.allow_publication(request, 8)  # prompt closes do not count
    request.output_token_ids.extend([2] * offset + [9])
    boundary = 8 + offset
    assert policy.allow_publication(request, boundary)
    assert not policy.allow_publication(request, boundary + 1)
    request.output_token_ids.extend([3, 9, 4])
    assert not policy.allow_publication(request, boundary + 4)
    # Never re-label a later Mamba checkpoint with an old clean boundary.
    assert not policy.allow_publication(request, boundary)
    assert policy.entries[request.request_id].max_shared_tokens == boundary


@pytest.mark.parametrize("strategy,sites", [("none", [0, 1]), ("cap", [1]),
                                           ("add", [1]), ("gated", [1])])
def test_none_and_final_norm_only_can_share_the_full_fixed_token_sequence(strategy, sites):
    policy = CleanHistoryPolicy(manifest())
    request = fake_request(strategy=strategy, sites=sites)
    request.output_token_ids.extend([9, 2, 3])
    assert policy.allow_publication(request, 11)
    assert not policy.entries[request.request_id].affects_cache


def test_request_isolation_cleanup_and_plan_immutability():
    policy = CleanHistoryPolicy(manifest())
    dirty, clean = fake_request("dirty"), fake_request("clean")
    dirty.output_token_ids.append(9)
    assert not policy.allow_publication(dirty, 9)
    assert policy.allow_publication(clean, 8)
    dirty.sampling_params.extra_args["tact_k"] = 3
    with pytest.raises(ValueError, match="cannot change"):
        policy.observe(dirty)
    dirty.sampling_params.extra_args["tact_k"] = 2
    policy.release(dirty)
    assert policy.allow_publication(fake_request("dirty"), 8)


def test_rollback_is_rejected_and_terminal_unconsumed_close_does_not_freeze():
    policy = CleanHistoryPolicy(manifest())
    request = fake_request()
    assert policy.allow_publication(request, 8)
    request.output_token_ids.append(9)
    assert policy.observe(request).decision.first_close == 8
    assert not policy.entries[request.request_id].frozen
    request.output_token_ids.clear()
    with pytest.raises(RuntimeError, match="shrank"):
        policy.observe(request)


def test_coordinator_blocks_the_whole_native_publication_call():
    calls = []
    native = SimpleNamespace(cache_blocks=lambda r, n: calls.append(n), group_block_sizes=(4, 8))
    policy = CleanHistoryPolicy(manifest())
    coordinator = CleanHistoryCoordinator(native, policy)
    request = fake_request()
    coordinator.cache_blocks(request, 8)
    request.output_token_ids.extend([2, 9])
    coordinator.cache_blocks(request, 10)
    coordinator.cache_blocks(request, 8)
    assert calls == [8]  # allocation/checkpoint internals never receive a clipped old position
    assert coordinator.group_block_sizes == (4, 8)


def native_fixture():
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec,
    )

    init_none_hash(sha256)
    groups = [KVCacheGroupSpec(["attention"], FullAttentionSpec(
        4, num_kv_heads=1, head_size=2, dtype=torch.float32)),
        KVCacheGroupSpec(["linear"], MambaSpec(
            4, shapes=((2, 2), (2, 2)), dtypes=(torch.float32, torch.float32),
            mamba_cache_mode="align"))]
    manager = KVCacheManager(KVCacheConfig(80, [], groups), max_model_len=128,
                             scheduler_block_size=4, hash_block_size=4)
    events = []
    policy = CleanHistoryPolicy(manifest(), events.append)
    manager.coordinator = CleanHistoryCoordinator(manager.coordinator, policy)
    hasher = get_request_block_hasher(4, sha256)
    return manager, policy, hasher, events


def real_request(name, prompt, hasher, strategy="cap"):
    from vllm import SamplingParams
    from vllm.v1.request import Request

    return Request(name, prompt, SamplingParams(extra_args=fields(strategy), max_tokens=64),
                   pooling_params=None, block_hasher=hasher)


def advance(manager, request, count):
    manager.new_step_starts()
    blocks = manager.allocate_slots(request, count)
    assert blocks is not None
    request.num_computed_tokens += count
    return blocks


@pytest.mark.parametrize("strategy", ["cap", "gated", "add"])
@pytest.mark.parametrize("offset", [0, 2, 4])
def test_actual_allocate_slots_and_explicit_publish_keep_dirty_hybrid_tail_private(strategy, offset):
    manager, policy, hasher, events = native_fixture()
    prompt = [2, 9, 3, 4] * 2
    producer = real_request("producer", prompt, hasher, strategy)
    advance(manager, producer, 8)
    for token in [2] * offset + [9] + [3] * 10:
        producer.append_output_token_ids(token)
        advance(manager, producer, 1)
        # Exercise the second native cache-publication entry point as well.
        manager.cache_blocks(producer, producer.num_computed_tokens)
    boundary = len(prompt) + offset
    entry = policy.entries["producer"]
    assert entry.frozen and entry.max_shared_tokens <= boundary
    manager.free(producer)
    policy.release(producer)
    history = list(producer.all_token_ids)
    consumer = real_request("consumer", history + [5] * (4 - len(history) % 4), hasher)
    manager.new_step_starts()
    blocks, hits, _ = manager.get_computed_blocks(consumer)
    assert 0 < hits <= boundary
    assert blocks.blocks[1][-1].block_hash_num_tokens <= boundary
    # Cached history has no generated outputs: its old closes do not freeze.
    assert not policy.observe(consumer).frozen
    allocated = manager.allocate_slots(consumer, consumer.num_tokens - hits,
                                      num_new_computed_tokens=hits, new_computed_blocks=blocks)
    assert allocated is not None
    consumer.num_computed_tokens = consumer.num_tokens
    manager.free(consumer)
    policy.release(consumer)
    replay = real_request("replay", list(consumer.all_token_ids) + [6] * 4, hasher)
    manager.new_step_starts()
    _, clean_hits, _ = manager.get_computed_blocks(replay)
    assert clean_hits > boundary  # the re-prefilled historical tail is shareable again
    assert any(e["type"] == "cache_freeze" for e in events)
