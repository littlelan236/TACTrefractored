"""Explicit per-engine coordinator composition, without modifying vLLM classes."""
from __future__ import annotations


class CleanHistoryCoordinator:
    def __init__(self, native, policy):
        self.native = native
        self.policy = policy

    def __getattr__(self, name):
        return getattr(self.native, name)

    def cache_blocks(self, request, num_computed_tokens):
        # Gate the whole call, not a truncated num_tokens: Mamba's partial-tail
        # publisher can use its current checkpoint even when num_tokens is smaller.
        # Delegating after steering could label a dirty state with a clean prefix.
        # All allocation, private-state progression, CoW and freeing stay native.
        if self.policy.allow_publication(request, num_computed_tokens):
            self.native.cache_blocks(request, num_computed_tokens)
