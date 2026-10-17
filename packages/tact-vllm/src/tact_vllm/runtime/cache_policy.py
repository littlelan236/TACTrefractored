"""CPU-side publication policy; a request's live cache is never truncated."""
from __future__ import annotations

from dataclasses import dataclass

from tact_vllm.runtime.ledger import RequestEntry
from tact_vllm.runtime.plans import parse_plan

CLEAN_HISTORY_POLICY = "clean_history_v1"
CLEAN_HISTORY_SCHEDULER = "tact_vllm.adapters.scheduler.TactScheduler"


@dataclass
class CacheEntry:
    decision: RequestEntry
    prompt_length: int
    affects_cache: bool
    frozen: bool = False
    max_shared_tokens: int = 0
    blocked_publications: int = 0

    def summary(self):
        return {
            "strategy": self.decision.plan.strategy,
            "first_close_position": self.decision.first_close,
            "clean_prefix_end": self.decision.first_close if self.affects_cache else None,
            "affects_cache": self.affects_cache,
            "publication_frozen": self.frozen,
            "max_shared_tokens": self.max_shared_tokens,
            "blocked_publications": self.blocked_publications,
        }


class CleanHistoryPolicy:
    def __init__(self, manifest, emit=None):
        self.manifest = manifest
        self.entries: dict[str, CacheEntry] = {}
        self.emit = emit or (lambda event: None)

    def observe(self, request):
        plan = parse_plan(request.sampling_params.extra_args, self.manifest)
        entry = self.entries.get(request.request_id)
        if entry is None:
            # Final-norm-only steering cannot change a cache for fixed token IDs.
            affects_cache = plan.enabled and any(
                self.manifest["sites"][site]["kind"] == "decoder_output"
                for site in plan.sites
            )
            entry = CacheEntry(RequestEntry(0, 0, plan), request.num_prompt_tokens, affects_cache)
            self.entries[request.request_id] = entry
        elif entry.decision.plan != plan or entry.prompt_length != request.num_prompt_tokens:
            raise ValueError("An admitted TACT cache plan/prompt cannot change")
        previous = entry.decision.first_close
        entry.decision.observe(
            request.output_token_ids, request.num_prompt_tokens, self.manifest["close_id"]
        )
        if previous is None and entry.decision.first_close is not None:
            self.emit({"type": "cache_boundary", "request_id": request.request_id,
                       **entry.summary()})
        return entry

    def allow_publication(self, request, num_tokens):
        entry = self.observe(request)
        close = entry.decision.first_close
        crosses_close = entry.affects_cache and close is not None and num_tokens > close
        if entry.frozen or crosses_close:
            entry.blocked_publications += 1
            if not entry.frozen:
                entry.frozen = True
                self.emit({"type": "cache_freeze", "request_id": request.request_id,
                           "attempted_tokens": num_tokens, **entry.summary()})
            return False
        entry.max_shared_tokens = max(entry.max_shared_tokens, num_tokens)
        return True

    def release(self, request):
        entry = self.observe(request)
        self.emit({"type": "cache_finished", "request_id": request.request_id, **entry.summary()})
        del self.entries[request.request_id]
