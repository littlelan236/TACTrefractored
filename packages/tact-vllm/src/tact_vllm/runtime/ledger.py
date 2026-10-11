from __future__ import annotations

from dataclasses import dataclass

from tact_vllm.runtime.plans import Plan


@dataclass
class RequestEntry:
    slot: int
    epoch: int
    plan: Plan
    scanned: int = 0
    first_close: int | None = None

    def observe(self, outputs: list[int], prompt_length: int, close_id: int):
        if len(outputs) < self.scanned:
            raise RuntimeError(
                "Accepted history shrank; streaming/speculative rollback is unsupported"
            )
        if self.first_close is None:
            for i in range(self.scanned, len(outputs)):
                if outputs[i] == close_id:
                    self.first_close = prompt_length + i
                    break
        self.scanned = len(outputs)

    def eligible_positions(self, start: int, count: int):
        return [start + offset == self.first_close for offset in range(count)]


class RequestLedger:
    def __init__(self, capacity: int):
        self.entries: dict[str, RequestEntry] = {}
        self.free = list(range(capacity, 0, -1))
        self.epochs = [0] * (capacity + 1)

    def admit(self, request_id: str, plan: Plan) -> RequestEntry:
        if request_id in self.entries:
            entry = self.entries[request_id]
            if entry.plan != plan:
                raise ValueError("An admitted TACT plan cannot change")
            return entry
        if not self.free:
            raise RuntimeError("TACT request buffer capacity exceeded")
        slot = self.free.pop()
        self.epochs[slot] += 1
        entry = RequestEntry(slot, self.epochs[slot], plan)
        self.entries[request_id] = entry
        return entry

    def release(self, request_id: str):
        entry = self.entries.pop(request_id)
        self.free.append(entry.slot)
        return entry
