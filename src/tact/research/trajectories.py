from __future__ import annotations

from pathlib import Path

from tact.artifacts import read_json


class TrajectoryReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = read_json(self.path)
        self.messages = self.data.get("messages", [])

    def steps(self):
        ordinal = 0
        for index, message in enumerate(self.messages):
            if message.get("role") != "assistant":
                continue
            observations = []
            for following in self.messages[index + 1:]:
                if following.get("role") == "assistant":
                    break
                if following.get("role") == "tool":
                    observations.append(following)
            yield {"step": ordinal, "message_index": index,
                   "reasoning": message.get("reasoning_content") or "",
                   "content": message.get("content") or "", "message": message,
                   "tool_calls": message.get("tool_calls") or [], "observations": observations,
                   "response": message.get("extra", {}).get("response")}
            ordinal += 1

    def responses(self):
        seen = set()
        for index, message in enumerate(self.messages):
            response = (message.get("extra") or {}).get("response")
            if not isinstance(response, dict):
                if response is not None:
                    yield {"message_index": index, "response": None, "malformed": True}
                continue
            # A response ID identifies one persisted call; without it, preserve each record.
            key = response.get("id")
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            yield {"message_index": index, "response": response, "malformed": False}

    def prepared_prefix(self, message_index: int) -> list[dict]:
        # OpenAI text-only runtime: no cache-control or multimodal rewriting enabled.
        prefix = [{k: v for k, v in m.items() if k != "extra"}
                  for m in self.messages[:message_index]]
        if any(m.get("role") == "exit" for m in prefix):
            raise ValueError("A terminal message cannot occur before an assistant request")
        return prefix

    def usage(self) -> dict:
        prompt, completion, missing, count = 0, 0, 0, 0
        for item in self.responses():
            count += 1
            usage = (item["response"] or {}).get("usage") or {}
            pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if type(pt) is int and pt >= 0:
                prompt += pt
            if type(ct) is int and ct >= 0:
                completion += ct
            if type(pt) is not int or type(ct) is not int or item["malformed"]:
                missing += 1
        calls = self.data.get("info", {}).get("model_stats", {}).get("api_calls")
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": prompt + completion, "persisted_responses": count,
                "missing_usage": missing, "api_calls": calls,
                "response_coverage_complete": missing == 0 and count > 0 and calls == count,
                "coverage_scope": "persisted agent query responses; unreturned HTTP retries unknown"}
