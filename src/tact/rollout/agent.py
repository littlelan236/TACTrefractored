"""Project agent: durable trajectories and optional action-cycle intervention."""
from __future__ import annotations
import json
from collections import deque
from pathlib import Path
from minisweagent.agents.default import DefaultAgent
from tact.artifacts import write_json, now
from tact.config import LoopIntervention


def action_signature(message: dict) -> str | None:
    """Compare parsed tool name/arguments, excluding IDs and reasoning.

    Preserve command whitespace: changing it inside quoted strings/heredocs can
    change shell semantics. JSON formatting/key order is normalized instead.
    """
    calls = message.get('tool_calls') or []
    if not calls:
        return None
    actions = []
    for call in calls:
        fn = call.get('function', call)
        args = fn.get('arguments', {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        actions.append({'name': fn.get('name'), 'arguments': args})
    return json.dumps(actions, sort_keys=True, ensure_ascii=False)


class ActionCycleDetector:
    def __init__(self, max_period=5, min_repeats=2):
        self.max_period = max_period
        self.min_repeats = min_repeats
        self.history = deque(maxlen=max_period * min_repeats)

    def observe(self, message: dict, step: int) -> dict | None:
        sig = action_signature(message)
        if sig is None:
            self.history.clear()
            return None
        self.history.append((step, sig))
        history = list(self.history)
        for period in range(1, self.max_period + 1):
            length = period * self.min_repeats
            if len(history) < length:
                continue
            suffix = history[-length:]
            values = [x[1] for x in suffix]
            if all(values[i] == values[i % period] for i in range(length)):
                event = {'trigger_step': step, 'period': period,
                         'repeats': self.min_repeats,
                         'matched_steps': [x[0] for x in suffix],
                         'action_signatures': values[:period]}
                self.history.clear()
                return event
        return None


class DurableAgent(DefaultAgent):
    def __init__(self, *args, loop_intervention=None, **kwargs):
        self.loop_settings = LoopIntervention.model_validate(loop_intervention or {})
        self.loop_detector = ActionCycleDetector(self.loop_settings.max_period,
                                                 self.loop_settings.min_repeats)
        self.loop_events = []
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        self.loop_detector.history.clear()
        self.loop_events = []
        return super().run(*args, **kwargs)

    def step(self):
        try:
            return super().step()
        except BaseException:
            # A format error, interrupted action, or terminal step breaks the cycle.
            self.loop_detector.history.clear()
            raise

    def execute_actions(self, message):
        observations = super().execute_actions(message)
        step = sum(m.get("role") == "assistant" for m in self.messages) - 1
        if self.loop_settings.enabled and not any(m.get('role') == 'exit' for m in observations):
            event = self.loop_detector.observe(message, step)
            if event:
                event.update({'at': now(), 'prompt': self.loop_settings.prompt,
                              'message_index': len(self.messages)})
                self.loop_events.append(event)
                self.add_messages(self.model.format_message(
                    role='user', content=self.loop_settings.prompt,
                    extra={'tact_loop_intervention': event}))
        return observations

    def serialize(self, *extra_dicts):
        data = super().serialize(*extra_dicts)
        if self.loop_settings.enabled:
            data['info']['loop_intervention'] = {
                'config': self.loop_settings.model_dump(),
                'count': len(self.loop_events), 'events': self.loop_events}
        return data

    def save(self, path, *extra_dicts):
        data = self.serialize(*extra_dicts)
        if path:
            path = Path(path)
            write_json(path, data)
            if self.loop_settings.enabled:
                audit = path.parent / 'loop-interventions.jsonl'
                temp = audit.with_suffix('.jsonl.tmp')
                temp.write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n'
                                        for e in self.loop_events), encoding='utf-8')
                temp.replace(audit)
        return data
