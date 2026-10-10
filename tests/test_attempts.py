from unittest.mock import Mock

from tact.artifacts import read_json, write_json
from tact.config import resolve_config
from tact.environments.containers import ContainerCorrupted
from tact.rollout.runner import process_task


def test_infrastructure_retry_uses_fresh_attempt_and_retains_previous(monkeypatch, tmp_path):
    import minisweagent.models.litellm_model as model_module

    import tact.environments.rollout_adapter as adapter
    from tact.rollout import runner
    cfg = resolve_config(overrides=["execution.infrastructure_retry_limit=1"])
    instance = {"instance_id": "test__task-1", "problem_statement": "task"}
    environments = []

    class FakeEnvironment:
        def __init__(self, factory, instance, context):
            environments.append(context)
        def close(self):
            pass

    class FakeAgent:
        def __init__(self, model, environment, output_path, **kwargs):
            self.path = output_path
        def run(self, task):
            if len(environments) == 1:
                raise ContainerCorrupted("container died")
            return {"exit_status": "Submitted", "submission": "patch"}
        def save(self, path, data):
            write_json(path, {**data, "messages": []})

    monkeypatch.setattr(adapter, "RolloutEnvironment", FakeEnvironment)
    monkeypatch.setattr(model_module, "LitellmModel", lambda **kwargs: object())
    monkeypatch.setattr(runner, "atomic_agent_class", lambda: FakeAgent)
    result = process_task(cfg, tmp_path / "run", instance, Mock())
    assert result["exit_status"] == "Submitted"
    assert len(environments) == 2
    attempts = tmp_path / "run/tasks/test__task-1/attempts"
    assert read_json(attempts / "attempt-000/attempt.json")["exit_status"] == "ContainerCorrupted"
    assert read_json(attempts / "attempt-001/attempt.json")["exit_status"] == "Submitted"
    assert (attempts / "attempt-000/test__task-1.traj.json").is_file()
