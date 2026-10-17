from __future__ import annotations

from minisweagent.environments.docker import DockerEnvironmentConfig

from tact.environments.containers import ContainerFactory, TaskContext


class RolloutEnvironment:
    def __init__(self, factory: ContainerFactory, instance: dict, context: TaskContext):
        self.factory, self.context = factory, context
        cfg = factory.cfg
        self.config = DockerEnvironmentConfig(
            image=factory.spec(instance)["image_id"], cwd=cfg["environment"]["cwd"],
            env=factory.policy_for(instance).variables(), timeout=cfg["timeouts"]["tool_command_seconds"],
            container_timeout=str(cfg["timeouts"]["container_lifetime_seconds"]),
            executable=cfg["environment"]["docker_executable"],
            pull_timeout=cfg["timeouts"]["container_start_seconds"],
        )
        self.container = factory.create(instance, context)
        self.container.start()

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None):
        # Keep mini-swe-agent's native submission-marker detection.
        from minisweagent.environments.docker import DockerEnvironment
        result = self.factory.execute(self.container, action.get("command", ""),
                                      timeout or self.config.timeout, cwd or self.config.cwd)
        DockerEnvironment._check_finished(self, result)
        return result

    def get_template_vars(self, **kwargs):
        import platform
        return {**self.config.model_dump(), **platform.uname()._asdict(), **kwargs}

    def serialize(self):
        return {"info": {"config": {"environment": self.config.model_dump(mode="json"),
                                   "environment_type": "tact.environments.RolloutEnvironment"}}}

    def close(self):
        self.factory.close(self.container)
