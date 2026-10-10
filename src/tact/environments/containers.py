from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import docker

from tact.artifacts import hash_json, now, write_json
from tact.environments.policy import EnvironmentPolicy, parse_probe, probe_command


class EnvironmentSetupFailed(RuntimeError):
    pass


class ContainerCorrupted(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskContext:
    run_id: str
    instance_id: str
    attempt_id: str
    role: str
    output_dir: Path


def image_name(instance: dict) -> str:
    return instance.get("image_name") or instance.get("docker_image") or (
        "swebench/sweb.eval.x86_64." + instance["instance_id"].replace("__", "_1776_") + ":latest"
    ).lower()


class ManagedContainer:
    """Docker SDK-compatible container with bootstrap at its start boundary."""

    def __init__(self, factory, container, context: TaskContext, instance: dict, spec: dict):
        self.factory, self.raw, self.context, self.instance, self.spec = (
            factory, container, context, instance, spec
        )
        self.started = False

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def start(self):
        if self.started:
            return
        self.raw.start()
        self.started = True
        try:
            bootstrap = self.factory.policy.bootstrap(self.instance["repo"])
            result = self.factory.execute(self, bootstrap,
                                          self.factory.cfg["timeouts"]["bootstrap_seconds"])
            (self.context.output_dir / "bootstrap.log").write_text(result["output"])
            if result["returncode"] != 0:
                raise EnvironmentSetupFailed(f"Bootstrap failed: {result['output'][-2000:]}")
            self.factory.probe(self, "environment.json")
            self.factory.event(self.context, "ready", container_id=self.id)
        except Exception:
            self.factory.close(self)
            raise

    def exec_run(self, cmd, **kwargs):
        # Official harness copy/patch helpers call this API directly.
        env = self.factory.policy.variables()
        supplied = kwargs.pop("environment", None) or {}
        if not isinstance(supplied, dict):
            supplied = dict(entry.split("=", 1) for entry in supplied)
        kwargs["environment"] = {**supplied, **env}
        kwargs.setdefault("workdir", self.factory.cfg["environment"]["cwd"])
        return self.raw.exec_run(cmd, **kwargs)

    def remove(self, **kwargs):
        try:
            return self.raw.remove(**kwargs)
        finally:
            self.factory.event(self.context, "removed", container_id=self.id)


class ContainerFactory:
    def __init__(self, cfg: dict, images: dict | None = None):
        self.cfg = cfg
        self.client = docker.DockerClient(base_url=cfg["environment"]["docker_host"],
                                          timeout=cfg["timeouts"]["container_start_seconds"])
        self.policy = EnvironmentPolicy(cfg)
        self.images = dict(images or {})
        self._lock = threading.Lock()

    def event(self, context: TaskContext, event: str, **extra):
        context.output_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"at": now(), "event": event, "role": context.role,
                           "run_id": context.run_id, "instance_id": context.instance_id,
                           "attempt_id": context.attempt_id, **extra}) + "\n"
        with self._lock, (context.output_dir / "container-events.jsonl").open("a") as f:
                f.write(line)

    def lock_images(self, instances: list[dict]) -> dict:
        for row in instances:
            iid = row["instance_id"]
            if iid not in self.images:
                img = self.client.images.get(image_name(row))
                self.images[iid] = {"image_id": img.id, "requested": image_name(row),
                                    "repo_digests": img.attrs.get("RepoDigests", [])}
            else:
                self.client.images.get(self.images[iid]["image_id"])
        return dict(self.images)

    def spec(self, instance: dict) -> dict:
        e = self.cfg["environment"]
        self.lock_images([instance])
        return {"image_id": self.images[instance["instance_id"]]["image_id"],
                "user": e["user"], "platform": e["platform"], "cwd": e["cwd"],
                "environment": self.policy.variables(),
                "volumes": {e["wheelhouse"]: {"bind": "/opt/tact-wheels", "mode": "ro"}},
                "command": ["sleep", str(self.cfg["timeouts"]["container_lifetime_seconds"])],
                "policy_hash": self.policy.identity(instance["repo"])["policy_hash"]}

    def create(self, instance: dict, context: TaskContext) -> ManagedContainer:
        spec = self.spec(instance)
        context.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(context.output_dir / "container-spec.json", {**spec, "spec_hash": hash_json(spec)})
        write_json(context.output_dir / "environment-policy.json", self.policy.identity(instance["repo"]))
        container = self.client.containers.create(
            image=spec["image_id"], user=spec["user"], platform=spec["platform"],
            working_dir=spec["cwd"], environment=spec["environment"], volumes=spec["volumes"],
            command=spec["command"], detach=True,
            name="tact-" + context.role + "-" + uuid.uuid4().hex[:16],
            labels={"tact.project": str(self.cfg["project_root"]),
                    "tact.run": context.run_id, "tact.task": context.instance_id,
                    "tact.attempt": context.attempt_id, "tact.role": context.role},
        )
        self.event(context, "created", container_id=container.id, spec_hash=hash_json(spec))
        return ManagedContainer(self, container, context, instance, spec)

    def execute(self, container: ManagedContainer, command: str, seconds: int,
                cwd: str | None = None) -> dict:
        if container.started:
            container.raw.reload()
            if container.raw.status != "running":
                raise ContainerCorrupted(f"Task container {container.id} is {container.raw.status}")
        result = container.exec_run(
            ["timeout", "--signal=TERM", "--kill-after=5", str(seconds), "bash", "-lc",
             "source /opt/miniconda3/bin/activate && conda activate testbed && " + command],
            workdir=cwd or self.cfg["environment"]["cwd"],
        )
        code = result.exit_code
        return {"output": result.output.decode("utf-8", errors="replace"),
                "returncode": code, "exception_info": "CommandTimeout" if code in [124, 137] else ""}

    def probe(self, container: ManagedContainer, filename: str) -> dict:
        out = self.execute(container, probe_command(), self.cfg["timeouts"]["bootstrap_seconds"])
        if out["returncode"] != 0:
            raise EnvironmentSetupFailed(out["output"][-2000:])
        data = parse_probe(out["output"])
        data["spec_hash"] = hash_json(container.spec)
        data["policy_hash"] = container.spec["policy_hash"]
        data["packages_hash"] = hash_json(data["packages"])
        write_json(container.context.output_dir / filename, data)
        return data

    def close(self, container: ManagedContainer):
        try:
            container.remove(force=True)
        except docker.errors.NotFound:
            pass

    def cleanup_owned(self, run_id: str):
        for container in self.client.containers.list(all=True, filters={"label": [
            "tact.project=" + self.cfg["project_root"], "tact.run=" + run_id
        ]}):
            container.remove(force=True)
