from __future__ import annotations

import concurrent.futures
import inspect
import time
from pathlib import Path

from tact.artifacts import hash_json, identifier, now, read_json, update_status, write_json
from tact.config import select_instances
from tact.environments.containers import ContainerFactory, TaskContext


def install_harness_adapter(cfg: dict, factory: ContainerFactory, rows: list[dict], output: Path):
    import importlib.metadata

    from swebench.harness import run_evaluation as official
    from swebench.harness.test_spec import python as sb_python
    from swebench.harness.test_spec.test_spec import TestSpec

    if importlib.metadata.version("swebench") != "4.1.0":
        raise RuntimeError("Harness adapter is validated only against swebench 4.1.0")
    required = {"test_spec", "pred", "client", "run_id", "timeout"}
    if not required.issubset(inspect.signature(official.run_instance).parameters):
        raise RuntimeError("SWE harness interface changed")
    instances = {r["instance_id"]: r for r in rows}
    original_script = TestSpec.eval_script
    original_get = sb_python.requests.get

    # Keep exact downloaded metadata content in the project, then prefer it offline.
    cache = Path(cfg["project_root"]) / "data/swebench/requirements-cache"
    cache.mkdir(parents=True, exist_ok=True)
    def cached_get(url, *args, **kwargs):
        import requests
        if not url.startswith("https://raw.githubusercontent.com/"):
            return original_get(url, *args, **kwargs)
        p = cache / (hash_json(url) + ".txt")
        if p.exists():
            response = requests.Response()
            response.status_code = 200
            response._content = p.read_bytes()
            response.url = url
            return response
        kwargs.setdefault("timeout", 30)
        try:
            response = original_get(url, *args, **kwargs)
            response.raise_for_status()
        except requests.RequestException:
            parts = url.removeprefix("https://raw.githubusercontent.com/").split("/", 3)
            if len(parts) != 4:
                raise
            owner, repo, revision, path = parts
            response = original_get(f"https://cdn.jsdelivr.net/gh/{owner}/{repo}@{revision}/{path}",
                                    *args, **kwargs)
            response.raise_for_status()
        p.write_bytes(response.content)
        return response

    def build_container(test_spec, client, run_id, logger, nocache, force_rebuild):
        if force_rebuild or nocache:
            raise ValueError("Project harness uses locked prebuilt images, not implicit rebuilds")
        iid = test_spec.instance_id
        ctx = TaskContext(output.parent.parent.name, iid, output.name, "harness",
                          output / "environment" / iid)
        return factory.create(instances[iid], ctx)

    # This is scoped to a dedicated evaluation process, not docker SDK globally.
    official.build_container = build_container
    official.cleanup_container = lambda client, container, logger: factory.close(container) if container else None
    official.RUN_EVALUATION_LOG_DIR = output / "harness"
    sb_python.requests.get = cached_get
    TestSpec.eval_script = property(lambda self: factory.policy.wrap_eval(original_script.fget(self)))

    def execute_eval(container, cmd, timeout):
        started = time.monotonic()
        out = factory.execute(container, "source /opt/miniconda3/bin/activate && " + cmd, timeout)
        timed_out = out["returncode"] in [124, 137]
        if not timed_out:
            try:
                factory.probe(container, "environment-after-eval.json")
            except Exception as exc:
                write_json(container.context.output_dir / "post-eval-probe-error.json", {"error": str(exc)})
                raise
        return out["output"], timed_out, time.monotonic() - started

    official.exec_run_with_timeout = execute_eval
    return official


def evaluate(cfg: dict, run: Path, eval_id: str, gold: bool = False) -> dict:
    output = run / "evaluation" / identifier(eval_id)
    if output.exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    output.mkdir(parents=True)
    write_json(output / "resolved-config.json", cfg)
    rows, _ = select_instances(cfg)
    images = read_json(run / "images.json") if (run / "images.json").exists() else None
    factory = ContainerFactory(cfg, images)
    update_status(output / "status.json", "evaluation")
    outcomes = {}
    try:
        write_json(run / "images.json", factory.lock_images(rows))
        official = install_harness_adapter(cfg, factory, rows, output)
        if gold:
            predictions = {r["instance_id"]: {"instance_id": r["instance_id"],
                "model_name_or_path": "gold", "model_patch": r["patch"]} for r in rows}
        else:
            predictions = {r["instance_id"]: r for r in read_json(run / "predictions.json")}
        write_json(output / "predictions.json", list(predictions.values()))
        specs = {r["instance_id"]: official.make_test_spec(r, namespace="swebench") for r in rows}

        def worker(row):
            iid = row["instance_id"]
            pred = predictions.get(iid)
            if not pred or not pred.get("model_patch"):
                return iid, {"outcome": "MissingPatch", "completed": False, "resolved": False}
            result = official.run_instance(specs[iid], pred, False, False, factory.client,
                                            eval_id, cfg["timeouts"]["evaluation_seconds"])
            outcome = "RESOLVED" if result["resolved"] else "UNRESOLVED"
            if not result["completed"]:
                outcome = "EvaluationFailed"
                logs = list((output / "harness" / eval_id).glob(f"*/{iid}/run_instance.log"))
                if logs and "Test timed out after" in logs[0].read_text(errors="replace"):
                    outcome = "EvaluationTimeout"
            return iid, {"outcome": outcome, **result}

        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["execution"]["harness_workers"]) as pool:
            futures = [pool.submit(worker, row) for row in rows]
            for future in concurrent.futures.as_completed(futures):
                iid, result = future.result()
                outcomes[iid] = result
                write_json(output / "progress.json", outcomes)
        resolved = sum(r["resolved"] for r in outcomes.values())
        failures = [i for i, r in outcomes.items() if r["outcome"] in ["EvaluationFailed", "EvaluationTimeout"]]
        summary = {"evaluation_id": eval_id, "gold": gold, "requested": len(rows),
                   "resolved": resolved, "resolve_rate": resolved / len(rows),
                   "graded": sum(r["completed"] for r in outcomes.values()),
                   "complete": not failures and (not gold or resolved == len(rows)),
                   "failures": failures, "outcomes": outcomes, "finished_at": now()}
        write_json(output / "summary.json", summary)
        update_status(output / "status.json", "complete" if summary["complete"] else "failed")
        return summary
    except BaseException as exc:
        update_status(output / "status.json", "failed", error=str(exc))
        raise
    finally:
        factory.cleanup_owned(run.name)
