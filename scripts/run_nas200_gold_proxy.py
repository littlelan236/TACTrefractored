"""Run the fixed NAS cohort's gold evaluation with explicit metadata proxying.

Only GitHub raw/jsDelivr metadata requests use Clash. Docker socket requests,
containers, original runtime configuration, and other jobs are unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def install_metadata_proxy(proxy, cache):
    import requests
    from tact.artifacts import hash_json, read_json, write_json

    original = requests.sessions.Session.request
    hosts = {"raw.githubusercontent.com", "cdn.jsdelivr.net"}
    cache.mkdir(parents=True, exist_ok=True)

    def request(session, method, url, **kwargs):
        if method.upper() != "GET" or urlsplit(url).hostname not in hosts:
            return original(session, method, url, **kwargs)
        path = cache / (hash_json(url) + ".json")
        if path.exists():
            item = read_json(path)
            if item["url"] != url:
                raise ValueError("Metadata cache URL mismatch")
            response = requests.Response()
            response.status_code = item["status"]
            response.url = url
            response._content = bytes.fromhex(item["content_hex"])
            return response
        kwargs["proxies"] = {"http": proxy, "https": proxy}
        kwargs["timeout"] = (10, 40)
        for attempt in range(3):
            try:
                response = original(session, method, url, **kwargs)
                if response.status_code in {200, 404, 410}:
                    write_json(path, {"url": url, "status": response.status_code,
                                      "content_hex": response.content.hex()})
                    print("metadata", response.status_code, url, flush=True)
                    return response
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    requests.sessions.Session.request = request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="verified200-nas-gold-20261015")
    parser.add_argument("--eval-id", default="gold-proxy-20261015")
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--phase", choices=["launch", "preflight", "evaluate"], default="launch")
    args = parser.parse_args()
    from tact.artifacts import identifier, write_json
    from tact.config import load_resolved, select_instances

    run = ROOT / "runs" / identifier(args.run_id)
    eval_id = identifier(args.eval_id)
    cfg = load_resolved(run / "resolved-config.json")
    rows, _ = select_instances(cfg)
    if len(rows) != 200 or not all(r.get("nas_image") for r in rows):
        raise ValueError("This launcher only accepts the fixed 200-task NAS cohort")
    logs = ROOT / "logs/nas200" / eval_id
    logs.mkdir(parents=True, exist_ok=True)
    if args.phase == "launch":
        import fcntl
        with (logs / "launcher.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (run / "evaluation" / eval_id).exists():
                raise FileExistsError("Evaluation ID already exists")
            for phase in ["preflight", "evaluate"]:
                write_json(logs / "status.json", {"stage": phase, "run_id": run.name,
                    "eval_id": eval_id, "tasks": len(rows), "proxy": args.proxy})
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                           "--run-id", run.name, "--eval-id", eval_id,
                           "--proxy", args.proxy, "--phase", phase]
                with (logs / (phase + ".log")).open("w") as output:
                    code = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                          stdout=output, stderr=subprocess.STDOUT).returncode
                if code:
                    write_json(logs / "status.json", {"stage": phase + "_failed", "exit_code": code,
                        "run_id": run.name, "eval_id": eval_id})
                    raise SystemExit(code)
            write_json(logs / "status.json", {"stage": "finished", "run_id": run.name,
                "eval_id": eval_id, "summary": str(run / "evaluation" / eval_id / "summary.json")})
            return
    cache = Path(cfg["environment"]["requirements_cache"]) / "nas200-proxy-http"
    install_metadata_proxy(args.proxy, cache)
    if args.phase == "preflight":
        from tact.environments.containers import ContainerFactory
        from tact.evaluation.runner import install_harness_adapter
        from tact.artifacts import read_json
        factory = ContainerFactory(cfg, read_json(run / "images.json"))
        try:
            official = install_harness_adapter(cfg, factory, rows, logs)
            completed = []
            for row in rows:
                official.make_test_spec(row, namespace="swebench")
                completed.append(row["instance_id"])
                print("spec", len(completed), "/", len(rows), row["instance_id"], flush=True)
            write_json(logs / "preflight.json", {"complete": True, "tasks": len(completed),
                "instance_ids": completed, "proxy": args.proxy, "http_cache": str(cache)})
        finally:
            factory.client.close()
    else:
        from tact.evaluation.runner import evaluate
        summary = evaluate(cfg, run, eval_id, gold=True)
        print(json.dumps({k:v for k,v in summary.items() if k != "outcomes"}, indent=2), flush=True)
        if not summary["complete"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
