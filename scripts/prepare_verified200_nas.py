"""Reproducible held-out selection and resumable, deduplicated NAS image download.

Selection uses the analysis environment (SciPy); registry/download uses runner.
No Docker pull, prune, daemon restart, or existing dataset modification occurs.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/swebench/verified200-nas"
STORE = Path("/VM/wangtianyi/tactRefractor/verified200-20261015")
DIFFICULTIES = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def category(row):
    paths = re.findall(r"^diff --git a/(.*?) b/", row["patch"], re.M)
    repo = row["repo"]
    if repo == "django/django":
        # Classify by changed subsystem, not issue keywords or model outcomes.
        def one(p):
            if "admindocs/" in p or p.startswith("docs/"):
                return "documentation"
            if p.startswith("django/test/"):
                return "test_framework"
            if any(x in p for x in ["serializ", "django/utils/date", "django/utils/numberformat", "django/utils/duration", "django/utils/encoding"]):
                return "serialization_formatting"
            if any(x in p for x in ["django/core/management/", "django/core/checks/", "django/utils/autoreload", "django/conf/"]):
                return "static_analysis"
            if p.startswith("django/db/") or p.startswith("django/contrib/contenttypes/") or "django/core/cache/backends/db" in p:
                return "orm_database"
            return "web_http"
        votes = collections.Counter(one(p) for p in paths)
        return sorted(votes, key=lambda k: (-votes[k], k))[0], paths
    if repo == "sympy/sympy":
        return ("serialization_formatting" if any("sympy/printing/" in p or "sympy/parsing/" in p or "sympy/utilities/codegen" in p or "sympy/codegen/" in p or "sympy/utilities/lambdify" in p for p in paths) else "symbolic_math"), paths
    if repo == "astropy/astropy":
        return ("serialization_formatting" if any("astropy/io/" in p or "astropy/units/format/" in p for p in paths) else "scientific_arrays"), paths
    mapping = {"pydata/xarray": "scientific_arrays", "scikit-learn/scikit-learn": "machine_learning", "matplotlib/matplotlib": "visualization", "mwaskom/seaborn": "visualization", "sphinx-doc/sphinx": "documentation", "pytest-dev/pytest": "test_framework", "pylint-dev/pylint": "static_analysis", "psf/requests": "web_http", "pallets/flask": "web_http"}
    return mapping[repo], paths


def select():
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    if (DATA / "selection.json").exists():
        raise FileExistsError("Selection already exists; use manifests/download to resume")
    source = ROOT / "data/swebench/verified70/source-500.json"
    rows = read(source)
    old = read(ROOT / "data/swebench/instances.json")
    old_ids = {r["instance_id"] for r in old}
    candidates = sorted([r for r in rows if r["instance_id"] not in old_ids], key=lambda r: r["instance_id"])
    categories = read(ROOT / "data/swebench/verified70/catalog.json")["categories"]
    cats = list(categories)
    labels = [category(r)[0] for r in candidates]
    available_d = collections.Counter(r["difficulty"] for r in candidates)
    quotas = dict.fromkeys(DIFFICULTIES, 0)
    for _ in range(200):
        d = min((d for d in DIFFICULTIES if quotas[d] < available_d[d]), key=lambda d: (quotas[d], DIFFICULTIES.index(d)))
        quotas[d] += 1
    groups = [[i for i, c in enumerate(labels) if c == cat] for cat in cats]
    cells = [[i for i in g if candidates[i]["difficulty"] == d] for g in groups for d in DIFFICULTIES]
    all_groups = groups + cells
    n = len(candidates)
    slots = sum(len(g) for g in all_groups)
    a = lil_matrix((len(DIFFICULTIES) + len(all_groups), n + slots))
    lb = []; costs = np.zeros(n + slots)
    for i, r in enumerate(candidates):
        # Tie breaking is deterministic and independent of any benchmark outcome.
        costs[i] = int(hashlib.sha256(("verified200-20261015:" + r["instance_id"]).encode()).hexdigest()[:8], 16) / 2**32 / 10000
    for j, d in enumerate(DIFFICULTIES):
        for i, r in enumerate(candidates):
            if r["difficulty"] == d:
                a[j, i] = 1
        lb.append(quotas[d])
    cursor = n
    for j, g in enumerate(all_groups):
        row = len(DIFFICULTIES) + j
        for i in g:
            a[row, i] = 1
        for k in range(len(g)):
            a[row, cursor] = -1
            costs[cursor] = (10000 if j < len(groups) else 1) * (2 * k + 1)
            cursor += 1
        lb.append(0)
    result = milp(costs, integrality=np.ones(n + slots), bounds=Bounds(0, 1), constraints=LinearConstraint(a.tocsr(), lb, lb), options={"time_limit": 120})
    if not result.success:
        raise RuntimeError(result.message)
    selected = [r for i, r in enumerate(candidates) if result.x[i] > .5]
    assert len(selected) == 200 and not old_ids.intersection(r["instance_id"] for r in selected)
    catalog = [{"instance_id": r["instance_id"], "repo": r["repo"], "category": category(r)[0], "category_zh": categories[category(r)[0]], "difficulty": r["difficulty"], "issue_title": r["problem_statement"].splitlines()[0], "changed_paths": category(r)[1]} for r in selected]
    summary = {"schema_version": 1, "count": 200, "candidate_count": len(candidates), "source_dataset": "princeton-nlp/SWE-bench_Verified", "source_revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a", "source_sha256": sha(source), "existing_instances_sha256": sha(ROOT / "data/swebench/instances.json"), "overlap_existing": [], "difficulty_available": dict(available_d), "difficulty_counts": dict(collections.Counter(r["difficulty"] for r in selected)), "category_available": dict(collections.Counter(labels)), "category_counts": dict(collections.Counter(r["category"] for r in catalog)), "category_difficulty_counts": {cat: dict(collections.Counter(r["difficulty"] for r in catalog if r["category"] == cat)) for cat in cats}, "selection_policy": "No duplicate of any of the 266 existing instances. Official human fix-time difficulty quotas minimize squared imbalance subject to availability; functional-category squared imbalance minimized by integer optimization, then within-category difficulty imbalance. Deterministic SHA256 tie breaking; no model or gold-test outcomes used. Categories are project subsystem-based labels, not official SWE-bench labels.", "difficulty_limit": "All three >4h official tasks already belong to the existing cohort; only 26 unused 1-4h tasks exist. Exact four-way uniformity is impossible.", "categories": categories, "instance_ids": [r["instance_id"] for r in selected]}
    write(DATA / "selection.json", summary)
    write(DATA / "catalog.json", {"tasks": catalog, "categories": categories})
    write(DATA / "source-selected.json", selected)
    write(DATA / "splits.json", {"schema_version": 1, "partitions": {"axis-train": [], "axis-val": [], "eval": summary["instance_ids"]}})
    write(DATA / "exclusions.json", {"excluded": []})
    write(DATA / "fit-exclusions.json", {"instance_ids": summary["instance_ids"]})
    (DATA / "benchmark.yaml").write_text("# Independent held-out NAS cohort; all runtime defaults remain canonical.\nbenchmark:\n  instances: data/swebench/verified200-nas/instances.json\n  splits: data/swebench/verified200-nas/splits.json\n  exclusions: data/swebench/verified200-nas/exclusions.json\n  selection: eval\nresearch:\n  fit_exclusions: data/swebench/verified200-nas/fit-exclusions.json\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


class Registry:
    def __init__(self, proxy):
        import requests
        self.session = requests.Session()
        self.session.trust_env = False
        if proxy:
            self.session.proxies = {"http": proxy, "https": proxy}
        self.tokens = {}

    def get(self, repo, endpoint, stream=False, extra=None):
        accept = ", ".join(["application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json", "application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"])
        for attempt in range(8):
            try:
                if repo not in self.tokens:
                    r = self.session.get("https://auth.docker.io/token", params={"service": "registry.docker.io", "scope": f"repository:{repo}:pull"}, timeout=(15, 60))
                    r.raise_for_status()
                    self.tokens[repo] = r.json()["token"]
                r = self.session.get(f"https://registry-1.docker.io/v2/{repo}/{endpoint}", headers={"Authorization": "Bearer " + self.tokens[repo], "Accept": accept, **(extra or {})}, stream=stream, timeout=(15, 90))
                if r.status_code == 429:
                    r.close()
                    # Public Docker Hub cache; digest verification remains mandatory.
                    r = self.session.get(f"https://mirror.gcr.io/v2/{repo}/{endpoint}",
                                         headers={"Accept": accept, **(extra or {})},
                                         stream=stream, timeout=(15, 90))
                if r.status_code == 401:
                    self.tokens.pop(repo, None)
                r.raise_for_status()
                return r
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(30, 2**attempt))


def manifest_one(row, proxy):
    iid = row["instance_id"]
    p = STORE / "images" / (iid + ".json")
    if p.exists():
        return read(p)
    registry = Registry(proxy)
    repo = "swebench/sweb.eval.x86_64." + iid.replace("__", "_1776_").lower()
    response = registry.get(repo, "manifests/latest")
    raw = response.content
    top_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    m = response.json()
    if "manifests" in m:
        platforms = [x for x in m["manifests"] if x.get("platform", {}).get("os") == "linux" and x.get("platform", {}).get("architecture") == "amd64"]
        if len(platforms) != 1:
            raise ValueError(f"Ambiguous linux/amd64 image: {iid}")
        descriptor = platforms[0]
        response = registry.get(repo, "manifests/" + descriptor["digest"])
        raw = response.content
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == descriptor["digest"]
        m = response.json()
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    write(STORE / "manifests" / (digest[7:] + ".json"), m)
    # Persist the exact manifest bytes, since pretty printing changes its digest.
    blob_path = STORE / "blobs/sha256" / digest[7:]
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(raw)
    config = m["config"]
    config_bytes = registry.get(repo, "blobs/" + config["digest"]).content
    assert "sha256:" + hashlib.sha256(config_bytes).hexdigest() == config["digest"]
    (STORE / "blobs/sha256" / config["digest"][7:]).write_bytes(config_bytes)
    result = {"schema_version": 1, "instance_id": iid, "requested": repo + ":latest", "repository": repo, "top_digest": top_digest, "manifest_digest": digest, "manifest_size": len(raw), "image_id": digest, "config": config, "layers": m["layers"], "repo_digests": [repo + "@" + top_digest], "compressed_bytes": sum(x["size"] for x in m["layers"]), "platform": "linux/amd64", "metadata_received_bytes": len(config_bytes) + len(raw)}
    write(p, result)
    print("manifest", iid, round(result["compressed_bytes"] / 1e9, 3), "GB", flush=True)
    return result


def manifests(proxy, workers):
    if not os.path.ismount("/VM"):
        raise RuntimeError("/VM must be mounted; refusing writes to local fallback")
    STORE.mkdir(parents=True, exist_ok=True)
    rows = read(DATA / "source-selected.json")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda r: manifest_one(r, proxy), rows))
    layers = {}
    for r in results:
        for layer in r["layers"]:
            layers.setdefault(layer["digest"], {**layer, "repository": r["repository"]})
    missing = [l for l in layers.values() if not (STORE / "blobs/sha256" / l["digest"][7:]).exists()]
    summary = {"images": len(results), "unique_layers": len(layers), "download_layer_bytes": sum(l["size"] for l in missing), "unique_layer_bytes": sum(l["size"] for l in layers.values()), "sum_image_compressed_bytes": sum(r["compressed_bytes"] for r in results), "units": "GB=10^9 bytes, GiB=2^30 bytes; manifest payload excludes TLS/HTTP overhead and retry traffic", "store": str(STORE), "layers": list(layers.values())}
    write(STORE / "download-plan.json", summary)
    write(DATA / "download-plan.json", {k: v for k, v in summary.items() if k != "layers"})
    new_rows = []
    for row, record in zip(rows, results):
        row = dict(row)
        row["image_name"] = record["requested"]
        row["nas_image"] = {"store": str(STORE), "manifest": "images/" + row["instance_id"] + ".json", "manifest_sha256": sha(STORE / "images" / (row["instance_id"] + ".json"))}
        new_rows.append(row)
    write(DATA / "instances.json", new_rows)
    for name in ["selection.json", "catalog.json", "instances.json", "splits.json", "exclusions.json", "fit-exclusions.json", "benchmark.yaml", "download-plan.json"]:
        (STORE / "dataset").mkdir(exist_ok=True)
        (STORE / "dataset" / name).write_bytes((DATA / name).read_bytes())
    print(json.dumps({k: v for k, v in summary.items() if k != "layers"}, indent=2), flush=True)


def download(proxy, workers):
    if not os.path.ismount("/VM"):
        raise RuntimeError("NAS mount unavailable")
    import fcntl
    with (STORE / "download.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return download_locked(proxy, workers)


def seed_local():
    """Copy exact immutable compressed blobs already cached by our Docker daemon."""
    import docker
    client = docker.DockerClient(base_url="unix:///run/user/1002/docker.sock", timeout=120)
    try:
        root = Path(client.info()["DockerRootDir"])
    finally:
        client.close()
    content = root / "containerd/daemon/io.containerd.content.v1.content/blobs/sha256"
    copied = []
    for layer in read(STORE / "download-plan.json")["layers"]:
        source = content / layer["digest"][7:]
        target = STORE / "blobs/sha256" / layer["digest"][7:]
        if target.exists() or target.with_suffix(".part").exists() or not source.is_file():
            continue
        if source.stat().st_size != layer["size"] or sha(source) != layer["digest"][7:]:
            continue
        temp = target.with_suffix(".seed")
        shutil.copyfile(source, temp)
        if sha(temp) != layer["digest"][7:]:
            raise ValueError("Local seed copy corrupted")
        temp.replace(target)
        copied.append(layer)
        print("seeded", layer["digest"], layer["size"], flush=True)
    write(STORE / "local-seed.json", {"layers": copied, "bytes": sum(l["size"] for l in copied), "source": str(content), "source_modified": False})


class HashingWriter:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()

    def write(self, data):
        self.hash.update(data)
        return self.stream.write(data)


def rootless_layer(layer, wait):
    """Normalize unmappable ownership in Matplotlib's repository build layer.

    Original blobs remain immutable. File contents, paths, modes, links and
    timestamps are preserved; only uid/gid outside this daemon's map change.
    """
    metadata = STORE / "rootless/layers" / (layer["digest"][7:] + ".json")
    if metadata.exists():
        return read(metadata)
    source = STORE / "blobs/sha256" / layer["digest"][7:]
    while not source.exists():
        if not wait:
            raise FileNotFoundError(source)
        time.sleep(10)
    if sha(source) != layer["digest"][7:]:
        raise ValueError("Original rootless layer checksum mismatch")
    temp = source.with_suffix(".rootless-gz")
    changes = []
    with temp.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1) as compressed:
        hashed = HashingWriter(compressed)
        with tarfile.open(source, mode="r|gz") as before, tarfile.open(fileobj=hashed, mode="w|") as after:
            for member in before:
                changed = member.uid >= 65536 or member.gid >= 65536
                entry = copy.copy(member)
                entry.pax_headers = dict(member.pax_headers)
                if changed:
                    changes.append({"path": member.name, "uid": member.uid, "gid": member.gid})
                    if member.uid >= 65536:
                        entry.uid = 0; entry.uname = "root"; entry.pax_headers.pop("uid", None)
                    if member.gid >= 65536:
                        entry.gid = 0; entry.gname = "root"; entry.pax_headers.pop("gid", None)
                content = before.extractfile(member) if member.isfile() else None
                after.addfile(entry, content)
                if content is not None:
                    content.close()
        diff_id = "sha256:" + hashed.hash.hexdigest()
    if changes:
        digest = "sha256:" + sha(temp)
        descriptor = {**layer, "digest": digest, "size": temp.stat().st_size}
        temp.replace(STORE / "blobs/sha256" / digest[7:])
    else:
        temp.unlink()
        descriptor = layer
        diff_id = None
    result = {"original": layer, "derived": descriptor, "diff_id": diff_id, "ownership_changes": changes,
              "transformation": "Only UID/GID >=65536 become root; file contents and runtime config unchanged"}
    write(metadata, result)
    print("rootless layer", layer["digest"][7:19], "ownership changes", len(changes), flush=True)
    return result


def normalize_rootless(wait, skip_missing=False):
    rows = read(DATA / "instances.json")
    targets = [r for r in rows if r["repo"] == "matplotlib/matplotlib"]
    changed_tasks = []
    unavailable_tasks = []
    for row in targets:
        iid = row["instance_id"]
        p = STORE / "images" / (iid + ".json")
        record = read(p)
        if record.get("rootless_derivation"):
            changed_tasks.append(iid)
            continue
        if skip_missing and any(not (STORE / "blobs/sha256" / layer["digest"][7:]).exists()
                                for layer in record["layers"][-2:]):
            unavailable_tasks.append(iid)
            print("rootless skipped incomplete task", iid, flush=True)
            continue
        original_record = copy.deepcopy(record)
        config_path = STORE / "blobs/sha256" / record["config"]["digest"][7:]
        config = read(config_path)
        # qhull ownership is in the task/repository build layer, not Ubuntu/conda.
        # Audit the final two layers (including tiny final metadata when present).
        changed = []
        for index in range(max(0, len(record["layers"]) - 2), len(record["layers"])):
            result = rootless_layer(record["layers"][index], wait)
            if result["ownership_changes"]:
                record["layers"][index] = result["derived"]
                config["rootfs"]["diff_ids"][index] = result["diff_id"]
                changed.append(result)
        if not changed:
            continue
        archive = STORE / "original-images" / (iid + ".json")
        write(archive, original_record)
        raw = json.dumps(config, separators=(",", ":")).encode()
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        (STORE / "blobs/sha256" / digest[7:]).write_bytes(raw)
        record["config"] = {**record["config"], "digest": digest, "size": len(raw)}
        m = read(STORE / "blobs/sha256" / original_record["manifest_digest"][7:])
        m["config"] = record["config"]
        m["layers"] = record["layers"]
        raw = json.dumps(m, separators=(",", ":")).encode()
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        (STORE / "blobs/sha256" / digest[7:]).write_bytes(raw)
        record["manifest_digest"] = digest
        record["image_id"] = digest
        record["manifest_size"] = len(raw)
        record["requested"] = "tact-verified200-rootless/" + iid.replace("__", "-").lower() + ":v1"
        record["repo_digests"] = []
        record["compressed_bytes"] = sum(l["size"] for l in record["layers"])
        record["rootless_derivation"] = {"original_image_id": original_record["image_id"], "original_manifest_digest": original_record["manifest_digest"], "runtime_config_unchanged": True, "layers": changed}
        write(p, record)
        changed_tasks.append(iid)
    # Publish references only after all normalization products are complete.
    for row in rows:
        p = STORE / "images" / (row["instance_id"] + ".json")
        record = read(p)
        row["image_name"] = record["requested"]
        row["nas_image"]["manifest_sha256"] = sha(p)
    write(DATA / "instances.json", rows)
    (STORE / "dataset/instances.json").write_bytes((DATA / "instances.json").read_bytes())
    write(STORE / "rootless/summary.json", {"audited_tasks": len(targets) - len(unavailable_tasks), "unavailable_tasks": unavailable_tasks, "changed_tasks": changed_tasks, "original_blobs_retained": True, "network_bytes": 0})


def download_locked(proxy, workers):
    plan = read(STORE / "download-plan.json")
    state_lock = threading.Lock()
    journal = STORE / "download-events.jsonl"
    start = time.time()
    counters = {"complete_layers": 0, "received_bytes_this_process": 0, "errors": [], "pid": os.getpid(), "started_at": start}

    def event(data):
        with state_lock:
            with journal.open("a") as f:
                f.write(json.dumps({"at": time.time(), **data}) + "\n")
            write(STORE / "download-status.json", {**counters, "elapsed_seconds": time.time() - start, "total_layers": len(plan["layers"]), "complete": False})

    def one(layer):
        digest = layer["digest"]
        target = STORE / "blobs/sha256" / digest[7:]
        if target.exists() and target.stat().st_size == layer["size"]:
            if sha(target) != digest[7:]:
                raise ValueError(f"Corrupt stored layer: {digest}")
            with state_lock:
                counters["complete_layers"] += 1
            event({"event": "cached", "digest": digest})
            return
        part = target.with_suffix(".part")
        registry = Registry(proxy)
        total_received = 0
        last_progress = time.monotonic()
        for attempt in range(10):
            try:
                offset = part.stat().st_size if part.exists() else 0
                if offset > layer["size"]:
                    part.unlink(); offset = 0
                if offset < layer["size"]:
                    r = registry.get(layer["repository"], "blobs/" + digest, stream=True, extra={"Range": f"bytes={offset}-"} if offset else None)
                    if offset and r.status_code != 206:
                        offset = 0
                    with r, part.open("ab" if offset else "wb") as f:
                        for block in r.iter_content(1024 * 1024):
                            f.write(block)
                            total_received += len(block)
                            with state_lock:
                                counters["received_bytes_this_process"] += len(block)
                            if time.monotonic() - last_progress > 10:
                                event({"event": "progress", "digest": digest, "received_bytes": total_received})
                                last_progress = time.monotonic()
                        f.flush(); os.fsync(f.fileno())
                if part.stat().st_size != layer["size"] or sha(part) != digest[7:]:
                    part.unlink()
                    raise ValueError("Layer checksum/size mismatch")
                part.replace(target)
                with state_lock:
                    counters["complete_layers"] += 1
                event({"event": "downloaded", "digest": digest, "expected_bytes": layer["size"], "received_bytes": total_received})
                print("layer", digest[7:19], counters["complete_layers"], "/", len(plan["layers"]), "received GB", round(counters["received_bytes_this_process"] / 1e9, 3), flush=True)
                return
            except Exception as exc:
                event({"event": "retry", "digest": digest, "attempt": attempt, "error": str(exc), "received_bytes": total_received})
                if attempt == 9:
                    raise
                time.sleep(min(30, 2**attempt))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        smoke_ids = ["pytest-dev__pytest-10051", "django__django-11179", "matplotlib__matplotlib-20676"]
        priority = {l["digest"] for iid in smoke_ids for l in read(STORE / "images" / (iid + ".json"))["layers"]}
        futures = {pool.submit(one, layer): layer for layer in sorted(plan["layers"], key=lambda x: (x["digest"] not in priority, x["size"]))}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                with state_lock:
                    counters["errors"].append({"digest": futures[future]["digest"], "error": str(exc)})
    write(STORE / "download-status.json", {**counters, "elapsed_seconds": time.time() - start, "total_layers": len(plan["layers"]), "complete": not counters["errors"] and counters["complete_layers"] == len(plan["layers"])})
    if counters["errors"]:
        raise RuntimeError(counters["errors"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "manifests", "seed-local", "download", "normalize-rootless"])
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wait", action="store_true", help="Wait for rootless build layers to finish downloading")
    parser.add_argument("--skip-missing", action="store_true", help="Skip unavailable rootless layers after an interrupted download")
    args = parser.parse_args()
    if args.stage == "select":
        select()
    elif args.stage == "manifests":
        manifests(args.proxy, args.workers)
    elif args.stage == "seed-local":
        seed_local()
    elif args.stage == "normalize-rootless":
        normalize_rootless(args.wait, args.skip_missing)
    else:
        download(args.proxy, args.workers)


if __name__ == "__main__":
    main()
