"""Prepare the frozen, diverse Verified-70 cohort for the existing TACT runners.

Run with the project's runner interpreter. Task containers still belong to
ContainerFactory; this script only provisions their official images.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/swebench/verified70"
DIFFICULTIES = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_metrics(patch):
    paths = re.findall(r"^diff --git a/(.*?) b/", patch, re.MULTILINE)
    added = sum(line.startswith("+") and not line.startswith("+++") for line in patch.splitlines())
    deleted = sum(line.startswith("-") and not line.startswith("---") for line in patch.splitlines())
    return {"files": paths, "file_count": len(paths), "added_lines": added,
            "deleted_lines": deleted, "changed_lines": added + deleted,
            "hunks": len(re.findall(r"^@@ ", patch, re.MULTILINE))}


def generate():
    if (ROOT / "imports/repartition-20261011-before").exists():
        raise RuntimeError("Active splits were migrated; generate would overwrite the reviewed exclusions. Use list/verify/pull-images instead.")
    source = read(DATA / "source-500.json")
    provenance = read(DATA / "source-provenance.json")
    choices = read(DATA / "selection.json")
    assert len(source) == len({r["instance_id"] for r in source}) == 500
    by_id = {r["instance_id"]: r for r in source}
    selected = choices["tasks"]
    assert len(selected) == len({r["instance_id"] for r in selected}) == 70
    assert set(Counter(r["category"] for r in selected).values()) == {7}
    original = {r["instance_id"] for r in read(ROOT / "data/swebench/instances.json")}
    assert not original.intersection(r["instance_id"] for r in selected)
    existing = {r["instance_id"]: r for r in read(DATA / "instances.json")} if (DATA / "instances.json").exists() else {}
    rows, catalog = [], []
    for choice in selected:
        iid = choice["instance_id"]
        row = dict(by_id[iid])
        assert row["difficulty"] in DIFFICULTIES
        assert row["patch"] and row["test_patch"] and row["base_commit"]
        assert isinstance(json.loads(row["FAIL_TO_PASS"]), list)
        row["source_dataset"] = provenance["dataset"] + ":test"
        row["source_revision"] = provenance["revision"]
        row["docker_image"] = (
            "swebench/sweb.eval.x86_64." + iid.replace("__", "_1776_") + ":latest"
        ).lower()
        for key in ("docker_image", "official_docker_image", "image_derivation"):
            if key in existing.get(iid, {}):
                row[key] = existing[iid][key]
        rows.append(row)
        catalog.append({**choice, "repo": row["repo"], "difficulty": row["difficulty"],
                        "issue_title": row["problem_statement"].strip().splitlines()[0],
                        "base_commit": row["base_commit"], "gold_patch": patch_metrics(row["patch"]),
                        "fail_to_pass_count": len(json.loads(row["FAIL_TO_PASS"])),
                        "docker_image": row["docker_image"]})
    write(DATA / "instances.json", rows)
    write(DATA / "splits.json", {"schema_version": 1, "partitions": {
        "axis-train": [], "axis-val": [], "eval": [r["instance_id"] for r in rows]},
        "source_policy": "New held-out Verified subset, disjoint from the original 196 tasks. "
                         "No axis fitting or hyperparameter selection on these tasks."})
    write(DATA / "exclusions.json", {"schema_version": 1, "excluded": [], "reasons": {}})
    write(DATA / "fit-exclusions.json", {"schema_version": 1, "instance_ids": [],
          "reason": "All 70 tasks are held out; no labels or fitted axes imported for this cohort."})
    write(DATA / "catalog.json", {"schema_version": 1, "tasks": catalog,
          "categories": choices["categories"], "difficulty_source": "official Verified human fix-time labels",
          "gold_trajectory_steps": None,
          "difficulty_note": "Fix-time labels are not agent trajectory lengths. Gold patch size is "
                             "supplementary implementation-scope evidence, not a measured difficulty score."})
    write(DATA / "manifest.json", {"schema_version": 1, "source": provenance,
          "count": 70, "category_counts": dict(Counter(r["category"] for r in catalog)),
          "repo_counts": dict(Counter(r["repo"] for r in rows)),
          "difficulty_counts": dict(Counter(r["difficulty"] for r in rows)),
          "selection_method": "Curated functional categories plus official fix-time labels and "
                              "gold-patch scope; no model performance or gold test outcomes used to sample.",
          "official_gt4h_tasks": [r["instance_id"] for r in rows if r["difficulty"] == ">4 hours"],
          "overlap_original_instances": [],
          "file_hashes": {name: sha(DATA / name) for name in [
              "source-500.json", "source-provenance.json", "selection.json", "instances.json",
              "splits.json", "exclusions.json", "fit-exclusions.json", "catalog.json"]}})
    with (DATA / "catalog.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_id", "category", "category_zh", "repo", "official_difficulty",
                         "gold_files", "gold_changed_lines", "gold_hunks", "issue_title", "selection_reason"])
        for r in catalog:
            writer.writerow([r["instance_id"], r["category"], r["category_zh"], r["repo"], r["difficulty"],
                             r["gold_patch"]["file_count"], r["gold_patch"]["changed_lines"],
                             r["gold_patch"]["hunks"], r["issue_title"], r["selection_reason"]])
    overlay = DATA / "benchmark.yaml"
    expected = """# Dataset-only overlay; runtime/model/agent/environment defaults remain canonical.
benchmark:
  instances: data/swebench/verified70/instances.json
  splits: data/swebench/verified70/splits.json
  exclusions: data/swebench/verified70/exclusions.json
  selection: eval
research:
  fit_exclusions: data/swebench/verified70/fit-exclusions.json
"""
    if overlay.exists() and overlay.read_text() != expected:
        raise ValueError("benchmark.yaml has local edits; refusing to overwrite")
    overlay.write_text(expected)
    table = ["# SWE-bench Verified-70", "", "来源：官方 `princeton-nlp/SWE-bench_Verified` test，500 条中的固定 70 条。",
             f"Revision：`{provenance['revision']}`。筛选日期：2026-10-11。", "",
             "10 类 × 7 条，覆盖官方四档修复耗时；与原工程 196 条任务无重叠。",
             "这些是人工审定的修复耗时标签，不是 gold agent 的轨迹步数。",
             "gold patch 的文件数、增删行数及每条选入理由在 catalog.json / catalog.csv。",
             "类型是本项目按问题描述和改动位置人工划分的功能类别，不是 SWE-bench 官方类型标签。", "",
             "该集合全部放在 eval；不能用于拟合轴或调超参数。选择不以模型得分或 gold 是否通过为条件。",
             "绘图类的官方难度只有前两档，不人为把大 patch 改标为 1–4h；其他类型覆盖更难档。", "",
             "| 类型 | 数量 | <15 min | 15 min–1h | 1–4h | >4h |", "|---|---:|---:|---:|---:|---:|"]
    for category, label in choices["categories"].items():
        counts = Counter(r["difficulty"] for r in catalog if r["category"] == category)
        table.append("| " + label + " | 7 | " + " | ".join(str(counts[d]) for d in DIFFICULTIES) + " |")
    table += ["", "## 接入方式", "", "使用已有 CLI 和 runner，无第二套容器或评测入口。", "", "```bash",
              ".venvs/runner/bin/tact --config data/swebench/verified70/benchmark.yaml config check",
              ".venvs/runner/bin/python scripts/prepare_verified70.py pull-images --workers 4",
              ".venvs/runner/bin/python scripts/prepare_verified70.py verify",
              ".venvs/runner/bin/tact --config data/swebench/verified70/benchmark.yaml gold --run-id verified70-gold",
              ".venvs/runner/bin/tact --config data/swebench/verified70/benchmark.yaml run --run-id verified70-baseline", "```", "",
              "按类型/难度查看任务：", "", "```bash",
              ".venvs/runner/bin/python scripts/prepare_verified70.py list --category orm_database",
              ".venvs/runner/bin/python scripts/prepare_verified70.py list --difficulty '1-4 hours'", "```", "",
              "子集运行可使用 CLI 的 `--set 'benchmark.selection=[id1,id2]'`，其余配置继承 canonical experiment.yaml。",
              "`source-500.json`/parquet 是固定 revision 的原始本地快照；运行只消费选定的 instances.json。",
              "下载镜像记录 image ID 与 digest，实际每次运行继续由 ContainerFactory 锁定镜像。",
              "`image-provisioning.json`、`verification.json` 和 gold run 的 summary.json 区分下载、配置检查与测试通过。",
              "如 rootless Docker 无法解包官方 Matplotlib 镜像的高 UID/GID 文件，可运行",
              "`scripts/prepare_verified70_rootless_images.py` 生成仅规范化超出映射范围 ownership 的实体派生镜像。",
              "原始 layer digest、逐文件 ownership 改动与 runtime config 对照保存在 assets/images/verified70-rootless。", "",
              "## 任务明细", "", "| 类型 | Task ID | 官方修复耗时 | Gold 文件/行数 | 选入理由 |",
              "|---|---|---|---:|---|"]
    for r in catalog:
        table.append(f"| {r['category_zh']} | {r['instance_id']} | {r['difficulty']} | "
                     f"{r['gold_patch']['file_count']} / {r['gold_patch']['changed_lines']} | {r['selection_reason']} |")
    (DATA / "README.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(read(DATA / "manifest.json"), ensure_ascii=False, indent=2))


def provision(workers):
    import docker
    import requests

    from tact.config import resolve_config

    cfg = resolve_config(DATA / "benchmark.yaml")
    rows = read(DATA / "instances.json")
    output = DATA / "image-provisioning.json"
    report = {"schema_version": 1, "tasks": {}}

    def one(row):
        client = docker.DockerClient(base_url=cfg["environment"]["docker_host"], timeout=600)
        iid, name = row["instance_id"], row["docker_image"]
        try:
            try:
                image = client.images.get(name)
                status = "existing"
            except docker.errors.ImageNotFound:
                repository, tag = name.rsplit(":", 1)
                for event in client.api.pull(repository, tag=tag, stream=True, decode=True):
                    if "error" in event:
                        raise docker.errors.DockerException(event["error"])
                image = client.images.get(name)
                status = "pulled"
            return iid, {"status": status, "requested": name, "image_id": image.id,
                         "repo_digests": image.attrs.get("RepoDigests", []), "bytes": image.attrs["Size"]}
        except (docker.errors.DockerException, requests.RequestException) as exc:
            return iid, {"status": "failed", "requested": name, "error": str(exc)}
        finally:
            client.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            iid, result = future.result()
            report["tasks"][iid] = result
            report["updated_at"] = datetime.now(UTC).isoformat()
            write(output, report)
            print(iid, result["status"], result.get("error", ""), flush=True)
    report["complete"] = all(r["status"] != "failed" for r in report["tasks"].values())
    write(output, report)
    return report["complete"]


def verify():
    from tact.config import resolve_config, select_instances
    from tact.environments.containers import ContainerFactory

    manifest = read(DATA / "manifest.json")
    for name, expected in manifest["file_hashes"].items():
        assert sha(DATA / name) == expected, name
    cfg = resolve_config(DATA / "benchmark.yaml")
    rows, partitions = select_instances(cfg)
    assert len(rows) in (67, 70) and not partitions["axis-train"] and not partitions["axis-val"]
    factory = ContainerFactory(cfg)
    try:
        images = factory.lock_images(rows)
    finally:
        factory.client.close()
    value = {"complete": True, "tasks": len(rows), "config_hash": cfg["config_hash"],
             "manifest_sha256": sha(DATA / "manifest.json"), "images": images,
             "checked_at": datetime.now(UTC).isoformat(),
             "checks": ["pinned source membership", "file hashes", "70 unique tasks", "10x7 categories",
                        "disjoint original 196", "held-out partitions", "canonical config resolver",
                        "ContainerFactory locks all task images (including documented rootless derivatives)"],
             "gold_validation": "See reports/verified70/gold-validation.json"}
    write(DATA / "verification.json", value)
    print(json.dumps({k: v for k, v in value.items() if k != "images"}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "pull-images", "verify", "list"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--category")
    parser.add_argument("--difficulty", choices=DIFFICULTIES)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    if args.action == "generate":
        generate()
    elif args.action == "pull-images":
        raise SystemExit(0 if provision(args.workers) else 1)
    elif args.action == "verify":
        verify()
    else:
        catalog = read(DATA / "catalog.json")
        if args.category and args.category not in catalog["categories"]:
            parser.error("Unknown category")
        for row in catalog["tasks"]:
            if args.category and row["category"] != args.category:
                continue
            if args.difficulty and row["difficulty"] != args.difficulty:
                continue
            print(row["instance_id"], row["category"], row["difficulty"], sep="\t")


if __name__ == "__main__":
    main()
