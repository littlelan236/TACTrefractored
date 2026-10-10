from __future__ import annotations

import json
from pathlib import Path

from tact.artifacts import identifier, now, read_json, selected_trajectory, sha256, write_json
from tact.config import select_instances
from tact.research.trajectories import TrajectoryReader

LABELS = {"OVERTHINKING", "OVERACTING", "CALIBRATED"}


def export_inputs(cfg: dict, run: Path, label_set: str) -> Path:
    output = Path(cfg["project_root"]) / "artifacts/labels" / identifier(label_set)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows, partitions = select_instances(cfg)
    membership = {iid: part for part, ids in partitions.items() for iid in ids}
    excluded = set(read_json(Path(cfg["research"]["fit_exclusions"]))["instance_ids"])
    records = []
    for row in rows:
        iid = row["instance_id"]
        if membership[iid] not in {"axis-train", "axis-val"} or iid in excluded:
            continue
        source = selected_trajectory(run, iid)
        if source is None:
            raise ValueError(f"Missing authoritative trajectory: {iid}")
        reader = TrajectoryReader(source)
        target = output / "inputs" / iid
        target.mkdir(parents=True)
        steps = [{k: s[k] for k in ["step", "message_index", "reasoning", "content", "tool_calls", "observations"]}
                 for s in reader.steps()]
        (target / "task.txt").write_text(row["problem_statement"], encoding="utf-8")
        (target / "steps.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in steps))
        width = cfg["research"]["annotation_chunk_steps"]
        for start in range(0, len(steps), width):
            blocks = []
            for s in steps[start:start + width]:
                blocks.append(f"### Step {s['step']}\n\nREASONING\n{s['reasoning']}\n\n"
                              f"ACTION\n{s['content']}\n{json.dumps(s['tool_calls'], ensure_ascii=False)}\n\n"
                              f"OBSERVATION\n{json.dumps(s['observations'], ensure_ascii=False)}\n")
            (target / f"chunk_{start // width:03d}.md").write_text("\n".join(blocks), encoding="utf-8")
        record = {"instance_id": iid, "partition": membership[iid], "n_steps": len(steps),
                  "source_traj": str(source.relative_to(Path(cfg["project_root"]))),
                  "source_sha256": sha256(source)}
        write_json(target / "manifest.json", record)
        records.append(record)
    write_json(output / "manifest.json", {"schema_version": 1, "label_set": label_set,
        "run_id": run.name, "model": cfg["model_descriptor"], "created_at": now(),
        "exclude_step_zero": cfg["research"]["exclude_step_zero_from_fit"], "tasks": records})
    return output


def import_labels(root: Path, label_set: str, source: Path):
    output = root / "artifacts/labels" / identifier(label_set)
    manifest = read_json(output / "manifest.json")
    destination = output / "labels"
    if destination.exists():
        raise FileExistsError(destination)
    validated = []
    # Validate every file before committing any imports.
    for task in manifest["tasks"]:
        iid = task["instance_id"]
        file = source / (iid + ".jsonl")
        if not file.is_file():
            raise ValueError(f"Missing labels: {iid}")
        rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        first = 1 if manifest["exclude_step_zero"] else 0
        expected = set(range(first, task["n_steps"]))
        seen = set()
        for row in rows:
            step = row.get("step")
            if row.get("instance_id") != iid or type(step) is not int or step in seen or step not in expected:
                raise ValueError(f"Invalid/duplicate label step: {iid} {step}")
            if row.get("label") not in LABELS or row.get("confidence") not in {"high", "medium", "low"}:
                raise ValueError(f"Invalid label/confidence: {iid} {step}")
            if row.get("source_sha256") != task["source_sha256"]:
                raise ValueError(f"Labels must bind exact source_sha256: {iid}")
            seen.add(step)
        if seen != expected:
            raise ValueError(f"Incomplete label coverage: {iid}")
        validated.append((file, rows))
    destination.mkdir()
    for file, rows in validated:
        (destination / file.name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_json(output / "labels-provenance.json", {"imported_at": now(), "source": str(source),
               "files": {p.name: sha256(p) for p in destination.iterdir()}})
