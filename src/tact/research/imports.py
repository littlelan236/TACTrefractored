from __future__ import annotations

import shutil
from pathlib import Path

from tact.artifacts import now, read_json, sha256, write_json
from tact.config import select_instances
from tact.rollout.runner import generate_predictions, prepare_run


def import_trajectories(cfg: dict, source: Path, run_id: str) -> Path:
    """Copy historical native trajectories; old paths remain provenance text only."""
    rows, _ = select_instances(cfg)
    sources = {}
    for row in rows:
        iid = row["instance_id"]
        candidates = [source / iid / f"{iid}.traj.json", source / f"{iid}.traj.json"]
        files = [p for p in candidates if p.is_file()]
        if len(files) != 1:
            raise ValueError(f"Expected one source trajectory for {iid}")
        data = read_json(files[0])
        if data.get("instance_id") != iid or "messages" not in data:
            raise ValueError(f"Invalid source trajectory: {iid}")
        name = data.get("info", {}).get("config", {}).get("model", {}).get("model_name", "")
        if name.removeprefix("openai/") != Path(cfg["model"]["path"]).name:
            raise ValueError(f"Historical trajectory model does not match selected checkpoint: {name}")
        sources[iid] = (files[0], data)
    run = prepare_run(cfg, run_id)
    provenance = []
    for iid, (file, data) in sources.items():
        output = run / "tasks" / iid / "attempts/imported-000"
        output.mkdir(parents=True)
        target = output / f"{iid}.traj.json"
        shutil.copyfile(file.resolve(), target)
        info = data.get("info", {})
        result = {"instance_id": iid, "attempt_id": "imported-000",
                  "exit_status": info.get("exit_status", "Unknown"),
                  "submission": info.get("submission", ""), "historical_import": True}
        write_json(output / "attempt.json", result)
        write_json(output.parent.parent / "result.json", result)
        provenance.append({"instance_id": iid, "source": str(file), "sha256": sha256(target)})
    write_json(run / "import-provenance.json", {"imported_at": now(), "source": str(source),
               "files": provenance, "historical_runtime": True,
               "identity_assurance": "source model name plus physically verified unchanged checkpoint assets; not original runtime attestation"})
    generate_predictions(run, cfg)
    write_json(run / "status.json", {"stage": "imported", "tasks": len(rows), "updated_at": now()})
    return run
