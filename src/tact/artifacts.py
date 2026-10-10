from __future__ import annotations

import hashlib
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read_json(path: Path) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".write-", delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        tmp = Path(f.name)
    tmp.replace(path)


def hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, default=str).encode()).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,150}", value):
        raise ValueError(f"Invalid artifact/run/task ID: {value!r}")
    return value


def project_path(root: Path, value: str | Path, *, exists: bool = True) -> Path:
    root = Path(root).resolve()
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Path leaves project: {path}")
    relative = path.absolute().relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Project assets must be physical copies: {current}")
    if exists and not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def selected_trajectory(run: Path, iid: str) -> Path | None:
    task = run / "tasks" / identifier(iid)
    result = task / "result.json"
    if not result.exists():
        return None
    attempt = identifier(read_json(result)["attempt_id"])
    p = task / "attempts" / attempt / f"{iid}.traj.json"
    return p if p.is_file() else None


def update_status(path: Path, stage: str, **values) -> None:
    write_json(path, {"stage": stage, "updated_at": now(), **values})
