from __future__ import annotations

import argparse
import json
from pathlib import Path

from tact.artifacts import identifier, project_path, read_json, write_json
from tact.config import load_resolved, resolve_config, root_path


def main():
    parser = argparse.ArgumentParser(prog="tact")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="dotted.key=value")
    sub = parser.add_subparsers(dest="command", required=True)
    config = sub.add_parser("config")
    config.add_argument("action", choices=["show", "check"])
    for command in ["run", "rollout", "gold", "environment-check"]:
        p = sub.add_parser(command)
        p.add_argument("--run-id", required=True)
        if command == "run":
            p.add_argument("--no-evaluate", action="store_true")
        if command == "rollout":
            p.add_argument("--resume-run", action="store_true")
    p = sub.add_parser("import-run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--source", type=Path, required=True)
    for command in ["evaluate", "report"]:
        p = sub.add_parser(command)
        p.add_argument("--run-id", required=True)
        if command == "evaluate":
            p.add_argument("--eval-id", required=True)
    p = sub.add_parser("compare")
    p.add_argument("first")
    p.add_argument("second")
    p = sub.add_parser("annotation-export")
    p.add_argument("--run-id", required=True)
    p.add_argument("--label-set", required=True)
    p = sub.add_parser("annotation-import")
    p.add_argument("--label-set", required=True)
    p.add_argument("--source", type=Path, required=True)
    for command in ["decision-points", "latent", "fit", "plot"]:
        p = sub.add_parser(command)
        p.add_argument("--run-id", required=True)
        if command in ["decision-points", "latent", "fit"]:
            p.add_argument("--set-id", required=True)
        if command == "fit":
            p.add_argument("--label-set", required=True)
        if command in ["fit", "plot"]:
            p.add_argument("--bundle-id", required=True)
    p = sub.add_parser("assets-check")
    p.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "assets-check":
            verify_assets(args.deep)
            return
        if args.command == "compare":
            from tact.reporting.metrics import compare_runs
            result = compare_runs(root_path() / "runs" / identifier(args.first),
                                  root_path() / "runs" / identifier(args.second))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.command == "annotation-import":
            from tact.research.annotation import import_labels
            import_labels(root_path(), args.label_set, args.source)
            return
        existing = args.command in {"evaluate", "report", "annotation-export", "decision-points", "latent", "fit", "plot"} or (
            args.command == "rollout" and args.resume_run)
        if existing:
            if args.config or args.set:
                raise ValueError("Existing runs use their immutable resolved config; overrides require a new run")
            run = project_path(root_path(), "runs/" + identifier(args.run_id))
            cfg = load_resolved(run / "resolved-config.json")
        else:
            cfg = resolve_config(args.config, args.set)
            if args.command == "config":
                print(json.dumps(cfg if args.action == "show" else {"valid": True,
                    "tasks": len(cfg["selected_ids"]), "model": cfg["model_descriptor"],
                    "sources": cfg["sources"],
                    "config_hash": cfg["config_hash"]}, ensure_ascii=False, indent=2))
                return
            if args.command == "import-run":
                from tact.research.imports import import_trajectories
                print(import_trajectories(cfg, args.source, args.run_id))
                return
            from tact.rollout.runner import prepare_run
            run = prepare_run(cfg, args.run_id)
        from tact.pipeline import pipeline, run_stage
        if args.command == "run":
            success = pipeline(cfg, run, evaluate=not args.no_evaluate)
            raise SystemExit(0 if success else 1)
        if args.command == "annotation-export":
            from tact.research.annotation import export_inputs
            print(export_inputs(cfg, run, args.label_set))
            return
        stage = "evaluate" if args.command == "gold" else args.command
        extra = []
        if args.command == "gold":
            extra = ["--gold", "--eval-id", "gold"]
        elif args.command == "evaluate":
            extra = ["--eval-id", identifier(args.eval_id)]
        for field in ["set_id", "label_set", "bundle_id"]:
            value = getattr(args, field, None)
            if value:
                extra += ["--" + field.replace("_", "-"), identifier(value)]
        code = run_stage(cfg, run, stage, extra)
        print(f"stage={stage} exit={code}; logs={run / 'logs' / (stage + '.log')}")
        raise SystemExit(code)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


def verify_assets(deep: bool):
    from tact.artifacts import sha256
    root = root_path()
    manifest = read_json(root / "imports/asset-migration.json")
    for entry in manifest["files"]:
        path = project_path(root, entry["destination"])
        if path.stat().st_nlink != 1:
            raise ValueError(f"Hardlinked asset: {path}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Asset size mismatch: {path}")
        if deep and sha256(path) != entry["sha256"]:
            raise ValueError(f"Asset hash mismatch: {path}")
    write_json(root / "imports/asset-verification.json", {"complete": True,
               "deep": deep, "files": len(manifest["files"])})
    print(f"Verified {len(manifest['files'])} physically copied assets (deep={deep})")
