"""Re-evaluate unsuccessful NAS gold tasks and publish a provenance-aware aggregate."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from run_nas200_gold_proxy import install_metadata_proxy
from tact.artifacts import hash_json, identifier, now, read_json, write_json
from tact.config import load_resolved, select_instances
from tact.evaluation.runner import evaluate


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="verified200-nas-repair-20261015")
    parser.add_argument("--eval-id", default="gold-repair-v1")
    parser.add_argument("--source-run", default="verified200-nas-gold-20261015")
    parser.add_argument("--source-eval", default="gold-proxy-20261015")
    parser.add_argument("--previous-combined", type=Path)
    parser.add_argument("--also", action="append", default=[])
    args = parser.parse_args()
    source = ROOT / "runs" / identifier(args.source_run)
    previous_path = source / "evaluation" / identifier(args.source_eval) / "summary.json"
    if args.previous_combined:
        previous_path = args.previous_combined.resolve()
    previous = read_json(previous_path)
    selected = [iid for iid, result in previous["outcomes"].items() if not result["resolved"]]
    selected = list(dict.fromkeys(selected + args.also))
    cfg = load_resolved(source / "resolved-config.json")
    rows, _ = select_instances(cfg)
    assert len(rows) == 200 and all(r.get("nas_image") for r in rows)
    assert set(previous["outcomes"]) == {r["instance_id"] for r in rows}
    cfg["benchmark"]["selection"] = selected
    cfg["config_hash"] = hash_json({k: v for k, v in cfg.items() if k != "config_hash"})
    run = ROOT / "runs" / identifier(args.run_id)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "resolved-config.json").exists():
        assert read_json(run / "resolved-config.json")["config_hash"] == cfg["config_hash"]
    else:
        write_json(run / "resolved-config.json", cfg)
        write_json(run / "images.json", read_json(source / "images.json"))
    log = ROOT / "logs/nas200" / run.name / identifier(args.eval_id)
    log.mkdir(parents=True, exist_ok=True)
    write_json(log / "status.json", {"stage": "evaluate", "tasks": len(selected), "at": now()})
    install_metadata_proxy("http://127.0.0.1:7890",
        Path(cfg["environment"]["requirements_cache"]) / "nas200-proxy-http")
    try:
        result = evaluate(cfg, run, identifier(args.eval_id), gold=True)
        outcomes = {**previous["outcomes"], **result["outcomes"]}
        summary = {"requested": len(rows), "resolved": sum(r["resolved"] for r in outcomes.values()),
                   "graded": sum(r["completed"] for r in outcomes.values()),
                   "complete": len(outcomes) == len(rows) and all(r["resolved"] for r in outcomes.values()),
                   "sources": {iid: str(run / "evaluation" / args.eval_id / "summary.json")
                               if iid in result["outcomes"] else previous.get("sources", {}).get(iid, str(previous_path))
                               for iid in outcomes},
                   "outcomes": outcomes, "finished_at": now()}
        write_json(log / "combined-summary.json", summary)
        write_json(log / "status.json", {"stage": "finished", "tasks": len(selected),
            "subset_resolved": result["resolved"], "total_resolved": summary["resolved"], "at": now()})
        print({k: v for k, v in summary.items() if k not in ["sources", "outcomes"]}, flush=True)
    except BaseException as exc:
        write_json(log / "status.json", {"stage": "failed", "error": str(exc), "at": now()})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
