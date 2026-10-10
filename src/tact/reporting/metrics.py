from __future__ import annotations

from collections import Counter
from pathlib import Path

from tact.artifacts import now, read_json, selected_trajectory, write_json
from tact.research.trajectories import TrajectoryReader


def report(cfg: dict, run: Path, eval_id: str | None = None) -> dict:
    evaluations = sorted((run / "evaluation").glob("*/summary.json"))
    summary_path = run / "evaluation" / eval_id / "summary.json" if eval_id else (evaluations[-1] if evaluations else None)
    evaluation = read_json(summary_path) if summary_path and summary_path.exists() else {}
    tasks = []
    for iid in cfg["selected_ids"]:
        source = selected_trajectory(run, iid)
        result_path = run / "tasks" / iid / "result.json"
        result = read_json(result_path) if result_path.exists() else {}
        final = TrajectoryReader(source) if source else None
        usage = final.usage() if final else None
        attempts = list((run / "tasks" / iid / "attempts").glob("*/*.traj.json"))
        cumulative = [TrajectoryReader(p).usage() for p in attempts]
        tasks.append({"instance_id": iid, "exit_status": result.get("exit_status", "pending"),
                      "selected_attempt": result.get("attempt_id"),
                      "steps": sum(1 for _ in final.steps()) if final else 0,
                      "tool_calls": sum(len(s["tool_calls"]) for s in final.steps()) if final else 0,
                      "final_attempt_usage": usage,
                      "all_attempt_tokens": sum(s["total_tokens"] for s in cumulative),
                      "all_attempt_response_coverage_complete": bool(cumulative) and all(s["response_coverage_complete"] for s in cumulative),
                      "evaluation": evaluation.get("outcomes", {}).get(iid)})
    result = {"run_id": run.name, "updated_at": now(), "config_hash": cfg["config_hash"],
              "model": cfg["model_descriptor"], "selected_tasks": len(tasks),
              "statuses": dict(Counter(t["exit_status"] for t in tasks)),
              "final_attempt_tokens": sum((t["final_attempt_usage"] or {}).get("total_tokens", 0) for t in tasks),
              "all_attempt_tokens": sum(t["all_attempt_tokens"] for t in tasks),
              "evaluation": evaluation, "tasks": tasks,
              "definitions": {"resolve_rate": "resolved / all selected tasks, including missing patches",
                "tokens": "server-reported persisted responses; unreturned HTTP retries unknown",
                "steps": "assistant messages in the selected authoritative attempt"}}
    output = run / "reports"
    write_json(output / "summary.json", result)
    text = [f"# {run.name}", "", f"Selected tasks: {len(tasks)}", "",
            f"Final-attempt tokens: {result['final_attempt_tokens']:,}",
            f"All-attempt tokens: {result['all_attempt_tokens']:,}", "",
            "Token counts cover persisted responses; unreturned HTTP retries remain unknown.", ""]
    if evaluation:
        text += [f"Resolved: {evaluation['resolved']}/{evaluation['requested']}",
                 f"Graded: {evaluation['graded']}/{evaluation['requested']}", ""]
    text += ["| Task | Exit | Steps | Tool calls | Evaluation |", "|---|---|---:|---:|---|"]
    for task in tasks:
        text.append(f"| {task['instance_id']} | {task['exit_status']} | {task['steps']} | {task['tool_calls']} | "
                    f"{(task['evaluation'] or {}).get('outcome', 'pending')} |")
    (output / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return result


def compare_runs(first: Path, second: Path) -> dict:
    a, b = read_json(first / "resolved-config.json"), read_json(second / "resolved-config.json")
    for key in ["model_descriptor", "input_hashes", "selected_ids", "environment", "timeouts",
                "generation", "agent", "execution"]:
        if a[key] != b[key]:
            raise ValueError(f"Runs are not matched: {key}")
    for key in ["backend", "tensor_parallel_size", "gpus", "max_num_seqs", "enable_prefix_caching"]:
        if a["serving"][key] != b["serving"][key]:
            raise ValueError(f"Serving mismatch: {key}")
    for run in [first, second]:
        if not (run / "reports/summary.json").is_file():
            report(read_json(run / "resolved-config.json"), run)
    left, right = read_json(first / "reports/summary.json"), read_json(second / "reports/summary.json")
    return {"first": first.name, "second": second.name, "matched": True,
            "tasks": [{"instance_id": a["instance_id"], "first": a, "second": b}
                      for a, b in zip(left["tasks"], right["tasks"])]}
