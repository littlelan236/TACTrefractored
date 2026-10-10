from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["rollout", "evaluate", "report", "environment-check",
                                         "decision-points", "latent", "fit", "plot"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--eval-id", default="candidate")
    parser.add_argument("--gold", action="store_true")
    parser.add_argument("--set-id")
    parser.add_argument("--label-set")
    parser.add_argument("--bundle-id")
    args = parser.parse_args()
    from tact.config import load_resolved
    cfg = load_resolved(args.run / "resolved-config.json")
    # Library global mini config is isolated in a project-owned empty location.
    os.environ["XDG_CONFIG_HOME"] = str(Path(cfg["project_root"]) / "assets/user-config")
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Worker received signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    if args.stage == "rollout":
        from tact.rollout.runner import run_rollout
        result = run_rollout(cfg, args.run)
    elif args.stage == "evaluate":
        from tact.evaluation.runner import evaluate
        result = evaluate(cfg, args.run, args.eval_id, args.gold)
    elif args.stage == "report":
        from tact.reporting.metrics import report
        result = report(cfg, args.run)
    elif args.stage == "environment-check":
        from tact.environments.check import check_environments
        result = check_environments(cfg, args.run)
    else:
        root = Path(cfg["project_root"])
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg["serving"]["gpus"]))
        if args.stage == "decision-points":
            from tact.research.decision_points import build_decisions
            result = {"path": str(build_decisions(cfg, args.run, args.set_id))}
        elif args.stage == "latent":
            from tact.research.extract import extract
            extract(cfg, root / "artifacts/latents" / args.set_id)
            result = {"complete": True}
        elif args.stage == "fit":
            from tact.research.axes import fit_axes
            result = {"path": str(fit_axes(cfg, root / "artifacts/latents" / args.set_id,
                       root / "artifacts/labels" / args.label_set, args.bundle_id))}
        else:
            from tact.research.plots import plot_bundle
            plot_bundle(root / "artifacts/axes" / args.bundle_id)
            result = {"complete": True}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("complete", True) else 1)


if __name__ == "__main__":
    main()
