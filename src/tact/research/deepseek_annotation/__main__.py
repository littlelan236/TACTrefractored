from __future__ import annotations
import argparse,asyncio,json
from pathlib import Path
from tact.artifacts import read_json,selected_trajectory,sha256
from tact.config import load_resolved
from tact.research.annotation import export_inputs
from .client import DeepSeekClient
from .pipeline import run_jobs
def main():
 p=argparse.ArgumentParser(prog="tact-deepseek-annotate");p.add_argument("--run-id",required=True);p.add_argument("--label-set",required=True);p.add_argument("--max-trajectories",type=int,default=8);p.add_argument("--max-inflight-requests",type=int,default=16);p.add_argument("--model",default="deepseek-chat");p.add_argument("--resume",action="store_true");p.add_argument("--max-loop-steps",type=int,default=10);p.add_argument("--instance-id",action="append",default=[]);p.add_argument("--include-fit-exclusions",action="store_true");args=p.parse_args();root=Path.cwd();run=root/"runs"/args.run_id;cfg=load_resolved(run/"resolved-config.json");output=root/"artifacts/labels"/args.label_set
 if not output.exists() and args.include_fit_exclusions:
  cfg["research"]["fit_exclusions"]="/tmp/empty-fit-exclusions.json";Path(cfg["research"]["fit_exclusions"]).write_text(json.dumps({"instance_ids":[]})); export_inputs(cfg,run,args.label_set)
 manifest=read_json(output/"manifest.json")
 if args.instance_id: manifest["tasks"]=[x for x in manifest["tasks"] if x["instance_id"] in set(args.instance_id)]
 if not manifest["tasks"]: raise ValueError("No requested instance-id found in label manifest")
 manifest["all_steps_labeled"]=True
 (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8");jobs=[]
 for task in manifest["tasks"]:
  iid=task["instance_id"];traj=selected_trajectory(run,iid)
  if traj is None or sha256(traj)!=task["source_sha256"]:raise RuntimeError(f"trajectory provenance mismatch: {iid}")
  inp=output/"inputs"/iid;steps=[json.loads(x) for x in (inp/"steps.jsonl").read_text(encoding="utf-8").split(chr(10)) if x.strip()]
  jobs.append({"instance_id":iid,"task":(inp/"task.txt").read_text(encoding="utf-8"),"steps":steps,"source_sha256":task["source_sha256"]})
 client=DeepSeekClient(output/"cache",model=args.model,semaphore=asyncio.Semaphore(args.max_inflight_requests));results=asyncio.run(run_jobs(jobs,output,client,args.max_trajectories,args.max_loop_steps));(output/"run-results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(results,ensure_ascii=False,indent=2));
 if any(not x["ok"] for x in results):raise SystemExit(1)
if __name__=="__main__":main()
