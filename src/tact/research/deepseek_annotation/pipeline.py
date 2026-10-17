from __future__ import annotations
import asyncio,json
from pathlib import Path
from .checkpoint import CheckpointStore
from .loops import detect_loop_cutoff
from .prompts import MAINTAINER,VERIFIER,ANNOTATOR,STATE_SCHEMA,VALIDATION_SCHEMA,payload
from .schemas import initial_state,validate_state,validate_labels

class TrajectoryWorker:
 def __init__(self,job,output,client,max_validation_rounds=3,max_loop_steps=10):
  self.job=job;self.output=Path(output);self.client=client;self.max_validation_rounds=max_validation_rounds;self.max_loop_steps=max_loop_steps;self.store=CheckpointStore(self.output)
 async def run(self):
  iid=self.job["instance_id"];state=initial_state();labels=[];all_steps=self.job["steps"]
  loop=detect_loop_cutoff(all_steps,max_loop_steps=self.max_loop_steps);steps=all_steps[:loop.get("effective_step_count",len(all_steps))]
  chunks=[steps[i:i+10] for i in range(0,len(steps),10)]
  for ci,chunk in enumerate(chunks):
   if self.store.done(iid,ci):state,old=self.store.load(iid,ci);labels.extend(old);continue
   proposed=await self.client.complete("maintainer",MAINTAINER,payload(self.job["task"],state,chunk),STATE_SCHEMA,{"instance_id":iid,"chunk":ci});proposed=validate_state(proposed)
   validated=await self._validate(state,chunk,proposed,ci)
   annotated=[];prefix=[]
   for candidate in chunk:
    schema={"labels":[{"step":candidate["step"],"label":"OVERTHINKING|OVERACTING|CALIBRATED","justification":"string","confidence":"high|medium|low"}]}
    result=await self.client.complete("annotator",ANNOTATOR,payload(self.job["task"],state,[candidate],prefix=prefix),schema,{"instance_id":iid,"chunk":ci,"step":candidate["step"]})
    if "labels" not in result and "label" in result:result={"labels":[{"step":candidate["step"],"label":result["label"],"justification":result.get("justification",result.get("reasoning",result.get("reason",""))),"confidence":result.get("confidence","low")} ]}
    annotated.extend(validate_labels(result,{candidate["step"]}));prefix.append(candidate)
   annotated=[{**r,"instance_id":iid,"source_sha256":self.job["source_sha256"],"chunk_index":ci} for r in annotated]
   meta={"previous_verified_state":state,"instance_id":iid,"chunk_index":ci,"source_sha256":self.job["source_sha256"],"steps":[s["step"] for s in chunk],"loop_detection":loop}
   self.store.commit(iid,ci,proposed,validated,annotated,meta);state=validated;labels.extend(annotated)
  out=self.output/"generated";out.mkdir(parents=True,exist_ok=True);(out/(iid+".jsonl")).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in sorted(labels,key=lambda x:x["step"])),encoding="utf-8")
  summary={"instance_id":iid,"labelled_steps":len(labels),"original_steps":len(all_steps),"loop_detection":loop,"status":"completed_loop_truncated" if loop.get("truncated") else "completed"}
  (self.output/"states"/iid/"trajectory-manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
  return summary
 async def _validate(self,previous,chunk,current,ci):
  for ri in range(self.max_validation_rounds+1):
   result=await self.client.complete("validator",VERIFIER,payload(self.job["task"],previous,chunk,proposed=current),VALIDATION_SCHEMA,{"instance_id":self.job["instance_id"],"chunk":ci,"round":ri});corrections=result.get("corrections")
   if not isinstance(corrections,list):raise ValueError("validator corrections must be a list")
   if not corrections:return validate_state({"known_facts":result.get("validated_known_facts",result.get("known_facts",current["known_facts"])),"progress":current["progress"]})
   if ri>=self.max_validation_rounds:raise RuntimeError(f"state validation failed: {corrections}")
   current=await self.client.complete("maintainer_revision",MAINTAINER,payload(self.job["task"],previous,chunk,proposed=current,corrections=corrections),STATE_SCHEMA,{"instance_id":self.job["instance_id"],"chunk":ci,"round":ri,"revision":True});current=validate_state(current)
  raise RuntimeError("unreachable")

async def run_jobs(jobs,output,client,max_trajectories=8,max_loop_steps=10):
 sem=asyncio.Semaphore(max_trajectories)
 async def one(job):
  async with sem:
   try:return {"instance_id":job["instance_id"],"ok":True,"result":await TrajectoryWorker(job,output,client,max_loop_steps=max_loop_steps).run()}
   except Exception as e:return {"instance_id":job["instance_id"],"ok":False,"error":repr(e)}
 return await asyncio.gather(*(one(j) for j in jobs))
