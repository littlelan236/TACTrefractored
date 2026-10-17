from __future__ import annotations
import hashlib,json,re
from typing import Any

def _signature(step:dict[str,Any])->str:
 commands=[]
 for call in step.get("tool_calls") or []:
  fn=call.get("function",call) if isinstance(call,dict) else {};args=fn.get("arguments",{}) if isinstance(fn,dict) else {}
  if isinstance(args,str):
   try:args=json.loads(args)
   except json.JSONDecodeError:pass
  command=args.get("command","") if isinstance(args,dict) else str(args);commands.append(re.sub(r"\s+"," ",str(command)).strip())
 value="\n".join(commands) if commands else str(step.get("content") or "")
 return hashlib.sha1(value.encode("utf-8","replace")).hexdigest()

def detect_loop_cutoff(steps:list[dict[str,Any]],*,max_loop_steps:int=10)->dict[str,Any]:
 n=len(steps)
 if n<20:return {"truncated":False,"original_step_count":n,"effective_step_count":n}
 sig=[_signature(x) for x in steps];best=None
 for start in range(n):
  for period in range(1,min(6,(n-start)//2+1)):
   repeats=1
   while start+(repeats+1)*period<=n and sig[start+(repeats-1)*period:start+repeats*period]==sig[start+repeats*period:start+(repeats+1)*period]:repeats+=1
   if repeats<2 or (period==1 and repeats<3):continue
   candidate=(start,period,repeats)
   if best is None or candidate[:2]<best[:2]:best=candidate
 if best is None:return {"truncated":False,"original_step_count":n,"effective_step_count":n}
 start,period,repeats=best;cutoff=min(n,start+max_loop_steps)
 return {"truncated":cutoff<n,"original_step_count":n,"effective_step_count":cutoff,"loop_start_step":start,"loop_period":period,"loop_repeats_observed":repeats,"retained_loop_steps":cutoff-start,"detection":"exact_consecutive_tool_action_cycle"}
