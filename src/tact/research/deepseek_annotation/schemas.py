from __future__ import annotations
LABELS={"OVERTHINKING","OVERACTING","CALIBRATED"}; CONFIDENCES={"high","medium","low"}; PHASES={"understanding_issue","localizing_code","editing","testing","verifying","stuck_looping"}
def initial_state():
 return {"known_facts":{"files_read":{},"hypotheses":"","edits_made":"","test_results":"","key_findings":""},"progress":{"phase":"understanding_issue","progress_trend":[],"steps_since_progress":0,"repetition_tracker":{"files_read_multiple":[],"repeated_patterns":[],"circular_reasoning":[]}}}
def validate_state(v):
 if not isinstance(v,dict) or not isinstance(v.get("known_facts"),dict) or not isinstance(v.get("progress"),dict): raise ValueError("invalid rolling state")
 k,p=v["known_facts"],v["progress"]
 for x in ("files_read","hypotheses","edits_made","test_results","key_findings"):
  if x not in k: raise ValueError("missing known_facts."+x)
 if p.get("phase") not in PHASES or type(p.get("steps_since_progress")) is not int or p["steps_since_progress"]<0: raise ValueError("invalid progress")
 tr=p.get("repetition_tracker")
 if not isinstance(tr,dict) or any(not isinstance(tr.get(x),list) for x in ("files_read_multiple","repeated_patterns","circular_reasoning")): raise ValueError("invalid repetition_tracker")
 return v
def validate_labels(v,expected):
 if not isinstance(v,dict) or not isinstance(v.get("labels"),list): raise ValueError("missing labels")
 seen=set()
 for r in v["labels"]:
  if not isinstance(r,dict) or type(r.get("step")) is not int or r["step"] not in expected or r["step"] in seen: raise ValueError("invalid label step")
  if r.get("label") not in LABELS or r.get("confidence") not in CONFIDENCES or not str(r.get("justification","")).strip(): raise ValueError("invalid label")
  seen.add(r["step"])
 if seen!=expected: raise ValueError("incomplete labels")
 return v["labels"]
