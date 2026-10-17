from __future__ import annotations
import json
MAINTAINER="""You are a trajectory state tracker. Track only observable facts in known_facts: files read and what was learned, hypotheses, edits, test results, and key findings. Track progress and repetition separately. Do not judge labels or speculate. Output JSON only."""
VERIFIER="""You are a factual auditor. Audit only known_facts against raw steps. Report only HALLUCINATION, OMISSION, or CONTRADICTION. Do not correct paraphrases, subjective framing, progress, or repetition. If no error exists return corrections=[] and copy known_facts unchanged. Output JSON only."""
ANNOTATOR="""You are an expert judge of epistemic efficiency. Label each step against what the agent knew BEFORE it. CALIBRATED expands known facts with new evidence. OVERTHINKING re-derives known information, cycles hypotheses, or deliberates about a question a simple tool call would settle. OVERACTING repeats known observations/actions or acts without processing a recent observation. If multiple failures occur use OVERTHINKING > OVERACTING > CALIBRATED. When uncertain use low confidence. Label ONLY the candidate step(s) in chunk. chunk_prefix is past context only: never label prefix steps. Use previous_verified_state as BEFORE-chunk knowledge and increment it conceptually with prefix only. Never use future steps. Output JSON only."""
STATE_SCHEMA={"known_facts":{"files_read":{},"hypotheses":"string","edits_made":"string","test_results":"string","key_findings":"string"},"progress":{"phase":"understanding_issue|localizing_code|editing|testing|verifying|stuck_looping","progress_trend":["string"],"steps_since_progress":0,"repetition_tracker":{"files_read_multiple":[],"repeated_patterns":[],"circular_reasoning":[]}}}
VALIDATION_SCHEMA={"corrections":[{"type":"HALLUCINATION|OMISSION|CONTRADICTION","field":"string","message":"string"}],"validated_known_facts":{}}
ANNOTATION_SCHEMA={"labels":[{"step":0,"label":"OVERTHINKING|OVERACTING|CALIBRATED","justification":"string","confidence":"high|medium|low"}]}
def payload(task,state,chunk,prefix=None,proposed=None,corrections=None):
 d={"task":task,"previous_verified_state":state,"chunk":chunk}
 if prefix is not None:d["chunk_prefix"]=prefix
 if proposed is not None:d["proposed_state"]=proposed
 if corrections is not None:d["validator_corrections"]=corrections
 return d
