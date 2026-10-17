from tact.research.decision_points import template_messages,tools_schema
def validate_saved_decisions(decisions,reader,rows,tok,close):
 expected={r["step"] for r in rows}
 if not decisions or {d.get("step") for d in decisions}!=expected:return None
 by={s["step"]:s for s in reader.steps()}
 for d in decisions:
  s=by.get(d.get("step"));resp=s.get("response") or {};reported=(resp.get("usage") or {}).get("prompt_tokens")
  if not isinstance(reported,int) or d.get("prompt_tokens")!=reported:return None
  prompt=tok.apply_chat_template(template_messages(reader.prepared_prefix(s["message_index"])),tools=tools_schema(),tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True,preserve_thinking=True)
  if len(prompt)!=reported:return None
 return decisions
