from __future__ import annotations
import asyncio,hashlib,json,os,random,re,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
class DeepSeekError(RuntimeError): pass

def parse_json_response(text: str):
 text=text.strip()
 if text.startswith("```"):
  text=re.sub(r"^```(?:json)?\s*","",text,flags=re.I)
  text=re.sub(r"\s*```$","",text)
 try:return json.loads(text)
 except json.JSONDecodeError:
  decoder=json.JSONDecoder(); value,_=decoder.raw_decode(text); return value

class DeepSeekClient:
 def __init__(self,cache_dir,model="deepseek-chat",base_url=None,api_key=None,timeout=180,retries=5,semaphore=None):
  self.cache_dir=Path(cache_dir);self.cache_dir.mkdir(parents=True,exist_ok=True);self.failed_dir=self.cache_dir/"failed";self.failed_dir.mkdir(parents=True,exist_ok=True);self.model=model;self.base_url=(base_url or os.getenv("DEEPSEEK_BASE_URL","https://api.deepseek.com")).rstrip("/")+"/chat/completions";self.api_key=api_key or os.getenv("DEEPSEEK_API_KEY");self.timeout=timeout;self.retries=retries;self.semaphore=semaphore or asyncio.Semaphore(16)
  if not self.api_key:raise DeepSeekError("DEEPSEEK_API_KEY is not set")
 @staticmethod
 def _hash(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
 async def complete(self,role,system_prompt,payload,schema,metadata=None):
  req={"model":self.model,"temperature":0,"messages":[{"role":"system","content":system_prompt+" Return ONE JSON object only. Never concatenate JSON objects. Do not emit explanations outside JSON. Use validated_known_facts for validator output."},{"role":"user","content":json.dumps({"input":payload,"required_output_structure":schema,"instruction":"Return exactly this JSON structure. Use actual enum values, not pipe-separated choices. No extra keys."},ensure_ascii=False)}],"response_format":{"type":"json_object"},"stream":False};key=self._hash({"request":req,"schema":schema});cache=self.cache_dir/(key+".json")
  if cache.exists():return json.loads(cache.read_text(encoding="utf-8"))["parsed"]
  body=json.dumps(req,ensure_ascii=False).encode();started=time.time();last=None
  for attempt in range(1,self.retries+1):
   result=None;raw=""
   try:
    async with self.semaphore:result=await asyncio.to_thread(self._request,body)
    raw=result.get("choices",[{}])[0].get("message",{}).get("content","")
    parsed=parse_json_response(raw);record={"request":req,"schema":schema,"role":role,"model":self.model,"request_hash":key,"attempt":attempt,"elapsed_seconds":time.time()-started,"usage":result.get("usage"),"response":result,"parsed":parsed,"metadata":metadata or {}}
    cache.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8");return parsed
   except Exception as exc:
    last=exc
    failure={"request":req,"schema":schema,"role":role,"model":self.model,"request_hash":key,"attempt":attempt,"elapsed_seconds":time.time()-started,"metadata":metadata or {},"error":repr(exc),"raw_content":raw,"response":result}
    (self.failed_dir/(key+f".attempt-{attempt}.json")).write_text(json.dumps(failure,ensure_ascii=False,indent=2),encoding="utf-8")
    if attempt<self.retries:await asyncio.sleep(min(30,2**(attempt-1))+random.random())
  raise DeepSeekError(f"{role} failed after {self.retries} attempts: {last}")
 def _request(self,body):
  req=Request(self.base_url,data=body,headers={"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json"},method="POST")
  try:
   with urlopen(req,timeout=self.timeout) as response:return json.loads(response.read())
  except HTTPError as e:raise DeepSeekError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:1000]}")
  except URLError as e:raise DeepSeekError(str(e))
