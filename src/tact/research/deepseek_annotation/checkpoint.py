from __future__ import annotations
import hashlib,json,os,shutil
from pathlib import Path
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
class CheckpointStore:
 def __init__(self,root):self.root=Path(root)
 def chunk_dir(self,iid,index):return self.root/"states"/iid/f"chunk_{index:03d}"
 def done(self,iid,index):return (self.chunk_dir(iid,index)/"manifest.json").is_file()
 def load(self,iid,index):
  d=self.chunk_dir(iid,index);return json.loads((d/"validated_state.json").read_text()),json.loads((d/"labels.json").read_text())
 def commit(self,iid,index,proposed,validated,labels,metadata):
  final=self.chunk_dir(iid,index);tmp=final.with_name(final.name+".tmp")
  if tmp.exists():shutil.rmtree(tmp)
  tmp.mkdir(parents=True)
  for name,value in (("proposed_state.json",proposed),("validated_state.json",validated),("labels.json",labels)):(tmp/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
  manifest={**metadata,"state_sha256":digest(validated),"labels_sha256":digest(labels),"complete":True}
  (tmp/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8");os.replace(tmp,final)
