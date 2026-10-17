from pathlib import Path
import datetime, fcntl, json, os, socket, subprocess, time
ROOT=Path('/home/wangtianyi/tactRefractor')
D=ROOT/'experiments/lda-registry-seed42-eval-20261015'
DEPENDENCY=ROOT/'experiments/tact-easysteer-seed42-eval-20261015'
PYTHON=str(ROOT/'.venvs/runner/bin/python')
ARMS=('cap','gated')

def read(p): return json.loads(Path(p).read_text())
def save(s):
 s['updated_at']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
 t=D/'lda-queue-status.json.tmp'; t.write_text(json.dumps(s,indent=2)+'\n'); t.replace(D/'lda-queue-status.json')
def ready(cfg):
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
 mem={int(a):(int(b),int(c)) for a,b,c in (x.split(',') for x in rows.splitlines())}
 s=cfg['serving']; sock=socket.socket(); sock.settimeout(1)
 busy=sock.connect_ex((s['host'],s['port']))==0; sock.close()
 return (not busy and all(mem[g][0]>=int(mem[g][1]*s['gpu_memory_utilization'])+2048 for g in s['gpus'])), {str(g):mem[g][0] for g in s['gpus']}
def dependency_ready():
 q=read(DEPENDENCY/'lda-queue-status.json'); ev={}
 for arm in ARMS:
  x=q['arms'][arm]; run=ROOT/'runs'/x['run_id']; summary=run/'evaluation/candidate/summary.json'
  ok=x.get('state') in {'finished','failed'} and x.get('rollout_exit_code')==0 and x.get('report_exit_code')==0 and summary.exists()
  detail={'queue_state':x.get('state'),'summary_exists':summary.exists()}
  if summary.exists():
   d=read(summary); ids=read(run/'selection.json')['instance_ids']; detail.update(outcomes=len(d.get('outcomes',{})),complete=d.get('complete'),failures=d.get('failures',[])); ok=ok and set(d.get('outcomes',{}))==set(ids)
  ev[arm]=detail
  detail['ready']=ok
 return all(x['ready'] for x in ev.values()),ev
def launch(arm,stage,cmd,state,procs):
 with (D/f'lda-{arm}-{stage}.log').open('a') as log:
  p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 procs[arm]=p; state['arms'][arm].update(state=stage+'_running',pid=p.pid,command=cmd); print('STARTED',arm,stage,p.pid,flush=True); save(state)
def main():
 lock=(D/'lda-queue.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (D/'lda-queue-status.json').exists(): raise RuntimeError('Existing LDA queue state; refusing duplicate')
 cfg={a:read(D/f'lda-{a}.json') for a in ARMS}; state={'queue_pid':os.getpid(),'stage':'waiting_easysteer_lda_seed42','arms':{},'eval_id':'candidate'}; procs={}
 for a in ARMS:
  rid=f'lda-registry-seed42-eval-20261015-{a}'
  if (ROOT/'runs'/rid).exists(): raise FileExistsError(rid)
  state['arms'][a]={'run_id':rid,'state':'waiting_easysteer_lda_seed42','eval_id':'candidate'}
 save(state)
 while True:
  if state['stage']=='waiting_easysteer_lda_seed42':
   ok,ev=dependency_ready(); state['easysteer_lda_dependency']=ev
   if ok:
    state['stage']='executing'
    for x in state['arms'].values(): x['state']='waiting_resources'
  if state['stage']=='executing':
   for a,x in state['arms'].items():
    st=x['state']
    if st=='waiting_resources':
     ok,free=ready(cfg[a]); x['free_mib']=free
     if ok: launch(a,'rollout',[PYTHON,'-m','tact','--config',str(D/f'lda-{a}.json'),'rollout','--run-id',x['run_id']],state,procs)
    elif st.endswith('_running'):
     p=procs[a]; rc=p.poll()
     if rc is None: continue
     stage=st.removesuffix('_running'); x[stage+'_exit_code']=rc; run=ROOT/'runs'/x['run_id']
     if stage=='rollout':
      ids=read(run/'selection.json')['instance_ids'] if (run/'selection.json').exists() else ['<missing-selection>']; missing=[i for i in ids if not (run/'tasks'/i/'result.json').exists()]
      x['missing_results']=missing; x['state']='waiting_harness' if rc==0 and not missing else 'rollout_failed'
     elif stage=='harness':
      sm=run/'evaluation/candidate/summary.json'
      if sm.exists(): x['evaluation']={k:v for k,v in read(sm).items() if k!='outcomes'}
      launch(a,'report',[PYTHON,'-m','tact','report','--run-id',x['run_id']],state,procs)
     else: x['state']='finished' if rc==0 and x.get('harness_exit_code')==0 else 'failed'
     save(state)
   if not any(x['state']=='harness_running' for x in state['arms'].values()):
    for a,x in state['arms'].items():
     if x['state']=='waiting_harness':
      out=ROOT/'runs'/x['run_id']/'evaluation/candidate'
      if out.exists(): x['state']='blocked_existing_evaluation'; continue
      launch(a,'harness',[PYTHON,'-m','tact','evaluate','--run-id',x['run_id'],'--eval-id','candidate'],state,procs); break
   if all(x['state'] in {'finished','failed','rollout_failed','blocked_existing_evaluation'} for x in state['arms'].values()):
    state['stage']='finished'; save(state); break
  save(state); time.sleep(30)
if __name__=='__main__': main()
