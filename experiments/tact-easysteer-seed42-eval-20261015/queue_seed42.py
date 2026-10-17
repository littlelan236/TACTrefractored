"""Wait for both Registry arms, then EasySteer seed42 rollouts and serial harness."""
from pathlib import Path
import datetime,fcntl,json,os,socket,subprocess,time
ROOT=Path('/home/wangtianyi/tactRefractor')
DIRECTORY=ROOT/'experiments/tact-easysteer-seed42-eval-20261015'
DEPENDENCY=ROOT/'experiments/tact-registry-eval-20261014'
PYTHON=str(ROOT/'.venvs/runner/bin/python')
def read(path):
    return json.loads(path.read_text())
def save(state):
    state['updated_at']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    temp=DIRECTORY/'queue-status.json.tmp'
    temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(DIRECTORY/'queue-status.json')
def ready(cfg):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
    memory={int(a):(int(b),int(c)) for a,b,c in (r.split(',') for r in rows.splitlines())}
    s=cfg['serving'];sock=socket.socket();sock.settimeout(1)
    busy=sock.connect_ex((s['host'],s['port']))==0;sock.close()
    return not busy and all(memory[g][0]>=int(memory[g][1]*s['gpu_memory_utilization'])+2048 for g in s['gpus']),{str(g):memory[g][0] for g in s['gpus']}
def dependencies():
    rollout=read(DEPENDENCY/'controller-status.json')
    harness=read(DEPENDENCY/'harness-queue-status.json')
    evidence={}
    for arm in ['cap','gated']:
        r,h=rollout['arms'][arm],harness['arms'][arm]
        summary=ROOT/'runs'/r['run_id']/'evaluation/candidate/summary.json'
        info={'rollout':r['state'],'harness':h['state'],'summary_exists':summary.exists()}
        evidence[arm]=info
        if r['state']!='finished' or h['state'] not in ['finished','failed'] or not summary.exists():
            return False,evidence
        data=read(summary)
        ids=read(ROOT/'runs'/r['run_id']/'selection.json')['instance_ids']
        info.update(outcomes=len(data['outcomes']),failures=data.get('failures',[]),complete=data['complete'])
        if set(data['outcomes'])!=set(ids):return False,evidence
    return True,evidence

def main():
    lock=(DIRECTORY/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (DIRECTORY/'queue-status.json').exists():
        raise RuntimeError('Existing queue state: inspect before restarting; do not duplicate jobs')
    state={'queue_pid':os.getpid(),'stage':'waiting_registry','arms':{}}
    configs={};processes={}
    for arm in ['cap','gated']:
        configs[arm]=read(DIRECTORY/(arm+'.json'))
        run_id='tact-easysteer-seed42-eval-20261015-'+arm
        if (ROOT/'runs'/run_id).exists():raise FileExistsError(run_id)
        state['arms'][arm]={'run_id':run_id,'state':'waiting_dependencies','eval_id':'candidate'}
    save(state)
    def launch(arm,stage,command):
        with (DIRECTORY/(arm+'-'+stage+'.log')).open('a') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[arm]=p;state['arms'][arm].update(state=stage+'_running',pid=p.pid,command=command)
        print('STARTED',arm,stage,p.pid,flush=True);save(state)
    while True:
        if state['stage']=='waiting_registry':
            done,evidence=dependencies();state['dependencies']=evidence
            if done:
                state['stage']='executing'
                for item in state['arms'].values():item['state']='waiting_resources'
        if state['stage']=='executing':
            for arm,item in state['arms'].items():
                if item['state']=='waiting_resources':
                    ok,free=ready(configs[arm]);item['free_mib']=free
                    if ok:
                        launch(arm,'rollout',[PYTHON,'-m','tact','--config',str(DIRECTORY/(arm+'.json')),'rollout','--run-id',item['run_id']])
                elif item['state'].endswith('_running'):
                    rc=processes[arm].poll()
                    if rc is None:continue
                    stage=item['state'].removesuffix('_running');item[stage+'_exit_code']=rc
                    run=ROOT/'runs'/item['run_id']
                    if stage=='rollout':
                        ids=read(run/'selection.json')['instance_ids']
                        missing=[iid for iid in ids if not (run/'tasks'/iid/'result.json').exists()]
                        item.update(missing_results=missing,state='waiting_harness' if rc==0 and not missing else 'rollout_failed')
                    elif stage=='harness':
                        summary=run/'evaluation/candidate/summary.json'
                        if summary.exists():
                            data=read(summary);item['evaluation']={k:v for k,v in data.items() if k!='outcomes'}
                        launch(arm,'report',[PYTHON,'-m','tact','report','--run-id',item['run_id']])
                    else:
                        item['state']='finished' if rc==0 and item.get('harness_exit_code')==0 else 'failed'
                    save(state)
            # Serialize grading (12 workers each), retaining the original harness settings.
            if not any(i['state']=='harness_running' for i in state['arms'].values()):
                for arm,item in state['arms'].items():
                    if item['state']=='waiting_harness':
                        output=ROOT/'runs'/item['run_id']/'evaluation/candidate'
                        if output.exists():
                            item['state']='blocked_existing_evaluation';continue
                        launch(arm,'harness',[PYTHON,'-m','tact','evaluate','--run-id',item['run_id'],'--eval-id','candidate']);break
            terminal={'finished','failed','rollout_failed','blocked_existing_evaluation'}
            if all(i['state'] in terminal for i in state['arms'].values()):
                state['stage']='finished';save(state);break
        save(state);time.sleep(30)
if __name__=='__main__':
    main()
