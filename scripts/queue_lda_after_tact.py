"""Wait for TACT harness completion, then run matched LDA CAP/GATE."""
from __future__ import annotations
import argparse,copy,fcntl,json,os,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tact.artifacts import now,read_json,write_json,sha256
from tact.config import resolve_config,flatten_config
SOURCE='tact-easysteer-eval-20261013'
TARGET='lda-easysteer-eval-20261013'
EXP=ROOT/'experiments'/TARGET
PYTHON=ROOT/'.venvs/runner/bin/python'

def prepare():
    EXP.mkdir(parents=True,exist_ok=True)
    arms=[]
    for strategy in ['cap','gated']:
        source=ROOT/'experiments'/SOURCE/(strategy+'.json')
        old=read_json(source);cfg=copy.deepcopy(old)
        cfg['steering']['axes']=str(ROOT/'baseline/axes/lda')
        cfg['execution']['rollout_workers']=15
        cfg['serving']['max_num_seqs']=15
        a,b=flatten_config(old),flatten_config(cfg)
        changed={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
        assert changed=={'steering.axes','execution.rollout_workers','serving.max_num_seqs'},changed
        path=EXP/(strategy+'.json')
        if path.exists() and read_json(path)!=cfg:raise RuntimeError('Queued config changed: '+strategy)
        write_json(path,cfg)
        resolved=resolve_config(path)
        assert len(resolved['selected_ids'])==67
        arms.append({'strategy':strategy,'run_id':TARGET+'-'+strategy,'source_run_id':SOURCE+'-'+strategy,'config':str(path),'source_config':str(source),'source_config_sha256':sha256(source),'gpus':cfg['serving']['gpus'],'port':cfg['serving']['port'],'selected_ids':resolved['selected_ids']})
    write_json(EXP/'plan.json',{'created_at':now(),'axis':str(ROOT/'baseline/axes/lda'),'changes':['steering.axes','execution.rollout_workers=15','serving.max_num_seqs=15'],'harness_workers':read_json(EXP/'cap.json')['execution']['harness_workers'],'arms':arms})
    return arms

def dependency_state(arms):
    summaries=[]
    for arm in arms:
        run=ROOT/'runs'/arm['source_run_id'];summary=run/'evaluation/candidate/summary.json';status=run/'evaluation/candidate/status.json'
        if not summary.exists() or not status.exists():return False
        d=read_json(summary);s=read_json(status)
        if s.get('stage') not in {'complete','failed'}:return False
        if d.get('requested')!=len(arm['selected_ids']) or len(d.get('outcomes',{}))!=len(arm['selected_ids']):return False
        summaries.append({'run_id':arm['source_run_id'],'summary':str(summary),'sha256':sha256(summary),'resolved':d['resolved'],'graded':d['graded'],'requested':d['requested'],'complete':d['complete']})
    write_json(EXP/'dependency-results.json',{'at':now(),'evaluations':summaries})
    return True

def run(resume=False):
    arms=prepare();processes=[];logs=[]
    try:
        write_json(EXP/'status.json',{'stage':'queued','pid':os.getpid(),'at':now(),'waiting_for':[a['source_run_id'] for a in arms]})
        print('Queued: waiting for both TACT harness summaries',flush=True)
        while not dependency_state(arms):
            completed=ROOT/'experiments'/SOURCE/'completed.json'
            terminal=ROOT/'experiments'/SOURCE/'status.json'
            if completed.exists() or (terminal.exists() and read_json(terminal).get('stage')=='failed'):
                raise RuntimeError('TACT controller finished without two complete harness outcome sets; inspect dependency before restarting queue')
            time.sleep(60)
        # Original services must have released their ports before the new arms start.
        for arm in arms:
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',arm['port']))==0:raise RuntimeError(f"Port still occupied: {arm['port']}")
            if (ROOT/'runs'/arm['run_id']).exists() and not resume:raise RuntimeError('Run already exists: '+arm['run_id'])
        for arm in arms:
            handle=(EXP/(arm['strategy']+'.log')).open('a');logs.append(handle)
            if resume and (ROOT/'runs'/arm['run_id']).exists():
                cmd=[str(PYTHON),'-m','tact','rollout','--run-id',arm['run_id'],'--resume-run']
            else:
                cmd=[str(PYTHON),'-m','tact','--config',arm['config'],'run','--run-id',arm['run_id'],'--no-evaluate']
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT);processes.append(p);arm['pid']=p.pid
        write_json(EXP/'controller.json',{'pid':os.getpid(),'started_at':now(),'arms':arms})
        write_json(EXP/'status.json',{'stage':'rollout','pid':os.getpid(),'at':now()})
        for arm,p in zip(arms,processes):arm['rollout_exit_code']=p.wait()
        write_json(EXP/'status.json',{'stage':'harness','pid':os.getpid(),'at':now()})
        evals=[]
        for arm,handle in zip(arms,logs):
            if not (ROOT/'runs'/arm['run_id']/'predictions.json').exists():raise RuntimeError('No predictions for '+arm['run_id'])
            cmd=[str(PYTHON),'-m','tact','evaluate','--run-id',arm['run_id'],'--eval-id','candidate']
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT);evals.append(p);processes.append(p)
        for arm,p,handle in zip(arms,evals,logs):
            arm['harness_exit_code']=p.wait()
            result=subprocess.run([str(PYTHON),'-m','tact','report','--run-id',arm['run_id']],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
            arm['report_exit_code']=result.returncode
        failed=any(a['rollout_exit_code'] or a['harness_exit_code'] or a['report_exit_code'] for a in arms)
        write_json(EXP/'completed.json',{'finished_at':now(),'arms':arms})
        write_json(EXP/'status.json',{'stage':'failed' if failed else 'complete','at':now(),'arms':arms})
        if failed:raise RuntimeError('One or more LDA stages failed; see arm logs')
    except BaseException as exc:
        write_json(EXP/'status.json',{'stage':'failed','error':repr(exc),'at':now(),'pid':os.getpid()});raise
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for handle in logs:handle.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    EXP.mkdir(parents=True,exist_ok=True)
    with (EXP/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.check:
            arms=prepare();print('Validated LDA queue configs for',[(a['strategy'],a['gpus'],len(a['selected_ids'])) for a in arms])
        else:run(resume=args.resume)
