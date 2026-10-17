from pathlib import Path
import subprocess,json,os,time,fcntl,socket,datetime
root=Path('/home/wangtianyi/tactRefractor')
directory=root/'experiments/tact-registry-eval-20261014'
lock=(directory/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'controller_pid':os.getpid(),'arms':{}}
processes={}
def save():
 state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 temp=directory/'controller-status.json.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(directory/'controller-status.json')
def available(gpus,fraction,port):
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
 values={int(a): (int(b),int(d)) for a,b,d in (row.split(',') for row in rows.strip().splitlines())}
 enough=all(values[g][0]>=int(values[g][1]*fraction)+2048 for g in gpus)
 s=socket.socket();s.settimeout(1);busy=s.connect_ex(('127.0.0.1',port))==0;s.close()
 return enough and not busy, {str(g):values[g][0] for g in gpus},busy
for arm in ['cap','gated']:
 run_id='tact-registry-eval-20261014-'+arm
 if (root/'runs'/run_id).exists():raise FileExistsError(run_id)
 state['arms'][arm]={'run_id':run_id,'state':'waiting_resources'}
save()
while True:
 for arm in ['cap','gated']:
  item=state['arms'][arm]
  if item['state']=='waiting_resources':
   cfg=json.loads((directory/(arm+'.json')).read_text());serv=cfg['serving']
   ready,free,busy=available(serv['gpus'],serv['gpu_memory_utilization'],serv['port'])
   item.update(free_mib=free,port_busy=busy,gpus=serv['gpus'],port=serv['port'])
   if ready:
    command=[str(root/'.venvs/runner/bin/python'),'-m','tact','--config',str(directory/(arm+'.json')),'run','--run-id',item['run_id'],'--no-evaluate']
    output=(directory/(arm+'-launch.log')).open('a')
    process=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    output.close();processes[arm]=process;item.update(state='running',pid=process.pid,command=command)
    print('STARTED',arm,process.pid,flush=True)
  elif item['state']=='running':
   result=processes[arm].poll()
   if result is not None:
    item.update(state='finished' if result==0 else 'failed',exit_code=result)
    print('EXIT',arm,result,flush=True)
 save()
 if all(x['state'] in ['finished','failed'] for x in state['arms'].values()):break
 time.sleep(30)
