from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

from tact.artifacts import now, read_json, write_json


root = Path('/home/wangtianyi/tactRefractor')
directory = root / 'reports/speed20-20261011'
experiment = read_json(directory / 'experiment.json')
if (directory / 'controller.json').exists():
    raise RuntimeError('Experiment already launched; inspect controller.json')
gpu_rows = subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'
], text=True)
usage = {int(line.split(',')[0]): int(line.split(',')[1]) for line in gpu_rows.splitlines()}
for arm in experiment['arms']:
    if any(usage[gpu] > 512 for gpu in arm['gpus']):
        raise RuntimeError(f"Assigned GPUs are no longer free: {arm['name']}: {usage}")
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', arm['port'])) == 0:
            raise RuntimeError(f"Port occupied: {arm['port']}")

env = dict(os.environ)
env['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
env['MSWEA_GLOBAL_CONFIG_DIR'] = str(root / 'assets/user-config/mini-swe-agent')
env['PATH'] = str(root / '.venvs/runner/bin') + ':' + env.get('PATH', '')
env.pop('TACT_SSH_PASSWORD', None)
records = []
processes = {}
handles = {}
for arm in experiment['arms']:
    name = arm['name']
    args = [str(root / '.venvs/runner/bin/python'), '-m', 'tact', '--config', arm['config'],
            'run', '--run-id', arm['run_id'], '--no-evaluate']
    handles[name] = (directory / f'{name}-launch.log').open('w')
    start = time.monotonic()
    process = subprocess.Popen(args, cwd=root, env=env, stdout=handles[name],
                               stderr=subprocess.STDOUT, start_new_session=True)
    processes[name] = (process, start)
    records.append({**arm, 'pid': process.pid, 'started_at': now(), 'command': args,
                    'exit_code': None, 'elapsed_seconds': 0})
write_json(directory / 'controller.json', {'pid': os.getpid(), 'started_at': now(), 'arms': records})
print('Launched all three arms', [(x['name'], x['pid'], x['gpus']) for x in records], flush=True)

next_snapshot = 0.0
while any(record['exit_code'] is None for record in records):
    for record in records:
        if record['exit_code'] is not None:
            continue
        process, start = processes[record['name']]
        record['elapsed_seconds'] = time.monotonic() - start
        code = process.poll()
        if code is not None:
            record['exit_code'] = code
            record['finished_at'] = now()
            record['elapsed_seconds'] = time.monotonic() - start
            handles[record['name']].close()
            print('Finished', record['name'], code, record['elapsed_seconds'], flush=True)
            report_args = [str(root / '.venvs/runner/bin/python'), '-m', 'tact',
                           'report', '--run-id', record['run_id']]
            with (directory / f"{record['name']}-report.log").open('w') as log:
                reported = subprocess.run(report_args, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            record['report_exit_code'] = reported.returncode
    write_json(directory / 'controller.json', {'pid': os.getpid(), 'updated_at': now(), 'arms': records})
    if time.monotonic() >= next_snapshot:
        snapshot = {'at': now(), 'arms': []}
        for record in records:
            run = root / 'runs' / record['run_id']
            status = read_json(run / 'status.json') if (run / 'status.json').exists() else {'stage': 'starting'}
            results = [read_json(path) for path in (run / 'tasks').glob('*/result.json')]
            snapshot['arms'].append({'name': record['name'], 'elapsed_seconds': record['elapsed_seconds'],
                                     'exit_code': record['exit_code'], 'status': status,
                                     'terminal_tasks': len(results)})
        with (directory / 'progress.jsonl').open('a') as output:
            output.write(json.dumps(snapshot, ensure_ascii=False) + '\n')
        write_json(directory / 'progress.json', snapshot)
        try:
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True, timeout=15)
            with (directory / 'gpu-utilization.jsonl').open('a') as output:
                output.write(json.dumps({'at': now(), 'gpu_csv': gpu}) + '\n')
        except (subprocess.SubprocessError, OSError) as exc:
            print('GPU sampling error', repr(exc), flush=True)
        next_snapshot = time.monotonic() + 60
    time.sleep(5)

write_json(directory / 'completed.json', {'completed_at': now(), 'arms': records})
print('All three arms finished', flush=True)
