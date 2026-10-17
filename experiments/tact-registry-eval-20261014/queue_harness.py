"""Queue harness/report after each existing rollout controller arm exits successfully."""
from pathlib import Path
import datetime
import fcntl
import json
import os
import subprocess
import time

ROOT = Path('/home/wangtianyi/tactRefractor')
DIRECTORY = ROOT / 'experiments/tact-registry-eval-20261014'
EVAL_ID = 'candidate'

def read(path):
    return json.loads(path.read_text())

def save(state):
    state['updated_at'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    temp = DIRECTORY / 'harness-queue-status.json.tmp'
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(DIRECTORY / 'harness-queue-status.json')

def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False

def main():
    lock = (DIRECTORY / 'harness-queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = DIRECTORY / 'harness-queue-status.json'
    state = read(path) if path.exists() else {'arms': {
        arm: {'run_id': 'tact-registry-eval-20261014-' + arm,
              'state': 'waiting_rollout', 'eval_id': EVAL_ID}
        for arm in ['cap', 'gated']}}
    state['queue_pid'] = os.getpid()
    save(state)
    while True:
        controller = read(DIRECTORY / 'controller-status.json')
        for arm, item in state['arms'].items():
            if item['state'] != 'waiting_rollout':
                continue
            rollout = controller['arms'][arm]
            item['rollout_state'] = rollout['state']
            if rollout['state'] == 'failed':
                item.update(state='blocked_rollout_failed', rollout_exit_code=rollout.get('exit_code'))
                continue
            if rollout['state'] != 'finished':
                if not alive(controller['controller_pid']):
                    item.update(state='blocked_controller_stopped')
                continue
            run = ROOT / 'runs' / item['run_id']
            cfg = read(run / 'resolved-config.json')
            ids = cfg['selected_ids']
            missing = [iid for iid in ids if not (run / 'tasks' / iid / 'result.json').exists()]
            item.update(expected_tasks=len(ids), harness_workers=cfg['execution']['harness_workers'],
                        missing_results=missing)
            if missing:
                item.update(state='blocked_incomplete_rollout')
                continue
            output = run / 'evaluation' / EVAL_ID
            if output.exists():
                item.update(state='blocked_existing_evaluation', evaluation_directory=str(output))
                continue
            command = [str(ROOT / '.venvs/runner/bin/python'), '-m', 'tact', 'evaluate',
                       '--run-id', item['run_id'], '--eval-id', EVAL_ID]
            with (DIRECTORY / (arm + '-harness.log')).open('a') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                item.update(state='harness_running', pid=process.pid, command=command)
                save(state)
                print('HARNESS_STARTED', arm, process.pid, flush=True)
                rc = process.wait()
            summary = output / 'summary.json'
            item.update(harness_exit_code=rc, summary_path=str(summary), summary_exists=summary.exists())
            if summary.exists():
                data = read(summary)
                item['summary'] = data
            report = [str(ROOT / '.venvs/runner/bin/python'), '-m', 'tact', 'report',
                      '--run-id', item['run_id']]
            with (DIRECTORY / (arm + '-report.log')).open('a') as log:
                item['report_exit_code'] = subprocess.run(report, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT).returncode
            item['state'] = 'finished' if rc == 0 and item['report_exit_code'] == 0 else 'failed'
            print('HARNESS_EXIT', arm, rc, flush=True)
            save(state)
        save(state)
        if all(item['state'] != 'waiting_rollout' for item in state['arms'].values()):
            break
        time.sleep(30)

if __name__ == '__main__':
    main()
