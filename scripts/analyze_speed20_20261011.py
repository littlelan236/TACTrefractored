from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

from tact.artifacts import now, read_json, write_json
from tact.research.trajectories import TrajectoryReader


parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=Path, default=Path('/home/wangtianyi/tactRefractor/reports/speed20-20261011'))
args = parser.parse_args()
directory = args.directory
root = directory.parents[1]
experiment = read_json(directory / 'experiment.json')
controller = read_json(directory / 'controller.json')
controls = {x['name']: x for x in controller['arms']}


def seconds(start, finish):
    return (datetime.fromisoformat(finish) - datetime.fromisoformat(start)).total_seconds()


def ratio(a, b):
    return a / b if b else None


def fmt(value, digits=2):
    return '—' if value is None else f'{value:,.{digits}f}'


def get_request_fields(path):
    reader = TrajectoryReader(path)
    usage = reader.usage()
    response_rows = list(reader.responses())
    close_repetitions = 0
    missing_reasoning = 0
    for step in reader.steps():
        # reasoning parser consumes the canonical first close. A remaining close
        # in returned content/reasoning indicates a non-canonical repeat. This
        # is a text-level lower bound, not exact token-level trigger attestation.
        reasoning = step['reasoning']
        content = step['content']
        close_repetitions += reasoning.count('</think>') + content.count('</think>')
        missing_reasoning += int(not reasoning.strip())
    timestamps = [m.get('extra', {}).get('timestamp') for m in reader.messages
                  if isinstance(m.get('extra', {}).get('timestamp'), (int, float))]
    return {
        **usage,
        'steps': sum(1 for _ in reader.steps()),
        'tool_calls': sum(len(s['tool_calls']) for s in reader.steps()),
        'extra_close_text_occurrences': close_repetitions,
        'steps_missing_reasoning': missing_reasoning,
        'first_response_timestamp': min(timestamps) if timestamps else None,
        'last_response_timestamp': max(timestamps) if timestamps else None,
    }


def engine_metrics(name):
    path = directory / 'engine-metrics' / name / 'latest.prom'
    timestamp = directory / 'engine-metrics' / name / 'timestamp.json'
    if not path.is_file():
        return None
    sums = Counter()
    outcomes = Counter()
    for line in path.read_text().splitlines():
        if not line.startswith('vllm:'):
            continue
        match = re.match(r'vllm:([^\s{]+)(?:\{([^}]*)\})?\s+(\S+)', line)
        if not match:
            continue
        metric, labels, value = match.groups()
        try:
            number = float(value)
        except ValueError:
            continue
        if not math.isfinite(number):
            continue
        sums[metric] += number
        if metric == 'request_success_total':
            reason = re.search(r'finished_reason="([^"]+)"', labels or '')
            outcomes[reason.group(1) if reason else 'unknown'] += number
    return {
        'sampled_at': read_json(timestamp)['at'] if timestamp.exists() else None,
        'completed_requests': sums['e2e_request_latency_seconds_count'],
        'request_outcomes': dict(outcomes),
        'sampled_prompt_tokens': sums['prompt_tokens_total'],
        'sampled_completion_tokens': sums['generation_tokens_total'],
        'mean_request_latency_seconds': ratio(sums['e2e_request_latency_seconds_sum'], sums['e2e_request_latency_seconds_count']),
        'mean_ttft_seconds': ratio(sums['time_to_first_token_seconds_sum'], sums['time_to_first_token_seconds_count']),
        'mean_inter_token_latency_ms': ratio(1000 * sums['inter_token_latency_seconds_sum'], sums['inter_token_latency_seconds_count']),
        'request_prefill_seconds_sum': sums['request_prefill_time_seconds_sum'],
        'request_decode_seconds_sum': sums['request_decode_time_seconds_sum'],
        'request_queue_seconds_sum': sums['request_queue_time_seconds_sum'],
        'preemptions': sums['num_preemptions_total'],
        'scope': 'Last live Prometheus snapshot, potentially missing final requests. Histogram time sums span concurrent requests and are not GPU busy time.',
    }


all_tasks = []
arm_summaries = []
for arm in experiment['arms']:
    name = arm['name']
    control = controls[name]
    run = root / 'runs' / arm['run_id']
    status = read_json(run / 'status.json') if (run / 'status.json').exists() else {'stage': 'starting'}
    arm_tasks = []
    starts, finishes = [], []
    attempt_cost = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'steps': 0, 'persisted_responses': 0}
    coverage = []
    for iid in experiment['selected_ids']:
        task_dir = run / 'tasks' / iid
        result = read_json(task_dir / 'result.json') if (task_dir / 'result.json').exists() else {}
        attempts = sorted((task_dir / 'attempts').glob('*/*.traj.json'))
        for path in attempts:
            usage = get_request_fields(path)
            for key in attempt_cost:
                attempt_cost[key] += usage[key]
            coverage.append(usage['response_coverage_complete'])
        selected_path = task_dir / 'attempts' / result.get('attempt_id', 'attempt-000') / (iid + '.traj.json')
        final = get_request_fields(selected_path) if selected_path.is_file() else {}
        duration = seconds(result['started_at'], result['finished_at']) if result.get('finished_at') else None
        if result.get('started_at'):
            starts.append(result['started_at'])
        if result.get('finished_at'):
            finishes.append(result['finished_at'])
        row = {'arm': name, 'instance_id': iid, 'exit_status': result.get('exit_status', 'pending'),
               'attempt_id': result.get('attempt_id'), 'task_seconds': duration,
               'steps': final.get('steps', 0), 'tool_calls': final.get('tool_calls', 0),
               'prompt_tokens': final.get('prompt_tokens', 0), 'completion_tokens': final.get('completion_tokens', 0),
               'total_tokens': final.get('total_tokens', 0), 'persisted_responses': final.get('persisted_responses', 0),
               'response_coverage_complete': final.get('response_coverage_complete', False),
               'extra_close_text_occurrences': final.get('extra_close_text_occurrences', 0),
               'steps_missing_reasoning': final.get('steps_missing_reasoning', 0)}
        arm_tasks.append(row)
    completed = sum(x['exit_status'] != 'pending' for x in arm_tasks)
    span_end = max(finishes) if completed == len(arm_tasks) and finishes else now()
    span = seconds(min(starts), span_end) if starts else None
    startup = seconds(control['started_at'], min(starts)) if starts else None
    final_cost = {k: sum(x[k] for x in arm_tasks) for k in attempt_cost}
    durations = [x['task_seconds'] for x in arm_tasks if x['task_seconds'] is not None]
    # Raw engine evidence is retained and summarized without loading large
    # worker logs fully into memory. Registry reports every dispatch; EasySteer
    # contains aggregate replay records and optional diagnostics.
    verification = read_json(run / 'serving/verification.json') if (run / 'serving/verification.json').exists() else None
    evidence = {'files': 0, 'record_types': {}, 'decision_records': 0,
                'decision_request_ids': 0, 'corrected_sites': 0, 'nonzero_delta_samples': 0}
    types = Counter()
    request_ids = set()
    sample_records = []
    service_path = run / 'serving/service.json'
    if service_path.exists():
        service = read_json(service_path)
        evidence_path = service.get('evidence_directory')
        if evidence_path:
            for path in Path(evidence_path).glob('worker-*.jsonl'):
                evidence['files'] += 1
                for line in path.open():
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    types[rec.get('type', 'aggregate')] += 1
                    if rec.get('type') in {'decision', 'request_end', 'request_finished'}:
                        evidence['decision_records'] += 1
                        request_id = rec.get('request_id') or rec.get('req_id')
                        if request_id:
                            request_ids.add(str(request_id))
                        if len(sample_records) < 3:
                            sample_records.append(rec)
                        for values in rec.get('evidence_sums', []):
                            if values and abs(values[-1]) > 0:
                                evidence['nonzero_delta_samples'] += 1
                                evidence['corrected_sites'] += 1
    evidence['record_types'] = dict(types)
    evidence['decision_request_ids'] = len(request_ids)
    evidence['samples'] = sample_records
    summary = {**arm, 'exit_code': control.get('exit_code'), 'status': status,
               'completed_tasks': completed, 'requested_tasks': len(arm_tasks),
               'statuses': dict(Counter(x['exit_status'] for x in arm_tasks)),
               'pipeline_elapsed_seconds': control.get('elapsed_seconds'),
               'server_startup_seconds': startup,
               'rollout_task_span_seconds': span,
               'rollout_timing_final': completed == len(arm_tasks),
               'sum_task_seconds': sum(durations),
               'median_task_seconds': median(durations) if durations else None,
               'final_attempt': final_cost, 'all_attempts': attempt_cost,
               'persisted_response_coverage_complete': bool(coverage) and all(coverage),
               'completion_tokens_per_rollout_second': ratio(attempt_cost['completion_tokens'], span),
               'steps_per_rollout_second': ratio(attempt_cost['steps'], span),
               'rollout_seconds_per_step': ratio(span, attempt_cost['steps']),
               'graph_verification': verification, 'evidence': evidence,
               'engine_metrics': engine_metrics(name),
               'extra_close_text_occurrences': sum(x['extra_close_text_occurrences'] for x in arm_tasks),
               'tasks': arm_tasks}
    arm_summaries.append(summary)
    all_tasks.extend(arm_tasks)

base = arm_summaries[0]
for summary in arm_summaries:
    summary['rollout_time_ratio_vs_baseline'] = ratio(summary['rollout_task_span_seconds'], base['rollout_task_span_seconds'])
    summary['completion_token_ratio_vs_baseline'] = ratio(summary['all_attempts']['completion_tokens'], base['all_attempts']['completion_tokens'])
    summary['step_ratio_vs_baseline'] = ratio(summary['all_attempts']['steps'], base['all_attempts']['steps'])

baseline_tasks = {x['instance_id']: x for x in base['tasks']}
for arm in arm_summaries[1:]:
    pairs = []
    for task in arm['tasks']:
        baseline = baseline_tasks[task['instance_id']]
        if task['task_seconds'] is None or baseline['task_seconds'] is None:
            continue
        pairs.append({
            'instance_id': task['instance_id'],
            'baseline_status': baseline['exit_status'], 'steering_status': task['exit_status'],
            'both_submitted': baseline['exit_status'] == task['exit_status'] == 'Submitted',
            'task_time_ratio': ratio(task['task_seconds'], baseline['task_seconds']),
            'completion_ratio': ratio(task['completion_tokens'], baseline['completion_tokens']),
            'step_ratio': ratio(task['steps'], baseline['steps']),
        })
    submitted = [x for x in pairs if x['both_submitted']]
    arm['paired_task_ratios'] = pairs
    arm['paired_submitted_summary'] = {
        'count': len(submitted),
        'median_task_time_ratio': median(x['task_time_ratio'] for x in submitted) if submitted else None,
        'median_completion_ratio': median(x['completion_ratio'] for x in submitted) if submitted else None,
        'median_step_ratio': median(x['step_ratio'] for x in submitted) if submitted else None,
        'scope': 'Secondary matched Submitted subset, distinct from primary all-20 totals. Task times include shared-host concurrent tool/setup waits and batch effects.',
    }

finished = all(x.get('exit_code') is not None for x in arm_summaries)
result = {'experiment_id': experiment['experiment_id'], 'updated_at': now(), 'finished': finished,
          'all_cli_exits_zero': finished and all(x['exit_code'] == 0 for x in arm_summaries),
          'model': 'Qwen3.6-35B-A3B', 'strategy': 'CAP k=2', 'arms': arm_summaries,
          'limitations': experiment['limitations'],
          'definitions': {
              'rollout_task_span_seconds': 'earliest task-attempt start to latest task-attempt finish; includes container bootstrap/tool execution/model waiting; excludes model startup and server shutdown',
              'pipeline_elapsed_seconds': 'controller monotonic wall time from launching arm to observed CLI exit; includes startup, rollout, shutdown, report; process completion polling up to 5 seconds',
              'total_tokens': 'server-reported prompt + completion, summed over every persisted unique response in all attempts; repeated prompt context counted per model call',
              'completion_tokens_per_rollout_second': 'aggregate generated completion throughput across entire agent rollout, including tool/CPU waits; not pure decoder throughput',
              'steps': 'assistant messages in trajectories; includes final submit step; all-attempt totals shown separately',
              'extra_close_text_occurrences': 'text-level lower bound from returned reasoning/content; parser-consumed canonical close excluded; not exact token event count',
          }}
write_json(directory / 'comparison.json', result)
with (directory / 'per-task.csv').open('w', newline='', encoding='utf-8') as output:
    writer = csv.DictWriter(output, fieldnames=list(all_tasks[0]))
    writer.writeheader()
    writer.writerows(all_tasks)

lines = [f"# {experiment['experiment_id']}", '', f"Updated: {result['updated_at']}",
         f"Finished: {finished}; all CLI exits zero: {result['all_cli_exits_zero']} (agent outcomes listed separately)", '',
         'Qwen3.6-35B-A3B; TP=2 per arm; 4 rollout workers/max sequences per arm; same 20 held-out tasks; original TACT axes; CAP k=2. No harness evaluation in timing.', '',
         '| Arm | Completed | Rollout min | Startup min | Pipeline min | Prompt tokens | Completion tokens | Total tokens | Steps | Completion tok/s |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for arm in arm_summaries:
    cost = arm['all_attempts']
    lines.append(f"| {arm['name']} | {arm['completed_tasks']}/20 | {fmt(ratio(arm['rollout_task_span_seconds'],60))} | {fmt(ratio(arm['server_startup_seconds'],60))} | {fmt(ratio(arm['pipeline_elapsed_seconds'],60))} | {cost['prompt_tokens']:,} | {cost['completion_tokens']:,} | {cost['total_tokens']:,} | {cost['steps']:,} | {fmt(arm['completion_tokens_per_rollout_second'])} |")
lines += ['', 'Rollout time includes Docker setup, tools and model calls. Completion tok/s is effective agent throughput, not isolated decode speed.',
          'Persisted-response tokens may omit unreturned HTTP retries; coverage is checked against agent API call counts.', '',
          '## Conditions and limitations', '']
lines += ['- ' + note for note in experiment['limitations']]
lines += ['', '## Status and evidence', '']
for arm in arm_summaries:
    lines += [f"- {arm['name']}: exit={arm['exit_code']}; statuses={arm['statuses']}; response coverage={arm['persisted_response_coverage_complete']}; graph verification={arm['graph_verification']}."]
lines += ['', '## Native request metrics (last live snapshot)', '',
          '| Arm | Sampled requests | Mean request sec | Mean TTFT sec | Mean inter-token ms | Preemptions |',
          '|---|---:|---:|---:|---:|---:|']
for arm in arm_summaries:
    metric = arm['engine_metrics']
    if metric:
        lines.append(f"| {arm['name']} | {fmt(metric['completed_requests'],0)} | {fmt(metric['mean_request_latency_seconds'])} | {fmt(metric['mean_ttft_seconds'])} | {fmt(metric['mean_inter_token_latency_ms'])} | {fmt(metric['preemptions'],0)} |")
lines += ['', 'Snapshots can precede the final requests; use trajectory totals for complete token/step counts. Per-request times depend on input/output length and batching; they are not a fixed-workload kernel comparison.']
(directory / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'finished': finished, 'arms': [{k: arm[k] for k in [
    'name', 'exit_code', 'completed_tasks', 'statuses', 'rollout_task_span_seconds',
    'server_startup_seconds', 'all_attempts', 'completion_tokens_per_rollout_second',
    'graph_verification', 'persisted_response_coverage_complete']} for arm in arm_summaries]}, indent=2))
