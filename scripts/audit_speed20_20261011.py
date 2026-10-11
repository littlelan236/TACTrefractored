from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from tact.artifacts import now, read_json, sha256, write_json
from tact.config import load_resolved
from tact.research.trajectories import TrajectoryReader


root = Path('/home/wangtianyi/tactRefractor')
directory = root / 'reports/speed20-20261011'
experiment = read_json(directory / 'experiment.json')
configs = [load_resolved(root / 'runs' / arm['run_id'] / 'resolved-config.json') for arm in experiment['arms']]
matched_keys = ['model_descriptor', 'input_hashes', 'selected_ids', 'generation', 'agent', 'environment', 'timeouts', 'execution', 'template_hash']
matched = {}
for key in matched_keys:
    matched[key] = all(cfg[key] == configs[0][key] for cfg in configs)
    assert matched[key], key
for key in ['tensor_parallel_size', 'max_num_seqs', 'max_num_batched_tokens', 'gpu_memory_utilization', 'enable_prefix_caching', 'tool_parser', 'reasoning_parser', 'language_model_only']:
    matched['serving.' + key] = all(cfg['serving'][key] == configs[0]['serving'][key] for cfg in configs)
    assert matched['serving.' + key], key

probes = {}
behavior = []
for arm in experiment['arms']:
    run = root / 'runs' / arm['run_id']
    for iid in experiment['selected_ids']:
        task = run / 'tasks' / iid
        result_path = task / 'result.json'
        result = read_json(result_path) if result_path.exists() else {}
        attempt = result.get('attempt_id', 'attempt-000')
        base = task / 'attempts' / attempt
        env_path = base / 'environment.json'
        files = [p for p in base.glob('*') if p.is_file()]
        probe_file = next((p for p in files if 'probe' in p.name and p.suffix == '.json'), env_path if env_path.exists() else None)
        if probe_file:
            probes.setdefault(iid, {})[arm['name']] = read_json(probe_file)
        traj = base / (iid + '.traj.json')
        if not traj.is_file():
            continue
        reader = TrajectoryReader(traj)
        signatures = []
        for step in reader.steps():
            signature = {'reasoning': step['reasoning'], 'content': step['content'],
                         'commands': [tc.get('function') for tc in step['tool_calls']]}
            signatures.append(hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
        longest = current = 0
        previous = None
        for signature in signatures:
            current = current + 1 if signature == previous else 1
            previous = signature
            longest = max(longest, current)
        frequencies = Counter(signatures)
        behavior.append({'arm': arm['name'], 'instance_id': iid, 'exit_status': result.get('exit_status', 'pending'),
                         'steps': len(signatures), 'longest_identical_consecutive_steps': longest,
                         'most_frequent_identical_step_count': max(frequencies.values()) if frequencies else 0,
                         'trajectory_sha256': sha256(traj)})
environment_checks = []
for iid, values in probes.items():
    if len(values) != 3:
        continue
    fields = ['spec_hash', 'policy_hash', 'packages_hash', 'python', 'executable', 'conda', 'constraints', 'variables']
    checks = {key: len({json.dumps(value.get(key), sort_keys=True) for value in values.values()}) == 1 for key in fields}
    assert all(checks.values()), (iid, checks)
    environment_checks.append({'instance_id': iid, 'checks': checks})
write_json(directory / 'final-audit.json', {'at': now(), 'matching_checks': matched,
            'trajectory_audit': behavior, 'probe_files_available_tasks': len(probes),
            'environment_checks': environment_checks,
            'loop_definition': 'Exact equality of reasoning/content/tool function names and arguments, ignoring call IDs; a lower bound on semantic repetition.'})
print(json.dumps({'matching_checks': matched, 'long_repeats': [x for x in behavior if x['longest_identical_consecutive_steps'] >= 5]}, indent=2))
