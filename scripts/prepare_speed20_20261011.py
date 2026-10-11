from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from tact.artifacts import now, sha256, write_json
from tact.backends.axis_bundle import validate_axis_bundle
from tact.config import resolve_config
from tact.contracts import model_descriptor


root = Path('/home/wangtianyi/tactRefractor')
old = Path('/home/wangtianyi/DualBench')
experiment_id = 'speed20-20261011'
directory = root / 'reports' / experiment_id
directory.mkdir(parents=True, exist_ok=True)
assert not any((root / 'runs').glob(experiment_id + '-*')), 'Run already exists; do not change frozen inputs'
splits = json.loads((root / 'data/swebench/splits.json').read_text())['partitions']
instances = json.loads((root / 'data/swebench/instances.json').read_text())
by_id = {x['instance_id']: x for x in instances}

# A small repository-balanced cohort from held-out eval only. Selection uses old
# baseline length, never the outcomes of this experiment or steering performance.
quotas = {
    'pvlib/pvlib-python': 4,
    'marshmallow-code/marshmallow': 3,
    'pydicom/pydicom': 3,
    'pylint-dev/astroid': 3,
    'django/django': 2,
    'sympy/sympy': 1,
    'sqlfluff/sqlfluff': 1,
    'pydata/xarray': 1,
    'pyvista/pyvista': 2,
}
candidates = []
for iid in splits['eval']:
    path = old / 'tact/traj/tact-env-pinned-191-20261007/agent' / iid / (iid + '.traj.json')
    if not path.is_file():
        continue
    traj = json.loads(path.read_text())
    if traj['info']['exit_status'] != 'Submitted':
        continue
    completions = sum(
        ((m.get('extra') or {}).get('response', {}).get('usage') or {}).get('completion_tokens', 0)
        for m in traj['messages']
    )
    candidates.append({
        'instance_id': iid,
        'repo': by_id[iid]['repo'],
        'old_baseline_steps': sum(m.get('role') == 'assistant' for m in traj['messages']),
        'old_baseline_completion_tokens': completions,
        'old_baseline_exit_status': traj['info']['exit_status'],
    })
selected = []
for repo, count in quotas.items():
    choices = sorted(
        (x for x in candidates if x['repo'] == repo),
        key=lambda x: (x['old_baseline_completion_tokens'], x['instance_id']),
    )[:count]
    assert len(choices) == count, (repo, len(choices), count)
    selected.extend(choices)
# Start longer examples first, preventing the same long tasks becoming late
# stragglers. Each arm uses exactly this frozen order.
selected.sort(key=lambda x: (-x['old_baseline_completion_tokens'], x['instance_id']))
assert len(selected) == 20
ids = [x['instance_id'] for x in selected]
write_json(directory / 'selection.json', {
    'created_at': now(),
    'policy': 'Held-out eval; 9-repository quotas; shortest prior Submitted baseline completions per repo; longest-first execution. Preliminary selected cohort, not a representative full benchmark or speed-only fixed-output control.',
    'quotas': quotas, 'tasks': selected,
})

source = old / 'tact/axes/tact-env-pinned-191-20261007-noloop46-annot-v1/tact/axes.npz'
bundle = root / 'artifacts/axes/original-tact-noloop46-20261011'
bundle.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, bundle / 'original-axes.npz')
descriptor = model_descriptor(root / 'assets/models/Qwen3.6-35B-A3B')
assert descriptor['layer_count'] == 40 and descriptor['hidden_size'] == 2048
old_model = Path('/home/wangtianyi/shared/models/Qwen3.6-35B-A3B')
for name, digest in descriptor['metadata_hashes'].items():
    assert sha256(old_model / name) == digest, ('Original model metadata mismatch', name)
identity = json.loads((root / 'assets/models/Qwen3.6-35B-A3B/tact-identity.json').read_text())
weight_entries = [x for x in identity['files'] if x['destination'].endswith('.safetensors')]
for item in weight_entries:
    original = old_model / Path(item['destination']).relative_to('assets/models/Qwen3.6-35B-A3B')
    assert original.stat().st_size == item['bytes']
with np.load(source, allow_pickle=False) as z:
    arrays = {key: z[key][1:].copy() for key in ['v_ot', 'v_oa_perp', 'mu_cal', 'sigma_ot', 'sigma_oa']}
    selected_indices = [int(i) - 1 for i in z['top_k']]
    for key, array in arrays.items():
        assert np.array_equal(array, z[key][1:]), key
    np.savez_compressed(bundle / 'axes.npz', **arrays)
    old_selected = z['top_k'].tolist()
manifest = {
    'schema_version': 1,
    'bundle_id': bundle.name,
    'model': descriptor,
    'complete': True,
    'selected_indices': selected_indices,
    'selected_sites': [descriptor['sites'][i] for i in selected_indices],
    'axes_sha256': sha256(bundle / 'axes.npz'),
    'created_at': now(),
    'provenance': {
        'source': str(source), 'source_sha256': sha256(source),
        'original_copy_sha256': sha256(bundle / 'original-axes.npz'),
        'original_hf_hidden_indices': old_selected,
        'conversion': 'Drop HF embedding row 0; translate original indices by -1 into hf-hidden-v1. Preserve all tensor values and dtypes; original final index 40 maps to final_norm. No refit, no renormalization.',
        'model_metadata_verified_against_original': True,
        'weight_identity_manifest_sha256': sha256(root / 'assets/models/Qwen3.6-35B-A3B/tact-identity.json'),
    },
}
assert manifest['provenance']['source_sha256'] == manifest['provenance']['original_copy_sha256']
write_json(bundle / 'manifest.json', manifest)

arms = [
    ('baseline', 'native_vllm', [0, 1], 18130, 'none', 'first_generated_think_close'),
    ('registry_cap', 'registry_vllm', [4, 5], 18131, 'cap', 'first_generated_think_close'),
    ('easysteer_cap', 'easysteer', [6, 7], 18132, 'cap', 'all_generated_think_close'),
]
arm_records = []
for name, backend, gpus, port, strategy, trigger in arms:
    config = {
        'schema_version': 1,
        'benchmark': {'selection': ids},
        'model': {'path': 'assets/models/Qwen3.6-35B-A3B', 'served_name': 'tact-speed35b', 'max_model_len': 262144},
        'serving': {
            'backend': backend, 'port': port, 'gpus': gpus, 'tensor_parallel_size': 2,
            'max_num_seqs': 4, 'max_num_batched_tokens': 8192,
            'startup_seconds': 1800, 'enable_prefix_caching': False,
            'graph_requirement': 'auto' if backend == 'native_vllm' else 'required',
            'easysteer_diagnostics': False,
        },
        'generation': json.loads((old / 'configs/runtime.json').read_text()),
        'timeouts': {'container_lifetime_seconds': 32400},
        'execution': {'rollout_workers': 4, 'infrastructure_retry_limit': 0},
        'steering': {'strategy': strategy, 'axes': str(bundle.relative_to(root)) if strategy != 'none' else None,
                     'k': 2.0, 'trigger': trigger},
    }
    generation_keys = ['temperature', 'seed', 'top_p', 'top_k', 'min_p', 'presence_penalty',
                       'repetition_penalty', 'max_tokens', 'enable_thinking', 'preserve_thinking']
    config['generation'] = {k: config['generation'][k] for k in generation_keys}
    config_path = directory / f'{name}.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    resolved = resolve_config(config_path)
    if strategy != 'none':
        validate_axis_bundle(resolved)
    write_json(directory / f'{name}-preflight.json', {
        'valid': True, 'tasks': len(ids), 'model_fingerprint': descriptor['model_fingerprint'],
        'config_hash': resolved['config_hash'], 'serving': resolved['serving'],
        'steering': resolved['steering'], 'generation': resolved['generation'],
    })
    arm_records.append({'name': name, 'run_id': f'{experiment_id}-{name}',
                        'config': str(config_path), 'gpus': gpus, 'port': port, 'backend': backend})
write_json(directory / 'experiment.json', {
    'experiment_id': experiment_id, 'created_at': now(), 'arms': arm_records,
    'selected_ids': ids, 'axes_bundle': str(bundle),
    'timing_scope': 'rollout only; no SWE harness; cold start and shutdown reported separately from task span',
    'parallel_task_workers_per_arm': 4,
    'infrastructure_retry_limit': 0,
    'limitations': ['native/registry vLLM 0.30 versus EasySteer fork 0.29',
                    'registry first-close versus EasySteer all-generated-close',
                    'registry runtime evidence overhead enabled as required by current backend',
                    'shared host CPU/Docker/network and pre-existing unrelated workload',
                    'small length-selected cohort; task generations can diverge and change workload'],
})
print(json.dumps({'directory': str(directory), 'tasks': ids, 'repositories': dict(Counter(x['repo'] for x in selected)),
                  'axes_original_selected': old_selected, 'arms': arm_records}, indent=2))
