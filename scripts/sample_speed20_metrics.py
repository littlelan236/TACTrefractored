from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from tact.artifacts import now, read_json, write_json

directory = Path('/home/wangtianyi/tactRefractor/reports/speed20-20261011')
experiment = read_json(directory / 'experiment.json')
session = requests.Session()
session.trust_env = False
while True:
    controls = read_json(directory / 'controller.json')
    statuses = {x['name']: x for x in controls['arms']}
    for arm in experiment['arms']:
        if statuses[arm['name']]['exit_code'] is not None:
            continue
        try:
            response = session.get(f"http://127.0.0.1:{arm['port']}/metrics", timeout=5)
            response.raise_for_status()
        except requests.RequestException:
            continue
        metric_directory = directory / 'engine-metrics' / arm['name']
        metric_directory.mkdir(parents=True, exist_ok=True)
        metric_file = metric_directory / 'latest.prom'
        temporary = metric_directory / 'latest.tmp'
        temporary.write_text(response.text, encoding='utf-8')
        temporary.replace(metric_file)
        samples = []
        for line in response.text.splitlines():
            if line.startswith('vllm:') and any(part in line for part in (
                'request_success', 'prompt_tokens_total', 'generation_tokens_total',
                'e2e_request_latency_seconds_sum', 'e2e_request_latency_seconds_count',
                'time_to_first_token_seconds_sum', 'time_to_first_token_seconds_count',
                'inter_token_latency_seconds_sum', 'inter_token_latency_seconds_count',
                'request_prefill_time_seconds_sum', 'request_decode_time_seconds_sum',
                'request_queue_time_seconds_sum', 'num_requests_running', 'num_requests_waiting',
                'num_preemptions',
            )):
                samples.append(line)
        with (metric_directory / 'samples.jsonl').open('a') as f:
            f.write(json.dumps({'at': now(), 'metrics': samples}) + '\n')
        write_json(metric_directory / 'timestamp.json', {'at': now()})
    if all(x['exit_code'] is not None for x in controls['arms']):
        break
    time.sleep(20)
