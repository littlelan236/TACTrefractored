import collections
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/swebench/verified70'
OUT.mkdir(parents=True, exist_ok=True)
repo = 'princeton-nlp/SWE-bench_Verified'
session = requests.Session()
session.trust_env = False

def get(url):
    errors = []
    for proxies in ({'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}, {}):
        for attempt in range(2):
            try:
                print('Fetching', url, flush=True)
                r = session.get(url, timeout=(8, 45), proxies=proxies)
                r.raise_for_status()
                return r
            except requests.RequestException as e:
                errors.append(str(e))
    raise RuntimeError('; '.join(errors))

meta = get(f'https://huggingface.co/api/datasets/{repo}').json()
revision = 'c104f840cc67f8b6eec6f759ebc8b2693d585d4a'
paths = [s['rfilename'] for s in meta['siblings'] if s['rfilename'].startswith('data/test-') and s['rfilename'].endswith('.parquet')]
assert paths, list(meta)
rows = []
sources = []
for path in paths:
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}'
    content = get(url).content
    target = OUT / Path(path).name
    target.write_bytes(content)
    rows.extend(pq.read_table(target).to_pylist())
    sources.append({'url': url, 'file': target.name, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)})
assert len(rows) == 500 and len({r['instance_id'] for r in rows}) == 500
(OUT / 'source-500.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
(OUT / 'source-provenance.json').write_text(json.dumps({'dataset': repo, 'split': 'test', 'revision': revision, 'rows': len(rows), 'files': sources}, indent=2) + '\n')
print(json.dumps({'revision': revision, 'keys': list(rows[0]), 'difficulty': dict(collections.Counter(r.get('difficulty') for r in rows)), 'repos': dict(collections.Counter(r['repo'] for r in rows))}, indent=2))
