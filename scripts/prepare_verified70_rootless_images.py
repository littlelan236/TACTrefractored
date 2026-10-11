"""Derive rootless-compatible images by normalizing out-of-range tar ownership.

File bytes, paths, mode bits, symlinks and image runtime config are retained.
Original registry blobs are checksum-verified and transformations are audited.
"""
import copy
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import requests

root = Path.cwd()
cache = root / 'assets/images/verified70-rootless'
cache.mkdir(parents=True, exist_ok=True)
session = requests.Session()
session.trust_env = False
proxy = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
accept = 'application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def request(url, **kwargs):
    error = None
    for _ in range(4):
        try:
            response = session.get(url, proxies=proxy, timeout=(15, 180), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            error = exc
    raise error


def image(iid):
    repository = 'swebench/sweb.eval.x86_64.' + iid.replace('__', '_1776_')
    token = request('https://auth.docker.io/token', params={'service': 'registry.docker.io',
                    'scope': 'repository:' + repository + ':pull'}).json()['token']
    headers = {'Authorization': 'Bearer ' + token, 'Accept': accept}
    base = 'https://registry-1.docker.io/v2/' + repository
    index_response = request(base + '/manifests/latest', headers=headers)
    manifest = index_response.json()
    if 'manifests' in manifest:
        selected = next(m for m in manifest['manifests'] if m.get('platform', {}).get('architecture') == 'amd64')
        manifest_response = request(base + '/manifests/' + selected['digest'], headers=headers)
        manifest = manifest_response.json()
        manifest_digest = selected['digest']
    else:
        manifest_digest = index_response.headers['Docker-Content-Digest']
    folder = cache / iid
    folder.mkdir(exist_ok=True)
    (folder / 'original-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def blob(digest):
        path = cache / (digest.replace(':', '-') + '.blob')
        if not path.exists():
            temporary = path.with_suffix('.download')
            print('DOWNLOAD', iid, digest, flush=True)
            with request(base + '/blobs/' + digest, headers=headers, stream=True) as response, temporary.open('wb') as f:
                for block in response.iter_content(4 * 1024 * 1024):
                    f.write(block)
            assert digest_file(temporary) == digest.split(':')[1]
            temporary.replace(path)
        return path

    config = json.loads(blob(manifest['config']['digest']).read_text())
    original_config = copy.deepcopy(config)
    layers, diff_ids, audits = [], [], []
    for entry in manifest['layers']:
        source = blob(entry['digest'])
        layer = cache / (entry['digest'].replace(':', '-') + '.normalized.tar')
        audit_path = layer.with_suffix('.audit.json')
        if not layer.exists() or not audit_path.exists():
            changes = []
            print('NORMALIZE', entry['digest'], flush=True)
            temporary_layer = layer.with_suffix('.partial')
            with tarfile.open(source, 'r|*') as original, tarfile.open(temporary_layer, 'w') as normalized:
                for member in original:
                    updated = copy.copy(member)
                    updated.pax_headers = dict(member.pax_headers)
                    if member.uid >= 65536 or member.gid >= 65536:
                        changes.append({'path': member.name, 'uid': member.uid, 'gid': member.gid})
                        if member.uid >= 65536:
                            updated.uid = 0
                            updated.uname = 'root'
                            updated.pax_headers.pop('uid', None)
                            updated.pax_headers.pop('uname', None)
                        if member.gid >= 65536:
                            updated.gid = 0
                            updated.gname = 'root'
                            updated.pax_headers.pop('gid', None)
                            updated.pax_headers.pop('gname', None)
                    stream = original.extractfile(member) if member.isfile() else None
                    normalized.addfile(updated, stream)
            temporary_layer.replace(layer)
            audit = {'original_layer': entry['digest'], 'original_blob_bytes': source.stat().st_size,
                     'normalized_sha256': digest_file(layer), 'ownership_changes': changes,
                     'transformation': 'only uid/gid >=65536 normalized to root; file contents copied unchanged'}
            audit_path.write_text(json.dumps(audit, indent=2) + '\n')
        audit = json.loads(audit_path.read_text())
        layers.append(layer)
        diff_ids.append('sha256:' + audit['normalized_sha256'])
        audits.append(audit)
    config['rootfs']['diff_ids'] = diff_ids
    config_data = json.dumps(config, separators=(',', ':')).encode()
    config_name = hashlib.sha256(config_data).hexdigest() + '.json'
    target = 'tact-verified70-rootless/' + iid.replace('__', '-') + ':v1'
    archive = folder / 'image.docker.tar'
    with tarfile.open(archive, 'w') as tar:
        member = tarfile.TarInfo(config_name)
        member.size = len(config_data)
        tar.addfile(member, io.BytesIO(config_data))
        for index, layer in enumerate(layers):
            tar.add(layer, arcname=f'{index}/layer.tar', recursive=False)
        content = json.dumps([{'Config': config_name, 'RepoTags': [target],
                               'Layers': [f'{index}/layer.tar' for index in range(len(layers))]}]).encode()
        member = tarfile.TarInfo('manifest.json')
        member.size = len(content)
        tar.addfile(member, io.BytesIO(content))
    env = dict(os.environ, DOCKER_HOST='unix:///run/user/1002/docker.sock')
    subprocess.run([str(Path.home() / '.local/bin/docker'), 'load', '-i', str(archive)], env=env, check=True)
    audit = {'instance_id': iid, 'original_repository': repository,
             'original_manifest_digest': manifest_digest, 'derived_tag': target,
             'runtime_config_unchanged': original_config['config'] == config['config'],
             'layers': audits}
    (folder / 'derivation.json').write_text(json.dumps(audit, indent=2) + '\n')
    instances_path = root / 'data/swebench/verified70/instances.json'
    rows = json.loads(instances_path.read_text())
    for row in rows:
        if row['instance_id'] == iid:
            row['official_docker_image'] = row['docker_image']
            row['docker_image'] = target
            row['image_derivation'] = str((folder / 'derivation.json').relative_to(root))
    instances_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    path = root / 'data/swebench/verified70/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['file_hashes']['instances.json'] = digest_file(instances_path)
    manifest.setdefault('image_derivations', {})[iid] = str((folder / 'derivation.json').relative_to(root))
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('DERIVED', iid, target, sum(len(a['ownership_changes']) for a in audits), flush=True)


for task in ['matplotlib__matplotlib-25479', 'matplotlib__matplotlib-25775', 'matplotlib__matplotlib-24870']:
    image(task)
