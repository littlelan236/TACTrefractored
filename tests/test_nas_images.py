import hashlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock

import docker
import pytest

from tact.artifacts import write_json
from tact.environments.containers import ContainerFactory
from tact.environments import nas_images as module
from tact.environments.nas_images import NasImageStore, GIB


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "NAS_MOUNT", tmp_path)
    monkeypatch.setattr(module.os.path, "ismount", lambda p: True)
    root = tmp_path / "store"
    blobs = root / "blobs/sha256"
    blobs.mkdir(parents=True)
    def blob(content):
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        (blobs / digest[7:]).write_bytes(content)
        return {"digest": digest, "size": len(content)}
    config = blob(b'{"rootfs":{"type":"layers","diff_ids":[]}}')
    layer = blob(b"layer")
    manifest = blob(json.dumps({"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json", "config": config, "layers": [layer]}).encode())
    record = {"instance_id": "test__repo-1", "image_id": manifest["digest"], "config": config,
              "layers": [layer], "manifest_digest": manifest["digest"], "manifest_size": manifest["size"],
              "compressed_bytes": layer["size"], "requested": "swebench/test:latest", "repo_digests": []}
    path = root / "images/task.json"
    write_json(path, record)
    row = {"instance_id": record["instance_id"], "nas_image": {"store": str(root), "manifest": "images/task.json", "manifest_sha256": module.sha256(path)}}
    return NasImageStore(row), row


def test_archive_preserves_verified_oci_blobs_and_identity(archive):
    store, _ = archive
    descriptors = store.verify(deep=True)
    stream = io.BytesIO()
    store.write_archive(stream, descriptors)
    stream.seek(0)
    with tarfile.open(fileobj=stream) as tar:
        index = json.load(tar.extractfile("index.json"))
        assert index["manifests"][0]["digest"] == store.record["manifest_digest"]
        for d in descriptors:
            content = tar.extractfile("blobs/sha256/" + d["digest"][7:]).read()
            assert hashlib.sha256(content).hexdigest() == d["digest"][7:]


def test_corrupt_and_incomplete_downloads_cannot_load(archive):
    store, _ = archive
    p = store.blob(store.record["layers"][0]["digest"])
    p.write_bytes(b"xxxxx")
    with pytest.raises(ValueError, match="checksum"):
        store.verify(deep=True)
    p.unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        store.verify()


def test_manifest_tampering_and_traversal_rejected(archive):
    store, row = archive
    row["nas_image"]["manifest"] = "../escape.json"
    with pytest.raises(ValueError, match="escapes"):
        NasImageStore(row)
    row["nas_image"]["manifest"] = "images/task.json"
    store.path.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        NasImageStore(row)


def test_preflight_does_not_import_all_images_and_resume_checks_identity(archive):
    store, row = archive
    factory = ContainerFactory.__new__(ContainerFactory)
    factory.client = Mock()
    factory.images = {}
    identity = factory.lock_images([row])[row["instance_id"]]
    assert identity["image_id"] == store.image_id
    factory.client.images.get.assert_not_called()
    factory.images[row["instance_id"]]["image_id"] = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        factory.lock_images([row])


def test_original_dataset_still_uses_original_local_lookup():
    factory = ContainerFactory.__new__(ContainerFactory)
    factory.client = Mock()
    factory.images = {}
    factory.client.images.get.return_value = SimpleNamespace(id="original-id", attrs={"RepoDigests": ["original-digest"]})
    row = {"instance_id": "old__task-1", "image_name": "original:latest"}
    assert factory.lock_images([row])[row["instance_id"]] == {"image_id": "original-id", "requested": "original:latest", "repo_digests": ["original-digest"]}
    factory.client.images.get.assert_called_once_with("original:latest")


def test_existing_images_are_not_enrolled_for_eviction(archive, tmp_path):
    store, _ = archive
    store.cache = {}
    client = Mock()
    client.images.get.return_value = SimpleNamespace(id=store.image_id)
    store.ensure(client, "docker", "socket")
    assert store.cache == {}
    client.images.remove.assert_not_called()


def test_cache_never_removes_original_or_in_use_images(archive, tmp_path, monkeypatch):
    store, _ = archive
    store.cache = {"new-id": {"last_used": 0, "size": 50 * GIB}}
    store.state = tmp_path
    client = Mock()
    client.info.return_value = {"DockerRootDir": str(tmp_path)}
    client.containers.list.return_value = [object()]
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: SimpleNamespace(free=50 * GIB))
    with pytest.raises(docker.errors.DockerException, match="Insufficient"):
        store.make_room(client, 10 * GIB)
    client.images.remove.assert_not_called()
    assert "new-id" in store.cache


def test_cache_only_removes_tracked_unused_images(archive, tmp_path, monkeypatch):
    store, _ = archive
    store.cache = {"new-id": {"last_used": 0, "size": 50 * GIB}}
    store.state = tmp_path
    client = Mock()
    client.info.return_value = {"DockerRootDir": str(tmp_path)}
    client.containers.list.return_value = []
    usage = iter([SimpleNamespace(free=80 * GIB), SimpleNamespace(free=200 * GIB)])
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: next(usage))
    store.make_room(client, 10 * GIB)
    client.images.remove.assert_called_once_with("new-id", force=False)
    assert not store.cache


def test_rootless_derivation_only_changes_unmappable_ownership(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_verified200_nas.py"
    spec = importlib.util.spec_from_file_location("nas_prepare_test", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "STORE", tmp_path)
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        for name, uid, gid in [("normal", 0, 0), ("build/qhull.txt", 197609, 197121)]:
            member = tarfile.TarInfo(name)
            member.uid, member.gid, member.mode = uid, gid, 0o755
            member.mtime = 12345
            content = ("unchanged contents " + name).encode()
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    original = payload.getvalue()
    digest = "sha256:" + hashlib.sha256(original).hexdigest()
    blobs = tmp_path / "blobs/sha256"
    blobs.mkdir(parents=True)
    source = blobs / digest[7:]
    source.write_bytes(original)
    result = script.rootless_layer({"digest": digest, "size": len(original)}, False)
    assert source.read_bytes() == original
    assert len(result["ownership_changes"]) == 1
    derived = blobs / result["derived"]["digest"][7:]
    assert "sha256:" + hashlib.sha256(gzip.decompress(derived.read_bytes())).hexdigest() == result["diff_id"]
    with tarfile.open(derived, mode="r:gz") as tar:
        for member in tar:
            assert member.uid == member.gid == 0
            assert member.mode == 0o755 and member.mtime == 12345
            assert tar.extractfile(member).read() == ("unchanged contents " + member.name).encode()
