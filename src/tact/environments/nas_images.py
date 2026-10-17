"""Opt-in NAS OCI blob storage with a bounded local Docker image cache.

Only benchmark rows carrying nas_image enter this path. No original image is
pruned, no daemon setting is changed, and no model/runtime parameter is changed.
"""
from __future__ import annotations

from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

import docker

from tact.artifacts import read_json, sha256, write_json

NAS_MOUNT = Path("/VM")
GIB = 1024**3


class NasImageStore:
    def __init__(self, instance: dict):
        reference = instance["nas_image"]
        self.root = Path(reference["store"]).resolve()
        if not self.root.is_relative_to(NAS_MOUNT) or not os.path.ismount(NAS_MOUNT):
            raise ValueError("NAS image store must be below the mounted /VM filesystem")
        self.path = (self.root / reference["manifest"]).resolve()
        if not self.path.is_relative_to(self.root):
            raise ValueError("NAS image manifest escapes its store")
        if sha256(self.path) != reference["manifest_sha256"]:
            raise ValueError("NAS image manifest hash mismatch")
        self.record = read_json(self.path)
        if self.record["instance_id"] != instance["instance_id"]:
            raise ValueError("NAS image belongs to another task")
        # Docker 29's containerd image store identifies an imported single-platform
        # image by its manifest digest, rather than its OCI config digest.
        self.image_id = self.record["image_id"]
        if self.image_id != self.record["manifest_digest"]:
            raise ValueError("NAS image/manifest identity mismatch")

    def blob(self, digest: str) -> Path:
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError("Only SHA256 OCI blobs are supported")
        int(digest[7:], 16)
        path = (self.root / "blobs/sha256" / digest[7:]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("NAS blob escapes its store")
        return path

    def identity(self) -> dict:
        return {"image_id": self.image_id, "requested": self.record["requested"],
                "repo_digests": self.record["repo_digests"],
                "nas_manifest_sha256": sha256(self.path)}

    def verify(self, deep=False):
        descriptors = [self.record["config"], *self.record["layers"],
                       {"digest": self.record["manifest_digest"],
                        "size": self.record["manifest_size"]}]
        for descriptor in descriptors:
            path = self.blob(descriptor["digest"])
            if not path.is_file() or path.stat().st_size != descriptor["size"]:
                raise FileNotFoundError(f"NAS image download incomplete: {path}")
            if deep and "sha256:" + sha256(path) != descriptor["digest"]:
                raise ValueError(f"NAS blob checksum mismatch: {path}")
        return descriptors

    @contextmanager
    def creation_guard(self, client):
        import fcntl
        # All stores using this daemon share one lock and one ownership journal.
        root = Path(client.info()["DockerRootDir"])
        state = root.parent / "tact-nas-image-cache"
        state.mkdir(exist_ok=True)
        with (state / "lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.state = state
            self.cache = read_json(state / "images.json") if (state / "images.json").exists() else {}
            yield

    def save_cache(self):
        write_json(self.state / "images.json", self.cache)

    def ensure(self, client, executable: str, docker_host: str):
        """Called while creation_guard is held, through container creation."""
        try:
            image = client.images.get(self.image_id)
        except docker.errors.ImageNotFound:
            image = None
        if image is not None:
            if self.image_id in self.cache:
                self.cache[self.image_id]["last_used"] = time.time()
                self.save_cache()
            return
        descriptors = self.verify(deep=True)
        # Conservative headroom for extraction. Running containers are never evicted.
        estimate = max(self.record["compressed_bytes"] * 6, 8 * GIB)
        self.make_room(client, estimate)
        log_path = self.state / (self.image_id[7:] + ".load.log")
        with log_path.open("wb") as output:
            process = subprocess.Popen([executable, "--host", docker_host, "load"],
                                       stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT)
            try:
                self.write_archive(process.stdin, descriptors)
                process.stdin.close()
                code = process.wait(timeout=1800)
            except BaseException:
                process.kill(); process.wait()
                raise
        if code:
            raise docker.errors.DockerException(f"NAS image load failed; see {log_path}")
        image = client.images.get(self.image_id)
        self.cache[self.image_id] = {"store": str(self.root), "requested": self.record["requested"],
                                     "size": image.attrs["Size"], "last_used": time.time()}
        self.save_cache()

    def make_room(self, client, estimate: int):
        root = Path(client.info()["DockerRootDir"])
        reserve = 100 * GIB
        limit = 160 * GIB
        for image_id, entry in sorted(list(self.cache.items()), key=lambda item: item[1]["last_used"]):
            total = sum(v["size"] for v in self.cache.values())
            if shutil.disk_usage(root).free >= reserve + estimate and total + estimate <= limit:
                break
            # Existing non-NAS images never enter this journal. Refuse broad prune.
            if client.containers.list(all=True, filters={"ancestor": image_id}):
                continue
            try:
                client.images.remove(image_id, force=False)
            except docker.errors.ImageNotFound:
                pass
            except docker.errors.APIError:
                continue
            del self.cache[image_id]
            self.save_cache()
        if shutil.disk_usage(root).free < reserve + estimate:
            raise docker.errors.DockerException("Insufficient local space for NAS image; keep 100 GiB reserve. No original images were removed.")
        if sum(v["size"] for v in self.cache.values()) + estimate > limit:
            raise docker.errors.DockerException("NAS image cache capacity reached by protected/in-use images; reduce NAS task concurrency")

    def write_archive(self, stream, descriptors):
        """Stream an OCI archive to Docker 29's containerd image importer.

        Compressed blobs stay compressed; no temporary local image tar is made.
        """
        media_type = read_json(self.blob(self.record["manifest_digest"]))["mediaType"]
        descriptor = {"mediaType": media_type,
                      "digest": self.record["manifest_digest"], "size": self.record["manifest_size"],
                      "annotations": {"io.containerd.image.name": "docker.io/" + self.record["requested"],
                                      "org.opencontainers.image.ref.name": self.record["requested"]}}
        with tarfile.open(fileobj=stream, mode="w|") as archive:
            for name, value in [("oci-layout", {"imageLayoutVersion": "1.0.0"}),
                                ("index.json", {"schemaVersion": 2, "manifests": [descriptor]})]:
                data = json.dumps(value).encode()
                entry = tarfile.TarInfo(name); entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
            for item in descriptors:
                path = self.blob(item["digest"])
                entry = tarfile.TarInfo("blobs/sha256/" + item["digest"][7:])
                entry.size = item["size"]
                with path.open("rb") as source:
                    archive.addfile(entry, source)


def nas_images(cfg, action: str, deep=False, limit=8):
    from tact.config import select_instances
    rows, _ = select_instances(cfg)
    selected = [r for r in rows if r.get("nas_image")]
    if not selected:
        raise ValueError("Selected dataset has no NAS images")
    client = docker.DockerClient(base_url=cfg["environment"]["docker_host"], timeout=120)
    try:
        if action == "status":
            info = client.info()
            local = {image.id for image in client.images.list()}
            return {"tasks": len(selected), "local_images": sum(NasImageStore(r).image_id in local for r in selected),
                    "docker_root": info["DockerRootDir"], "free_GiB": shutil.disk_usage(info["DockerRootDir"]).free / GIB}
        checked = 0
        for row in selected[:limit] if action == "load" else selected:
            store = NasImageStore(row)
            if action == "check":
                store.verify(deep=deep)
            else:
                with store.creation_guard(client):
                    store.ensure(client, cfg["environment"]["docker_executable"], cfg["environment"]["docker_host"])
            checked += 1
        return {"action": action, "checked": checked, "selected": len(selected), "deep": deep}
    finally:
        client.close()
