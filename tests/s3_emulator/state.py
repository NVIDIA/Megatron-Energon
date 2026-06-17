# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, Optional
from uuid import uuid4

__all__ = ["S3State"]


class S3State:
    """A minimal, thread-safe, in-memory representation of an S3 object store.

    Optionally, a root_dir can be supplied to persist the store on the local
    file system. The directory structure mirrors the S3 layout:

        <root_dir>/<bucket>/<key>

    Buckets are directories, objects are stored as regular files. Metadata is
    not currently persisted beyond the object byte payload.
    """

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        """
        Args:
            root_dir: Path to persist the store on disk.
        """
        self._fs: Dict[str, Dict[str, bytes]] = {}
        self._last_modified: Dict[str, Dict[str, datetime]] = {}
        self._uploads: Dict[str, _MultipartUpload] = {}
        self._lock = RLock()
        self._root_dir = root_dir
        if self._root_dir is not None:
            self._root_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def list_buckets(self) -> list[str]:
        """List all buckets in the store.

        Returns:
            Sorted list of bucket names.
        """
        with self._lock:
            return sorted(self._fs.keys())

    def create_bucket(self, bucket: str) -> None:
        """Create a new bucket.

        Args:
            bucket: Name of the bucket to create.
        """
        with self._lock:
            if bucket in self._fs:
                print(f"Bucket '{bucket}' already exists")
                return
            self._fs[bucket] = {}
            self._last_modified[bucket] = {}
        if self._root_dir is not None:
            (self._root_dir / bucket).mkdir(parents=True, exist_ok=True)

    def delete_bucket(self, bucket: str) -> None:
        """Delete a bucket.

        Args:
            bucket: Name of the bucket to delete.
        """
        with self._lock:
            if bucket not in self._fs:
                raise KeyError(f"Bucket '{bucket}' does not exist")
            if self._fs[bucket]:
                raise RuntimeError("Bucket not empty")
            del self._fs[bucket]
            del self._last_modified[bucket]
        if self._root_dir is not None:
            bucket_path = self._root_dir / bucket
            if bucket_path.exists():
                for p in bucket_path.rglob("*"):
                    p.unlink()
                bucket_path.rmdir()

    def put_object(
        self, bucket: str, key: str, data: bytes, *, last_modified: datetime | None = None
    ) -> None:
        """Store an object in a bucket.

        Args:
            bucket: Name of the bucket.
            key: Object key.
            data: Object data.
            last_modified: Optional timestamp for the object. Defaults to now.
        """
        if not bucket:
            raise ValueError("Bucket name must be given")
        if last_modified is None:
            last_modified = datetime.now(timezone.utc)
        else:
            last_modified = last_modified.astimezone(timezone.utc)
        with self._lock:
            if bucket not in self._fs:
                self._fs[bucket] = {}
                self._last_modified[bucket] = {}
            self._fs[bucket][key] = data
            self._last_modified[bucket][key] = last_modified
        if self._root_dir is not None:
            obj_path = (self._root_dir / bucket / key).resolve()
            obj_path.parent.mkdir(parents=True, exist_ok=True)
            obj_path.write_bytes(data)
            os.utime(obj_path, (last_modified.timestamp(), last_modified.timestamp()))

    def get_object(self, bucket: str, key: str) -> bytes:
        """Retrieve an object from a bucket.

        Args:
            bucket: Name of the bucket.
            key: Object key.

        Returns:
            The object data.
        """
        with self._lock:
            try:
                return self._fs[bucket][key]
            except KeyError as exc:
                raise FileNotFoundError(f"{bucket}/{key}") from exc

    def get_object_last_modified(self, bucket: str, key: str) -> datetime:
        """Return the stored Last-Modified timestamp for an object."""

        with self._lock:
            try:
                return self._last_modified[bucket][key]
            except KeyError as exc:
                raise FileNotFoundError(f"{bucket}/{key}") from exc

    def copy_object(self, dest_bucket: str, dest_key: str, src_bucket: str, src_key: str) -> bytes:
        """Copy an object to another key (S3 CopyObject).

        Args:
            dest_bucket: Destination bucket.
            dest_key: Destination object key.
            src_bucket: Source bucket.
            src_key: Source object key.

        Returns:
            Copied object bytes (for ETag in the CopyObject XML response).

        Raises:
            FileNotFoundError: If the source object does not exist.
        """
        with self._lock:
            try:
                payload = bytes(self._fs[src_bucket][src_key])
            except KeyError as exc:
                raise FileNotFoundError(f"{src_bucket}/{src_key}") from exc
            if dest_bucket not in self._fs:
                self._fs[dest_bucket] = {}
                self._last_modified[dest_bucket] = {}
            last_modified = datetime.now(timezone.utc)
            self._fs[dest_bucket][dest_key] = payload
            self._last_modified[dest_bucket][dest_key] = last_modified
        if self._root_dir is not None:
            obj_path = (self._root_dir / dest_bucket / dest_key).resolve()
            obj_path.parent.mkdir(parents=True, exist_ok=True)
            obj_path.write_bytes(payload)
            os.utime(obj_path, (last_modified.timestamp(), last_modified.timestamp()))
        return payload

    def delete_object(self, bucket: str, key: str) -> None:
        """Delete an object from a bucket.

        Args:
            bucket: Name of the bucket.
            key: Object key.
        """
        with self._lock:
            try:
                del self._fs[bucket][key]
                del self._last_modified[bucket][key]
            except KeyError as exc:
                raise FileNotFoundError(f"{bucket}/{key}") from exc
        if self._root_dir is not None:
            obj_path = self._root_dir / bucket / key
            if obj_path.exists():
                obj_path.unlink(missing_ok=True)

    def list_objects(self, bucket: str) -> list[str]:
        """List all objects in a bucket.

        Args:
            bucket: Name of the bucket.

        Returns:
            Sorted list of object keys.
        """
        with self._lock:
            if bucket not in self._fs:
                raise KeyError(f"Bucket '{bucket}' does not exist")
            return sorted(self._fs[bucket].keys())

    STATE_FILE = "__state.json"

    def _load_from_disk(self) -> None:
        """Load persisted state from root_dir.

        The object payload itself is not loaded in memory to keep startup
        affordable. Only the structure (bucket -> keys) is persisted in a
        state file.
        """
        if self._root_dir is None:
            return
        state_file = self._root_dir / self.STATE_FILE
        if not state_file.exists():
            return
        try:
            mapping = json.loads(state_file.read_text())
        except Exception as err:  # noqa: BLE001
            print(f"Failed to read persisted state: {err}")
            return
        with self._lock:
            self._fs = {}
            self._last_modified = {}
            for bucket, keys in mapping.items():
                bucket_path = self._root_dir / bucket
                self._fs[bucket] = {}
                self._last_modified[bucket] = {}
                for key in keys:
                    object_path = bucket_path / key
                    self._fs[bucket][key] = (
                        object_path.read_bytes() if object_path.is_file() else b""
                    )
                    self._last_modified[bucket][key] = datetime.fromtimestamp(
                        object_path.stat().st_mtime if object_path.exists() else 0,
                        tz=timezone.utc,
                    )

    def flush(self) -> None:
        """Persist only the structure of the store to disk."""
        if self._root_dir is None:
            return
        mapping = {bucket: list(objects.keys()) for bucket, objects in self._fs.items()}
        (self._root_dir / self.STATE_FILE).write_text(json.dumps(mapping))

    def initiate_multipart(self, bucket: str, key: str) -> str:
        """Create a new multipart upload.

        Args:
            bucket: Name of the bucket.
            key: Object key.

        Returns:
            The upload ID.
        """
        with self._lock:
            upload_id = uuid4().hex
            self._uploads[upload_id] = _MultipartUpload(bucket, key)
            if bucket not in self._fs:
                self._fs[bucket] = {}
                self._last_modified[bucket] = {}
            return upload_id

    def upload_part(self, upload_id: str, part_number: int, data: bytes) -> None:
        """Upload a part of a multipart upload.

        Args:
            upload_id: The upload ID.
            part_number: The part number.
            data: The part data.
        """
        with self._lock:
            mp = self._uploads.get(upload_id)
            if mp is None:
                raise KeyError("Invalid upload_id")
            mp.parts[part_number] = data

    def complete_multipart(self, upload_id: str) -> None:
        """Complete a multipart upload.

        Args:
            upload_id: The upload ID.
        """
        with self._lock:
            mp = self._uploads.pop(upload_id, None)
            if mp is None:
                raise KeyError("Invalid upload_id")
            data = mp.assemble()
            if mp.bucket not in self._fs:
                self._fs[mp.bucket] = {}
                self._last_modified[mp.bucket] = {}
            last_modified = datetime.now(timezone.utc)
            self._fs[mp.bucket][mp.key] = data
            self._last_modified[mp.bucket][mp.key] = last_modified
        if self._root_dir is not None:
            obj_path = (self._root_dir / mp.bucket / mp.key).resolve()
            obj_path.parent.mkdir(parents=True, exist_ok=True)
            obj_path.write_bytes(data)
            os.utime(obj_path, (last_modified.timestamp(), last_modified.timestamp()))

    def abort_multipart(self, upload_id: str) -> None:
        """Abort a multipart upload.

        Args:
            upload_id: The upload ID.
        """
        with self._lock:
            self._uploads.pop(upload_id, None)

    def add_file(self, src: Path, dst: str):
        """Add a file or directory to the store.

        Args:
            src: Source file or directory path.
            dst: Destination path in S3 format (bucket/key).
        """
        if src.is_dir():
            dst = dst.removesuffix("/")
            for file in src.iterdir():
                self.add_file(file, dst=f"{dst}/{file.name}")
        elif src.is_file():
            bucket, key = dst.removeprefix("/").split("/", 1)
            last_modified = datetime.fromtimestamp(src.stat().st_mtime, tz=timezone.utc)
            self.put_object(bucket, key, src.read_bytes(), last_modified=last_modified)
        else:
            raise ValueError(f"Invalid file: {src}")


class _MultipartUpload:
    """Internal helper class for managing multipart uploads."""

    __slots__ = ("bucket", "key", "parts")

    def __init__(self, bucket: str, key: str):
        self.bucket = bucket
        self.key = key
        self.parts: Dict[int, bytes] = {}

    def assemble(self) -> bytes:
        """Assemble the uploaded parts into a complete object.

        Returns:
            The complete object data.
        """
        if not self.parts:
            return b""
        return b"".join(self.parts[n] for n in sorted(self.parts))
