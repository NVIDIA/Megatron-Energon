import json
from pathlib import Path
from threading import RLock
from typing import Dict, Optional
from uuid import uuid4

__all__ = ["S3State"]


class S3State:
    """A minimal, thread-safe, in-memory representation of an S3 object store.

    Optionally, a *root_dir* can be supplied to persist the store on the local
    file system. The directory structure mirrors the S3 layout::

        <root_dir>/<bucket>/<key>

    Buckets are directories, objects are stored as regular files. Metadata is
    *not* currently persisted beyond the object byte payload.
    """

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self._fs: Dict[str, Dict[str, bytes]] = {}
        self._uploads: Dict[str, _MultipartUpload] = {}
        self._lock = RLock()
        self._root_dir = root_dir
        if self._root_dir is not None:
            self._root_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # ======================================================================
    # Bucket helpers
    # ======================================================================

    def list_buckets(self) -> list[str]:
        with self._lock:
            return sorted(self._fs.keys())

    def create_bucket(self, bucket: str) -> None:
        with self._lock:
            if bucket in self._fs:
                print(f"Bucket '{bucket}' already exists")
                return
            self._fs[bucket] = {}
        if self._root_dir is not None:
            (self._root_dir / bucket).mkdir(parents=True, exist_ok=True)

    def delete_bucket(self, bucket: str) -> None:
        with self._lock:
            if bucket not in self._fs:
                raise KeyError(f"Bucket '{bucket}' does not exist")
            if self._fs[bucket]:
                raise RuntimeError("Bucket not empty")
            del self._fs[bucket]
        if self._root_dir is not None:
            bucket_path = self._root_dir / bucket
            if bucket_path.exists():
                for p in bucket_path.rglob("*"):
                    p.unlink()
                bucket_path.rmdir()

    # ======================================================================
    # Object helpers
    # ======================================================================

    def put_object(self, bucket: str, key: str, data: bytes) -> None:
        if not bucket:
            raise ValueError("Bucket name must be given")
        with self._lock:
            if bucket not in self._fs:
                self._fs[bucket] = {}
            self._fs[bucket][key] = data
        if self._root_dir is not None:
            obj_path = (self._root_dir / bucket / key).resolve()
            obj_path.parent.mkdir(parents=True, exist_ok=True)
            obj_path.write_bytes(data)

    def get_object(self, bucket: str, key: str) -> bytes:
        with self._lock:
            try:
                return self._fs[bucket][key]
            except KeyError as exc:
                raise FileNotFoundError(f"{bucket}/{key}") from exc

    def delete_object(self, bucket: str, key: str) -> None:
        with self._lock:
            try:
                del self._fs[bucket][key]
            except KeyError as exc:
                raise FileNotFoundError(f"{bucket}/{key}") from exc
        if self._root_dir is not None:
            obj_path = self._root_dir / bucket / key
            if obj_path.exists():
                obj_path.unlink(missing_ok=True)

    def list_objects(self, bucket: str) -> list[str]:
        with self._lock:
            if bucket not in self._fs:
                raise KeyError(f"Bucket '{bucket}' does not exist")
            return sorted(self._fs[bucket].keys())

    # ======================================================================
    # Persistence helpers
    # ======================================================================

    STATE_FILE = "__state.json"

    def _load_from_disk(self) -> None:
        """Load persisted state from *root_dir*.

        The object payload itself is not loaded in memory to keep startup
        affordable. Only the structure (bucket -> keys) is persisted in a
        *state* file.
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
            self._fs = {bucket: {key: b"" for key in keys} for bucket, keys in mapping.items()}

    def flush(self) -> None:
        """Persist *only* the structure of the store to disk."""
        if self._root_dir is None:
            return
        mapping = {bucket: list(objects.keys()) for bucket, objects in self._fs.items()}
        (self._root_dir / self.STATE_FILE).write_text(json.dumps(mapping))

    # ======================================================================
    # Multipart upload helpers
    # ======================================================================

    def initiate_multipart(self, bucket: str, key: str) -> str:  # noqa: D401
        """Create a new multipart upload and return its *upload_id*."""
        with self._lock:
            upload_id = uuid4().hex
            self._uploads[upload_id] = _MultipartUpload(bucket, key)
            # Ensure bucket exists for when we eventually complete
            if bucket not in self._fs:
                self._fs[bucket] = {}
            return upload_id

    def upload_part(self, upload_id: str, part_number: int, data: bytes) -> None:  # noqa: D401
        with self._lock:
            mp = self._uploads.get(upload_id)
            if mp is None:
                raise KeyError("Invalid upload_id")
            mp.parts[part_number] = data

    def complete_multipart(self, upload_id: str) -> None:  # noqa: D401
        with self._lock:
            mp = self._uploads.pop(upload_id, None)
            if mp is None:
                raise KeyError("Invalid upload_id")
            data = mp.assemble()
            if mp.bucket not in self._fs:
                self._fs[mp.bucket] = {}
            self._fs[mp.bucket][mp.key] = data
        # Persist data on disk if configured
        if self._root_dir is not None:
            obj_path = (self._root_dir / mp.bucket / mp.key).resolve()
            obj_path.parent.mkdir(parents=True, exist_ok=True)
            obj_path.write_bytes(data)

    def abort_multipart(self, upload_id: str) -> None:  # noqa: D401
        with self._lock:
            self._uploads.pop(upload_id, None)

    # ======================================================================
    # File helpers
    # ======================================================================

    def add_file(self, src: Path, dst: str):
        if src.is_dir():
            dst = dst.removesuffix("/")
            for file in src.iterdir():
                self.add_file(file, dst=f"{dst}/{file.name}")
        elif src.is_file():
            bucket, key = dst.removeprefix("/").split("/", 1)
            self.put_object(bucket, key, src.read_bytes())
        else:
            raise ValueError(f"Invalid file: {src}")


# ----------------------------------------------------------------------
# Internal helper data structures
# ----------------------------------------------------------------------


class _MultipartUpload:
    __slots__ = ("bucket", "key", "parts")

    def __init__(self, bucket: str, key: str):
        self.bucket = bucket
        self.key = key
        self.parts: Dict[int, bytes] = {}

    def assemble(self) -> bytes:  # noqa: D401
        if not self.parts:
            return b""
        return b"".join(self.parts[n] for n in sorted(self.parts))
