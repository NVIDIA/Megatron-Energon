# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import re
import shutil
from pathlib import Path as PathlibPath
from pathlib import PurePosixPath
from typing import BinaryIO, Generator, Literal, Optional, TextIO, Tuple, Union, overload

import multistorageclient as msc

from megatron.energon.errors import warn_deprecated


class EPath:
    """EPath work much like pathlib.Path but does support local as well as remote file systems.
    To initialize a remote path, prepend the path with "msc://" to use the Multi-Storage Client (MSC).
    For example:

        EPath("msc://profilename/my_datasets/webdataset-000.tar")

    You will need to have your MSC configuration (~/.msc_config.yaml) set up to access the object stores
    or use your rclone configuration. See https://nvidia.github.io/multi-storage-client/config/index.html
    for more information.
    """

    # The path without the protocol. Can also be in S3 for example
    internal_path: PurePosixPath
    # The profile used to access the file system
    profile: str
    # The file system
    fs: msc.StorageClient

    def __init__(
        self,
        initial_path: Union[str, "EPath", PathlibPath],
    ) -> None:
        if isinstance(initial_path, EPath):
            self.internal_path = initial_path.internal_path
            self.profile = initial_path.profile
            self.fs = initial_path.fs
        else:
            if isinstance(initial_path, PathlibPath):
                path = str(initial_path.absolute())
                profile = "default"
            else:
                protocol, profile, path = self._split_protocol(initial_path)
                if protocol is None or protocol == "file":
                    profile = "default"
                    path = str(PathlibPath(path).absolute())
                elif protocol == "rclone":
                    warn_deprecated("rclone:// protocol is deprecated. Use msc:// instead.")
                else:
                    assert protocol == "msc", f"Unknown protocol: {protocol}"
            if not path.startswith("/"):
                path = "/" + path
            self.internal_path = self._resolve(path)
            assert profile is not None
            self.profile = profile
            # Resolve the client. Only depends on the protocol and the first part of the path
            self.fs, _ = msc.resolve_storage_client(f"msc://{self.profile}")

    def __getstate__(self) -> dict:
        return {
            "internal_path": self.internal_path,
            "profile": self.profile,
            # Do not save the fs when serializing, to avoid leaking credentials
        }

    def __setstate__(self, state: dict) -> None:
        self.internal_path = state["internal_path"]
        self.profile = state["profile"]
        self.fs, _ = msc.resolve_storage_client(f"msc://{self.profile}")

    @staticmethod
    def _resolve(path: Union[str, PurePosixPath]) -> PurePosixPath:
        """Resolve a path, removing .. and . components."""
        if isinstance(path, str):
            path = PurePosixPath(path)
        parts = path.parts
        if parts[0] != "/":
            raise ValueError("Only absolute paths are supported")
        if ".." in parts or "." in parts:
            new_parts = []
            for part in parts[1:]:
                if part == "..":
                    if len(new_parts) == 0:
                        raise ValueError(f"Path above root: {path}")
                    new_parts.pop()
                elif part == ".":
                    pass
                else:
                    new_parts.append(part)
            path = PurePosixPath("/", *new_parts)
        return path

    @staticmethod
    def _split_protocol(path: str) -> Tuple[Optional[str], Optional[str], str]:
        regex = re.compile(r"^(?P<protocol>[a-z]+)://(?P<profile>[^/]+?)/(?P<path>.+)$")
        m = regex.match(path)
        if m is None:
            return None, None, path
        return m.group("protocol"), m.group("profile"), m.group("path")

    @property
    def _internal_str_path(self) -> str:
        """Return the path as used inside the file system, without the protocol and fs part."""
        return str(self.internal_path)

    @overload
    def open(self, mode: Literal["r", "w"] = "r", block_size: Optional[int] = None) -> TextIO: ...

    @overload
    def open(self, mode: Literal["rb", "wb"], block_size: Optional[int] = None) -> BinaryIO: ...

    def open(
        self, mode: Literal["r", "rb", "w", "wb"] = "r", block_size: Optional[int] = None
    ) -> Union[TextIO, BinaryIO]:
        return self.fs.open(self._internal_str_path, mode)

    def read_text(self) -> str:
        with self.open() as f:
            return f.read()

    def read_bytes(self) -> bytes:
        with self.open("rb") as f:
            return f.read()

    def write_text(self, text: str) -> None:
        with self.open("w") as f:
            f.write(text)

    def write_bytes(self, data: bytes) -> None:
        with self.open("wb") as f:
            f.write(data)

    def copy(self, target: "EPath") -> None:
        """Copy a file to a new path, possibly between different file systems.

        Args:
            target: The path to the local file to download to.
        """

        if self.is_file():
            if self.fs == target.fs:
                self.fs.copy(self._internal_str_path, target._internal_str_path)
            elif target.is_local():
                self.fs.download_file(self._internal_str_path, target._internal_str_path)
            elif self.is_local():
                target.fs.upload_file(target._internal_str_path, self._internal_str_path)
            else:
                with self.open("rb") as src_f, target.open("wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
        else:
            inner_path = EPath(self)
            for fpath in self.fs.list(self._internal_str_path):
                inner_path.internal_path = PurePosixPath("/" + fpath.key)
                inner_path.copy(target / inner_path.relative_to(self))

    @property
    def name(self) -> str:
        return self.internal_path.name

    @property
    def parent(self) -> "EPath":
        new_path = EPath(self)
        new_path.internal_path = self.internal_path.parent
        return new_path

    @property
    def url(self) -> str:
        if self.is_local():
            return self._internal_str_path
        int_path_str = str(self.internal_path)
        return f"msc://{self.profile}{int_path_str}"

    def is_local(self) -> bool:
        return self.profile == "default"

    def is_dir(self) -> bool:
        return self.fs.info(self._internal_str_path).type == "directory"

    def is_file(self) -> bool:
        return self.fs.is_file(self._internal_str_path)

    def mkdir(self, exist_ok: bool = True, parents: bool = False):
        pass

    def glob(self, pattern) -> Generator["EPath", None, None]:
        search_path_pattern = (self / pattern)._internal_str_path

        for path in self.fs.glob(search_path_pattern):
            assert isinstance(path, str)

            new_path = EPath(self)
            new_path.internal_path = self._resolve(self.internal_path / PurePosixPath(path))

            yield new_path

    def size(self) -> int:
        return self.fs.info(self._internal_str_path).content_length

    def with_suffix(self, suffix: str) -> "EPath":
        new_path = EPath(self)
        new_path.internal_path = self.internal_path.with_suffix(suffix)
        return new_path

    def move(self, target: "EPath") -> None:
        self.copy(target)
        self.unlink()

    def unlink(self) -> None:
        return self.fs.delete(self._internal_str_path)

    def relative_to(self, other: "EPath") -> str:
        assert self.profile == other.profile, "Can only use relative_to within same profile"

        return str(self.internal_path.relative_to(other.internal_path))

    def __truediv__(self, other: Union[str, "EPath"]) -> "EPath":
        if isinstance(other, EPath):
            # Always absolute
            return other
        if other.startswith("/") or "://" in other:
            return EPath(other)

        new_path = EPath(self)
        new_path.internal_path = self._resolve(self.internal_path / other)
        return new_path

    def __lt__(self, other: "EPath") -> bool:
        assert self.profile == other.profile, "Cannot compare paths from different profiles"

        return self.internal_path < other.internal_path

    def __str__(self) -> str:
        return self.url

    def __repr__(self) -> str:
        return f"EPath({str(self)!r})"

    def __hash__(self) -> int:
        return hash((self.internal_path, self.profile))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, EPath)
            and self.internal_path == other.internal_path
            and self.profile == other.profile
        )
