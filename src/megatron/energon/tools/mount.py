# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import os
import stat
from errno import ENOENT
from pathlib import Path

import click
from mfusepy import FUSE, FuseOSError, Operations

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.file_store import WebdatasetFileStore


class EnergonFS(Operations):
    """
    Read-only filesystem that exposes an energon WebdatasetFileStore.
    """

    def __init__(self, db_path: EPath, *, print_debug: int = 0) -> None:
        # First create a temporary directory and copy the db there

        # temp_dir = tempfile.mkdtemp()
        # temp_db_path = Path(temp_dir) / "index.db"
        # shutil.copy(db_path, temp_db_path)
        # self._wds_filestore = WebdatasetFileStore(EPath(temp_db_path))

        self._wds_filestore = WebdatasetFileStore(db_path)

        self._all_sample_parts = {
            key: size for key, size in self._wds_filestore.list_all_sample_parts()
        }

        # When a file is opened, we keep the bytes in memory for now (until it is closed)
        self._open_files = {}

        # Get current uid and gid
        self._uid = os.getuid()
        self._gid = os.getgid()

        # Get modification time of the db file
        try:
            self._mtime = os.path.getmtime(str(db_path))
        except FileNotFoundError:
            # Remote file systems have no modification time
            self._mtime = 0

        self._print = print_debug

    def getattr(self, path: str, fh: int = 0):
        if path[0] != "/":
            raise FuseOSError(ENOENT)

        print(f"getattr: {path}")
        if path == "/":
            return dict(
                st_mode=0o555 | stat.S_IFDIR,
                st_nlink=2,
                st_size=0,
                st_ctime=self._mtime,
                st_mtime=self._mtime,
                st_atime=self._mtime,
                st_uid=self._uid,
                st_gid=self._gid,
            )

        # Strip leading '/'
        path = path[1:]

        if path not in self._all_sample_parts:
            raise FuseOSError(ENOENT)

        file_size = self._all_sample_parts[path]

        return dict(
            st_mode=0o444 | stat.S_IFREG,
            st_nlink=1,
            st_size=file_size,
            st_ctime=self._mtime,
            st_mtime=self._mtime,
            st_atime=self._mtime,
            st_uid=self._uid,
            st_gid=self._gid,
        )

    def readdir(self, path: str, fh: int = 0):
        print(f"readdir: {path}")

        if path != "/":
            raise FuseOSError(ENOENT)

        # '.' and '..' plus our files
        yield "."
        yield ".."
        for entry in self._all_sample_parts.keys():
            yield entry

    def open(self, path: str, flags: int = 0):
        print(f"open: {path}")
        if path[0] != "/":
            raise FuseOSError(ENOENT)

        path = path[1:]

        # read-only: deny write flags
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND):
            raise FuseOSError(ENOENT)
        if path not in self._all_sample_parts:
            raise FuseOSError(ENOENT)
        file_bytes = self._wds_filestore[path]
        assert isinstance(file_bytes, bytes)
        self._open_files[path] = file_bytes

        # dummy file handle
        return 0

    def read(self, path: str, size: int, offset: int, fh: int = 0):
        if path[0] != "/":
            raise FuseOSError(ENOENT)

        path = path[1:]

        if path not in self._all_sample_parts:
            raise FuseOSError(ENOENT)
        data = self._open_files[path]
        return data[offset : offset + size]

    def close(self, path: str, fh: int = 0):
        if path[0] != "/":
            raise FuseOSError(ENOENT)

        path = path[1:]

        if path not in self._open_files:
            raise FuseOSError(ENOENT)
        del self._open_files[path]

    def __exit__(self, exc_t, exc_v, exc_tb):
        print("closing filestore")
        print(f"Number of open files: {len(self._open_files)}")
        self._wds_filestore.close()


@click.command(name="mount")
@click.argument(
    "path",
    type=click.Path(path_type=EPath),
)
@click.argument(
    "mountpoint",
    type=click.Path(path_type=Path),
)
@click.option(
    "--detach",
    "-d",
    is_flag=True,
    default=False,
    help="Run in background",
)
@click.option(
    "--sample-folders",
    "-s",
    is_flag=True,
    default=False,
    help="Present a virtual folder per sample. Otherwise a single folder with all sample parts.",
)
def command(path: EPath, mountpoint: Path, detach: bool, sample_folders: bool):
    """
    Mount an energon WebdatasetFileStore at the given mountpoint.

    The PATH should point to the folder with the dataset.
    """

    path = EPath(path)
    mountpoint = Path(mountpoint)
    print(f"Mounting {path} at {mountpoint}")
    mountpoint.mkdir(parents=True, exist_ok=True)

    energon_fs = EnergonFS(path)

    FUSE(
        operations=energon_fs,
        mountpoint=str(mountpoint),
        foreground=not detach,
        nothreads=True,
    )
