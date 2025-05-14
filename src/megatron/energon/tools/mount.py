# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import io
import stat
from typing import IO, Dict, Iterable, Optional, Union

import click
from ratarmount import FuseMount
from ratarmountcore import (
    FileInfo,
    FileVersionLayer,
    MountSource,
)
from ratarmountcore.fusepy import fuse

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.file_store import WebdatasetFileStore


class EnergonMount(MountSource):
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

        self._all_files = {key: size for key, size in self._wds_filestore.list_all_samples()}

        self._print = print_debug

    def isImmutable(self) -> bool:  # our DB never changes
        return True

    def listDir(self, path: str) -> Optional[Union[Iterable[str], Dict[str, FileInfo]]]:
        if path != "/":
            return None
        result = {}
        for key, size in self._all_files.items():
            result[key] = FileInfo(
                size=size,
                mtime=0,
                mode=0o444 | stat.S_IFREG,
                linkname="",
                uid=0,
                gid=0,
                userdata=[key],
            )
        return result

    def listDirModeOnly(self, path: str) -> Optional[Union[Iterable[str], Dict[str, int]]]:
        if path != "/":
            return None
        result = {}
        for key in self._all_files.keys():
            result[key] = 0o444 | stat.S_IFREG
        return result

    def getFileInfo(self, path: str, fileVersion: int = 0) -> Optional[FileInfo]:
        if path == "/":
            return FileInfo(
                size=0,
                mtime=0,
                mode=0o555 | stat.S_IFDIR,
                linkname="",
                uid=0,
                gid=0,
                userdata=[None],
            )
        else:
            try:
                size = self._all_files[path]
            except KeyError:
                return None

            return FileInfo(
                size=size,
                mtime=0,
                mode=0o444 | stat.S_IFREG,
                linkname="",
                uid=0,
                gid=0,
                userdata=[path],
            )

    def fileVersions(self, path: str) -> int:
        return 1 if self.getFileInfo(path) else 0

    def open(self, fileInfo: FileInfo, buffering: int = -1) -> IO[bytes]:
        if fileInfo.userdata is None:
            return None
        b = self._wds_filestore[fileInfo.userdata[0]]
        return io.BytesIO(b)

    def read(self, fileInfo: FileInfo, size: int, offset: int) -> bytes:
        with self.open(fileInfo, buffering=0) as f:
            f.seek(offset)
            return f.read(size)

    def statfs(self):
        block = 512
        return {"f_bsize": block, "f_frsize": block, "f_namemax": 255}

    def __exit__(self, exc_t, exc_v, exc_tb):
        self._wds_filestore.close()


@click.command(name="mount")
@click.argument(
    "path",
    type=click.Path(path_type=EPath),
)
@click.argument(
    "mountpoint",
    type=click.Path(path_type=EPath),
)
@click.option(
    "--foreground",
    "-f",
    is_flag=True,
    default=False,
    help="Run in foreground",
)
def command(path: EPath, mountpoint: EPath, foreground: bool):
    """
    Mount an energon WebdatasetFileStore at the given mountpoint.

    The PATH should point to the folder with the dataset.
    """

    path = EPath(path)
    mountpoint = EPath(mountpoint)

    mount_source = EnergonMount(path)
    root = FileVersionLayer(mount_source)

    ops = FuseMount(
        pathToMount="/",
        mountPoint=str(mountpoint),
        foreground=foreground,
    )
    ops.mountSource = root

    fuse.FUSE(
        operations=ops,
        mountpoint=str(mountpoint),
        foreground=foreground,
        nothreads=True,
    )
