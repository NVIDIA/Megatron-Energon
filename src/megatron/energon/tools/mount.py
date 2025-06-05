# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import os
import stat
from errno import EBADF, ENOENT
from pathlib import Path
from sqlite3 import OperationalError
from typing import Generator, Optional

import click

from megatron.energon.cache.file_store import WebdatasetFileStore
from megatron.energon.epathlib import EPath

MULTI_WARN = "WARNING_SAME_KEY_IN_MULTIPLE_TAR_FILES"


EnergonFS: Optional[type]


try:
    from mfusepy import FUSE, FuseOSError, Operations

    class _EnergonFS(Operations):
        """
        Read-only filesystem that exposes an energon WebdatasetFileStore.
        """

        def __init__(
            self,
            db_path: EPath,
            *,
            sample_folders: bool = False,
            print_debug: int = 0,
            allow_slow_mode: bool = False,
        ) -> None:
            self._sample_folders = sample_folders

            self._wds_filestore = WebdatasetFileStore(db_path)

            self._all_sample_parts = {}
            self._slow_mode = False
            try:
                for key, size, tar_file_id in self._wds_filestore.list_all_sample_parts():
                    if key not in self._all_sample_parts:
                        # Only take the first tar file id
                        self._all_sample_parts[key] = size
            except OperationalError:
                if not allow_slow_mode:
                    raise RuntimeError(
                        "The dataset was prepared with an older version of energon. Either update the dataset, or allow slow mode."
                    )
                else:
                    assert sample_folders, (
                        "Only sample_folders mode is supported when using slow mode."
                    )
                    self._slow_mode = True

            self._samples_with_multiple_tar_files = set()
            self._all_samples = {}
            for key, size, tar_file_id in self._wds_filestore.list_all_samples():
                if key not in self._all_samples:
                    self._all_samples[key] = size
                else:
                    self._samples_with_multiple_tar_files.add(key)

            self._total_size = None

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

        def statfs(self, path: str) -> dict:
            """Return information about the file system.

            This is called when the user runs `df` on the mount point.
            """

            if self._total_size is None:
                print("Computing total size...", end="", flush=True)
                self._total_size = self._wds_filestore.get_total_size()
                print(f"done: {self._total_size} bytes")

            return dict(
                f_bsize=512,
                f_blocks=self._total_size // 512,
                f_bavail=0,
                f_bfree=0,
                f_files=len(self._all_sample_parts) if not self._slow_mode else 0,
                f_ffree=0,
                f_namemax=1024,
            )

        def getattr(self, path: str, fh: int = 0) -> dict:
            """Return information about one file or folder.

            This is called when using `ls -l` etc.

            Returns a dict with the following keys:
            - st_mode: File mode (S_IFDIR, S_IFREG, etc.)
            - st_nlink: Number of links
            - st_size: Size of the file
            - st_ctime: Creation time
            - st_mtime: Modification time
            - st_atime: Access time
            - st_uid: User ID of the file
            - st_gid: Group ID of the file
            """

            if path[0] != "/":
                raise FuseOSError(ENOENT)

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

            if path.endswith(MULTI_WARN):
                return dict(
                    st_mode=0o000 | stat.S_IFBLK,
                    st_nlink=1,
                    st_size=0,
                    st_ctime=self._mtime,
                    st_mtime=self._mtime,
                )

            if self._sample_folders:
                folder, part_name = self._path_parts(path)
                if part_name != "":
                    # This is a sample part (file)
                    if folder not in self._all_samples:
                        raise FuseOSError(ENOENT)

                    full_name = f"{folder}.{part_name}"

                    if self._slow_mode and full_name not in self._all_sample_parts:
                        # Slow mode
                        for entry, size, tar_file_id in self._wds_filestore.list_sample_parts(
                            folder, slow_mode=True
                        ):
                            cur_full_name = f"{folder}.{entry}"
                            self._all_sample_parts[cur_full_name] = size

                    if full_name not in self._all_sample_parts:
                        raise FuseOSError(ENOENT)
                    file_size = self._all_sample_parts[full_name]
                    mode = 0o444 | stat.S_IFREG
                else:
                    # This is a sample (directory)
                    if path not in self._all_samples:
                        raise FuseOSError(ENOENT)
                    file_size = self._all_samples[path]
                    mode = 0o555 | stat.S_IFDIR
            else:
                if path not in self._all_sample_parts:
                    raise FuseOSError(ENOENT)
                file_size = self._all_sample_parts[path]
                mode = 0o444 | stat.S_IFREG

            return dict(
                st_mode=mode,
                st_nlink=1,
                st_size=file_size,
                st_ctime=self._mtime,
                st_mtime=self._mtime,
                st_atime=self._mtime,
                st_uid=self._uid,
                st_gid=self._gid,
            )

        def _path_parts(self, path: str) -> tuple[str, str]:
            """Split a path into a folder and a part name and check for errors.
            We only allow paths of the form "sample_key/part_name".
            The leading "/" must be stripped before.
            """

            path_parts = path.split("/")
            # path_parts [0] == "sample_key"
            # path_parts [1] == "part_name"

            if len(path_parts) > 2:
                raise FuseOSError(ENOENT)

            if len(path_parts) == 1:
                part_name = ""
            else:
                part_name = path_parts[1]

            return path_parts[0], part_name

        def readdir(self, path: str, fh: int = 0) -> Generator[str, None, None]:
            """List the contents of a directory.

            This is called when using `ls` etc.

            Returns a generator of the entries in the directory as strings.
            """

            if path[0] != "/":
                raise FuseOSError(ENOENT)

            path = path[1:]

            if self._sample_folders:
                if path == "":
                    yield "."
                    yield ".."
                    for entry in self._all_samples.keys():
                        yield entry
                else:
                    folder, part_name = self._path_parts(path)

                    if folder not in self._all_samples or part_name != "":
                        raise FuseOSError(ENOENT)

                    yield "."
                    yield ".."

                    single_tar_id = None
                    all_entries = list(
                        self._wds_filestore.list_sample_parts(folder, slow_mode=self._slow_mode)
                    )
                    for entry, size, tar_file_id in all_entries:
                        if single_tar_id is None:
                            single_tar_id = tar_file_id
                        elif single_tar_id != tar_file_id:
                            break
                        yield entry

                    if folder in self._samples_with_multiple_tar_files:
                        yield MULTI_WARN
            else:
                if path != "":
                    # Only "/" is allowed for listing all sample parts
                    raise FuseOSError(ENOENT)
                yield "."
                yield ".."
                for entry in self._all_sample_parts.keys():
                    yield entry
                for key in self._samples_with_multiple_tar_files:
                    yield f"{key}.{MULTI_WARN}"

        def open(self, path: str, flags: int = 0) -> int:
            """Open a file for reading.

            Actually, we already read the file into memory when it is opened.
            The read operation just returns a slice of the memory buffer.

            Returns a dummy file descriptor.
            """

            if path[0] != "/":
                raise FuseOSError(ENOENT)

            path = path[1:]

            # read-only: deny write flags
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND):
                raise FuseOSError(ENOENT)

            if self._sample_folders:
                folder, part_name = self._path_parts(path)
                if folder not in self._all_samples:
                    raise FuseOSError(ENOENT)
                full_name = f"{folder}.{part_name}"
                file_bytes, _ = self._wds_filestore[full_name]
            else:
                if path not in self._all_sample_parts:
                    raise FuseOSError(ENOENT)
                file_bytes, _ = self._wds_filestore[path]

            assert isinstance(file_bytes, bytes)
            self._open_files[path] = file_bytes

            # dummy file handle
            return 0

        def read(self, path: str, size: int, offset: int, fh: int = 0) -> bytes:
            """Read from an open file.

            This is called when using `read` etc.

            Returns the bytes object of a previously opened file.
            """

            if path[0] != "/":
                raise FuseOSError(EBADF)

            path = path[1:]

            if path not in self._open_files:
                raise FuseOSError(ENOENT)

            data = self._open_files[path]
            return data[offset : offset + size]

        def release(self, path: str, fh: int = 0) -> None:
            """Release an open file.

            This is called when the file is closed. We can now discard the memory buffer.
            """

            if path[0] != "/":
                raise FuseOSError(ENOENT)

            path = path[1:]

            if path not in self._open_files:
                raise FuseOSError(ENOENT)
            del self._open_files[path]

        def destroy(self, path: str) -> None:
            print("Closing energon mount.")
            if len(self._open_files) > 0:
                print(f"Number of still open files: {len(self._open_files)}")
            self._wds_filestore.close()

    EnergonFS = _EnergonFS
except (ImportError, OSError):
    # mfusepy or fuse not installed, so we can't mount the filesystem
    EnergonFS = None


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
@click.option(
    "--allow-slow",
    "-w",
    is_flag=True,
    default=False,
    help="Allow slow mode. This is useful if the dataset was prepared with an older version of energon.",
)
def command(path: EPath, mountpoint: Path, detach: bool, sample_folders: bool, allow_slow: bool):
    """
    Mount an energon WebdatasetFileStore at the given mountpoint.

    The PATH should point to the folder with the dataset.
    """

    path = EPath(path)
    mountpoint = Path(mountpoint)
    print(f"Mounting {path} at {mountpoint}...")
    mountpoint.mkdir(parents=True, exist_ok=True)

    if EnergonFS is not None:
        energon_fs = EnergonFS(path, sample_folders=sample_folders, allow_slow_mode=allow_slow)
    else:
        print("mfusepy or fuse not installed, so we can't mount the filesystem.")
        return

    print("Mount is ready to go.")

    try:
        FUSE(
            operations=energon_fs,
            mountpoint=str(mountpoint),
            foreground=not detach,
            nothreads=True,
            ro=True,
        )
    except RuntimeError as e:
        if str(e) != "7":
            raise
