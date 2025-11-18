# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import concurrent.futures
import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Sequence, Set

import click
import numpy as np
from tqdm import tqdm

from megatron.energon.epathlib import EPath

# mp.set_start_method("fork")

# Try to import numba for JIT compilation of hot paths
try:
    import numba as nb
    from numba import njit  # type: ignore[import-not-found]

    HAS_NUMBA = True
except ImportError:
    raise ImportError("numba is required for this module")

U8C = nb.types.Array(nb.types.uint8, 1, "C", readonly=False)
U8C_RO = nb.types.Array(nb.types.uint8, 1, "C", readonly=True)
INT64 = nb.types.int64
B1 = nb.types.boolean
VOID = nb.types.void

BLOCK_SIZE = 512
NUM_WORKERS = 16


class TarPatcherError(Exception):
    """Raised when a tar patch operation fails."""


@dataclass(slots=True)
class TarScanResult:
    """Result of scanning a tar archive."""

    sample_keys: Set[bytes] = field(default_factory=set)
    compatible: bool = True


@dataclass(slots=True)
class DatasetScanResult:
    """Aggregated result of scanning a set of tar archives."""

    compatible: bool
    duplicates: Dict[str, List[str]]
    scan_results: Dict[str, TarScanResult]

    @property
    def has_duplicates(self) -> bool:
        return bool(self.duplicates)


@njit(B1(U8C_RO), cache=True, fastmath=True, inline="always")
def _nb_is_zero_block(block: bytearray | bytes) -> bool:
    """Numba-optimized: Check if a block is all zeros."""
    for i in range(len(block)):
        if block[i] != 0:
            return False
    return True


@njit(INT64(U8C_RO), cache=True, inline="always")
def _nb_parse_size(size: np.ndarray) -> int:
    # Base-256 (binary) encoding (POSIX)
    # # [124:136]
    if size[0] & 0x80:
        # Numba doesn't support int.from_bytes, so implement base-256 parsing manually
        # Big endian encoding
        n = nb.int64(size[0] & 0x3F)
        for i in range(1, size.size):
            n = (n << 8) | nb.int64(size[i])
        # If the sign bit is set, compute the negative value per tar spec
        if size[0] & 0x40:
            # print("negative binary size")
            return 0
        return n

    # Parse ascii integer
    n = nb.int64(0)
    for i in range(size.size):
        byte = size[i]
        if byte == 0 or byte == 32:
            continue
        if byte < 48 or byte > 57:
            return 0
        n = (n * 8) + nb.int64(byte - 48)
    return n


@njit(nb.types.Tuple((U8C_RO, U8C_RO))(U8C_RO), cache=True, inline="always")
def split_ustar_path(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split path into (prefix, name) suitable for ustar:
    - name:  up to 100 bytes
    - prefix: up to 155 bytes
    Return (prefix_bytes, name_bytes) or ([]], []]) if it doesn't fit.
    """
    if len(path) <= 100:
        return np.empty(0, dtype=np.uint8), path

    # Try to split at a '/' so prefix <=155 bytes and name <=100 bytes.
    cut = -1
    for i in range(path.size):
        if path[i] == 47:  # ord(b"/") = 47
            if i <= 155:
                cut = i

    if cut == -1:
        return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)

    prefix_str = path[0:cut]
    name_str = path[cut + 1 :]

    prefix_b = prefix_str
    name_b = name_str

    if prefix_b.size > 155 or name_b.size > 100:
        return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)

    return prefix_b, name_b


@njit(INT64(U8C_RO), cache=True, fastmath=True, inline="always")
def _nb_compute_checksum(header: np.ndarray) -> int:
    """
    Numba-optimized: Compute tar header checksum.
    Treat chksum field (148-155) as spaces (0x20) during calculation.
    """
    total = nb.int64(0)
    for i in range(148):
        total += header[i]
    for i in range(148, 156):
        total += 32
    for i in range(156, len(header)):
        total += header[i]
    return total


@njit(VOID(INT64, U8C), cache=True, inline="always")
def _nb_format_chksum(val: int, dst: np.ndarray) -> None:
    dst[0] = (nb.uint8(val >> 15) & 0o7) | 0x30
    dst[1] = (nb.uint8(val >> 12) & 0o7) | 0x30
    dst[2] = (nb.uint8(val >> 9) & 0o7) | 0x30
    dst[3] = (nb.uint8(val >> 6) & 0o7) | 0x30
    dst[4] = (nb.uint8(val >> 3) & 0o7) | 0x30
    dst[5] = (nb.uint8(val) & 0o7) | 0x30
    dst[6] = 0
    dst[7] = 32


@njit(nb.types.Tuple((U8C_RO, B1))(U8C_RO), cache=True, inline="always")
def _nb_extract_full_path(hdr: np.ndarray) -> tuple[np.ndarray, bool]:
    for i in range(100):
        if hdr[i] == 0:
            name = hdr[0:i]
            break
    else:
        name = hdr[0:100]
    # if magic == b"ustar\0" or magic == b"ustar ":
    if (
        hdr[257] == 117
        and hdr[258] == 115
        and hdr[259] == 116
        and hdr[260] == 97
        and hdr[261] == 114
        and (hdr[262] == 0 or hdr[262] == 32)
    ):
        for i in range(345, 500):
            if hdr[i] == 0:
                prefix_field = hdr[345:i]
                break
        else:
            prefix_field = hdr[345:500]
        if prefix_field.size > 0 and name.size > 0:
            out = np.empty(prefix_field.size + 1 + name.size, dtype=np.uint8)
            out[: prefix_field.size] = prefix_field
            out[prefix_field.size] = 47
            out[prefix_field.size + 1 :] = name
            return out, True
        elif prefix_field.size > 0:
            return prefix_field, True
        else:
            return name, True
    else:
        return name, False


@njit(cache=True)
def pax_parse(data: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Parse PAX extended header into list of (key, value).
    Format lines: "%d key=value\n".
    """
    out = []
    i = nb.int64(0)
    n = len(data)
    while i < n:
        # parse length
        j = i
        while j < n and data[j] != 32:
            j += 1
        if j == n:
            break
        rec_len = nb.int64(0)
        for k in range(i, j):
            if data[k] < 48 or data[k] > 57:
                rec_len = 512
                break
            rec_len = (rec_len * 10) + (data[k] - 48)

        if i + rec_len > n:
            break

        sp = -1
        end = i + rec_len
        if data[end - 1] != 10:
            break
        end -= 1
        for k in range(i, end):
            if data[k] == 32:
                sp = k
                break
        else:
            break

        for i in range(sp + 1, end):
            # ord(b"=") = 61
            if data[i] == 61:
                key = data[sp + 1 : i]
                val = data[i + 1 : end]
                out.append((key, val))
                break

        i += rec_len
    return out


@njit(nb.types.Tuple((nb.types.boolean, INT64))(U8C, U8C_RO), cache=True)
def update_header(hdr: np.ndarray, prefix: np.ndarray) -> tuple[bool, int]:
    """Update the header with a new path, prefixing the path with prefix.

    Args:
        hdr: The header to update.
        prefix: The prefix to add to the path.

    Returns:
        True if the header was updated successfully, False otherwise.
        And the number of blocks to skip.

    """
    size_val = _nb_parse_size(hdr[124:136])

    typeflag = hdr[156]
    # ord(b"L") = 76, ord(b"K") = 75
    if typeflag == 76 or typeflag == 75:
        raise TarPatcherError("Unexpected GNU longname/longlink encountered during patch.")

    # ord(b"x") = 120, ord(b"g") = 103
    if typeflag == 120 or typeflag == 103:
        return False, size_val

    orig_path, is_ustar = _nb_extract_full_path(hdr)
    # new_path = prefix + (name_prefix + b'/' if len(name_prefix) > 0 and len(name) > 0 else b'') + name
    # ord(b"/") = 47
    new_path = np.empty(prefix.size + orig_path.size, dtype=np.uint8)
    new_path[: prefix.size] = prefix
    new_path[prefix.size :] = orig_path

    if is_ustar:
        new_prefix_b, new_name_b = split_ustar_path(new_path)
        if new_name_b.size == 0:
            raise TarPatcherError(
                "Internal error: ustar fields don't fit for " + repr(new_path) + "."
            )
        hdr[: new_name_b.size] = new_name_b
        for i in range(new_name_b.size, 100):
            hdr[i] = 0
        hdr[345 : 345 + new_prefix_b.size] = new_prefix_b
        for i in range(345 + new_prefix_b.size, 500):
            hdr[i] = 0
    else:
        new_name_b = new_path
        if new_name_b.size > 100:
            raise TarPatcherError(
                "Internal error: legacy name too long for " + repr(new_path) + "."
            )
        hdr[0 : new_name_b.size] = new_name_b
        for i in range(new_name_b.size, 100):
            hdr[i] = 0

    checksum = _nb_compute_checksum(hdr)
    _nb_format_chksum(checksum, hdr[148:156])

    return True, size_val


@njit(nb.types.boolean(U8C_RO, INT64, INT64))
def _nb_evaluate_pax_header(
    raw_data: np.ndarray,
    position: nb.int64,
    size_val: int,
) -> None:
    PAX_PATH_KEYS = (
        np.frombuffer(b"path", dtype=np.uint8),
        np.frombuffer(b"linkpath", dtype=np.uint8),
        np.frombuffer(b"gnu.path", dtype=np.uint8),
        np.frombuffer(b"gnu.linkpath", dtype=np.uint8),
        np.frombuffer(b"SCHILY.path", dtype=np.uint8),
        np.frombuffer(b"SCHILY.linkpath", dtype=np.uint8),
    )
    blocks = (size_val + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    data = raw_data[position : position + blocks]
    if data.size < blocks:
        raise TarPatcherError("Truncated PAX extended header data.")
    records = pax_parse(data[:size_val])
    for key, _ in records:
        for pax_key in PAX_PATH_KEYS:
            if key.size == pax_key.size and np.all(key == pax_key):
                return False
    return True


@njit(nb.types.Tuple((nb.types.boolean, nb.types.ListType(U8C_RO)))(U8C_RO, U8C_RO), cache=True)
def _nb_scan_file(raw_data: np.ndarray, prefix_bytes: np.ndarray) -> tuple[bool, list[np.ndarray]]:
    position = nb.int64(0)
    total = raw_data.size
    compatible = True
    sample_key_list = nb.typed.List.empty_list(U8C_RO)

    rd_buf = np.empty(65536, dtype=np.uint8)

    last_sample_key = np.empty(0, dtype=np.uint8)

    while True:
        header = raw_data[position : position + BLOCK_SIZE].copy()
        if position + BLOCK_SIZE > total:
            raise TarPatcherError("Unexpected EOF while reading header.")

        if _nb_is_zero_block(header):
            break

        size_val = _nb_parse_size(header[124:136])

        typeflag = header[156]
        # ord(b"L") = 76, ord(b"K") = 75
        if typeflag == 76 or typeflag == 75:
            # "Unexpected GNU longname/longlink encountered during patch."
            if compatible:
                print("Found GNU longname/longlink entry")
            compatible = False
            # TODO: Still parse the filename to get the sample key
            raise TarPatcherError(
                "Found GNU longname/longlink entry; in-place rename is unsupported."
            )

        # ord("x") = 120, ord("g") = 103
        if typeflag in (120, 103):
            if not _nb_evaluate_pax_header(raw_data, position, size_val):
                # "PAX header contains unsupported key; in-place rename is unsafe."
                if compatible:
                    print("PAX header contains unsupported key")
                compatible = False
        else:
            full_path, is_ustar = _nb_extract_full_path(header)

            new_path = np.empty(prefix_bytes.size + full_path.size, dtype=np.uint8)
            new_path[: prefix_bytes.size] = prefix_bytes
            new_path[prefix_bytes.size :] = full_path

            # Apply the regex for splitting the sample key from the extension.
            # Find last slash to find filename
            for i in range(full_path.size - 1, -1, -1):
                # ord('/') = 47
                if full_path[i] == 47:
                    last_path_idx = i
                    break
            else:
                last_path_idx = 0
            # Find extension (i.e. first dot after last slash)
            for i in range(last_path_idx + 1, full_path.size):
                # ord('.') = 46
                if full_path[i] == 46:
                    extension_idx = i
                    break
            else:
                extension_idx = 0
            sample_key = full_path[:extension_idx]
            if sample_key.size > 0:
                # Next group, store the sample key
                if last_sample_key.size != sample_key.size or not np.all(
                    last_sample_key == sample_key
                ):
                    sample_key_list.append(sample_key)
                    last_sample_key = sample_key

            if is_ustar:
                new_prefix_b, new_name_b = split_ustar_path(new_path)
                if new_name_b.size == 0:
                    # "Internal error: ustar fields don't fit."
                    if compatible:
                        print("Internal error: ustar fields don't fit")
                    compatible = False
            else:
                if new_path.size > 100:
                    # "New name too long for legacy header."
                    if compatible:
                        print("New name too long for legacy header")
                    compatible = False

        # Dummy read
        inc = BLOCK_SIZE + (size_val + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
        last_block_size = inc % 65536
        for i in range(position, position + inc - last_block_size, 65536):
            rd_buf[:] = raw_data[i : i + 65536]
        rd_buf[:last_block_size] = raw_data[position + inc - last_block_size : position + inc]
        position += inc

    return compatible, sample_key_list


@njit(VOID(U8C, U8C_RO), cache=True)
def _nb_process_file(raw_data: np.ndarray, prefix_bytes: np.ndarray) -> None:
    position = nb.int64(0)
    total = raw_data.size
    rd_buf = np.empty(65536, dtype=np.uint8)

    while True:
        header = raw_data[position : position + BLOCK_SIZE].copy()
        if position + BLOCK_SIZE > total:
            raise TarPatcherError("Unexpected EOF while reading header.")

        if _nb_is_zero_block(header):
            break

        was_updated, size_val = update_header(header, prefix_bytes)

        if was_updated:
            raw_data[position : position + BLOCK_SIZE] = header

        inc = BLOCK_SIZE + (size_val + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
        last_block_size = inc % 65536
        for i in range(position, position + inc - last_block_size, 65536):
            rd_buf[:] = raw_data[i : i + 65536]
        rd_buf[:last_block_size] = raw_data[position + inc - last_block_size : position + inc]
        position += inc


# Keys that, if present in PAX, mean path info is controlled by PAX,
# so in-place rename of classic headers is NOT safe.
PAX_PATH_KEYS = (
    np.frombuffer(b"path", dtype=np.uint8),
    np.frombuffer(b"linkpath", dtype=np.uint8),
    np.frombuffer(b"gnu.path", dtype=np.uint8),
    np.frombuffer(b"gnu.linkpath", dtype=np.uint8),
    np.frombuffer(b"SCHILY.path", dtype=np.uint8),
    np.frombuffer(b"SCHILY.linkpath", dtype=np.uint8),
)


split_name_re_bytes = re.compile(rb"^((?:.*/|)[^.]+)[.]([^/]*)$")


class TarPatcher:
    """Utility for scanning and renaming tar archive entries in place."""

    def __init__(self, *, show_progress: bool = True) -> None:
        self._show_progress = show_progress

    def dataset_scan(
        self, tar_files: Sequence[str], parent_path: EPath, num_workers: int = NUM_WORKERS
    ) -> DatasetScanResult:
        """Scan multiple tar files, checking compatibility for in-place renaming and for duplicate sample keys.
        Each tar_file string must be a relative or absolute path to a tar file.

        Args:
            tar_files: List of relative or absolute paths to the tar files to scan.
            parent_path: Parent path of the tar files, used if tar_files are relative paths.

        Returns:
            DatasetScanResult: Result of the scan.
        """

        scan_results: Dict[str, TarScanResult] = {}

        # Maps from sample key to list of tar files containing it
        duplicates: Dict[str, Set[str]] = {}

        compatible = True
        have_duplicates = False

        tasks: list[tuple[str, str]] = []
        for rel_tar_file in tar_files:
            tar_file_path = parent_path / rel_tar_file
            rel_file_path = tar_file_path.relative_to(parent_path)
            tar_file = str(tar_file_path)
            prefix = f"{rel_file_path}/"
            tasks.append((tar_file, prefix))

        if not tasks:
            return DatasetScanResult(
                compatible=True,
                duplicates={},
                scan_results={},
            )

        max_workers = min(len(tasks), num_workers)

        with tqdm(
            total=len(tasks),
            desc="Scanning dataset",
            unit="shards",
            disable=not self._show_progress,
        ) as dataset_pbar:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                jobs = [executor.submit(_scan_tar_worker, *task) for task in tasks]
                for future in concurrent.futures.as_completed(jobs):
                    try:
                        tar_file, result = future.result()
                    except:
                        import traceback

                        traceback.print_exc()
                        raise
                    scan_results[tar_file] = result
                    if not result.compatible:
                        compatible = False
                    for sample_key in result.sample_keys:
                        duplicates.setdefault(sample_key, set()).add(tar_file)
                        if len(duplicates[sample_key]) > 1:
                            have_duplicates = True

                    if have_duplicates and not compatible:
                        # Let's stop early if we have duplicates and the dataset is not compatible for fixing
                        break

                    dataset_pbar.update()

                for job in jobs:
                    job.cancel()

        duplicate_map = {key: sorted(paths) for key, paths in duplicates.items() if len(paths) > 1}

        return DatasetScanResult(
            compatible=compatible,
            duplicates=duplicate_map,
            scan_results=scan_results,
        )

    def dataset_apply_prefix(
        self,
        tar_files: Sequence[str],
        parent_path: EPath,
        num_workers: int = NUM_WORKERS,
    ) -> None:
        """Apply shard-specific prefixes to a set of tar files."""

        tasks: list[tuple[str, str]] = []
        for rel_tar_file in tar_files:
            tar_file_path = parent_path / rel_tar_file
            rel_file_path = tar_file_path.relative_to(parent_path)

            tar_file = str(tar_file_path)
            prefix = f"{rel_file_path}/"
            tasks.append((tar_file, prefix))

        if not tasks:
            return

        max_workers = min(len(tasks), num_workers)

        with tqdm(
            total=len(tasks),
            desc="Applying prefixes",
            unit="shards",
            disable=not self._show_progress,
        ) as dataset_pbar:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                jobs = [executor.submit(_apply_prefix_worker, *task) for task in tasks]
                for future in concurrent.futures.as_completed(jobs):
                    future.result()
                    dataset_pbar.update()

    def scan(self, tar_path: Path | str, prefix: str) -> TarScanResult:
        """Scan *tar_path* and evaluate compatibility for prefixing entries."""

        prefix_bytes = np.frombuffer(prefix.encode("utf-8"), dtype=np.uint8)

        raw_data = np.memmap(tar_path, dtype=np.uint8, mode="r+")

        compatible, sample_keys_list = _nb_scan_file(raw_data, prefix_bytes)

        # Convert numpy arrays to bytes for the set
        sample_keys_set = {sample_key.tobytes() for sample_key in sample_keys_list}
        if len(sample_keys_set) != len(sample_keys_list):
            from collections import Counter

            print(
                f"Duplicate sample keys within a single tar file {tar_path}: {len(sample_keys_list)} keys, {len(sample_keys_set)} unique keys"
            )
            print(
                f"Most common sample keys: {Counter(key.tobytes() for key in sample_keys_list).most_common(10)}"
            )
            compatible = False

        return TarScanResult(
            sample_keys=sample_keys_set,
            compatible=compatible,
        )

    def apply_prefix(
        self,
        tar_path: Path | str,
        prefix: str,
    ) -> None:
        """Apply *prefix* to entries in *tar_path* in place."""
        prefix_bytes = np.frombuffer(prefix.encode("utf-8"), dtype=np.uint8)
        raw_data = np.memmap(tar_path, dtype=np.uint8, mode="readwrite")
        _nb_process_file(raw_data, prefix_bytes)
        raw_data.flush()

    def _evaluate_pax_header(
        self,
        handle: BinaryIO,
        size_val: int,
        pbar: tqdm,
        result: TarScanResult,
    ) -> None:
        blocks = (size_val + BLOCK_SIZE - 1) // BLOCK_SIZE
        data = handle.read(blocks * BLOCK_SIZE)
        if len(data) < blocks * BLOCK_SIZE:
            raise TarPatcherError("Truncated PAX extended header data.")
        pbar.update(blocks * BLOCK_SIZE)
        records = pax_parse(data[:size_val])
        for key, _ in records:
            if any(
                key.size == pax_key.size and np.all(key == pax_key) for pax_key in PAX_PATH_KEYS
            ):
                result.compatible = False
                result.issues.append(
                    f"PAX header contains unsupported key {key!r}; in-place rename is unsafe."
                )

    def _skip_payload(self, handle: BinaryIO, size_val: int, pbar: tqdm) -> None:
        data_blocks = (size_val + BLOCK_SIZE - 1) // BLOCK_SIZE
        if data_blocks:
            handle.seek(data_blocks * BLOCK_SIZE, 1)
            pbar.update(data_blocks * BLOCK_SIZE)

    def _progress(self, total: int, desc: str, disable: bool = False) -> tqdm:
        return tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=desc,
            leave=False,
            disable=disable,
        )


def _scan_tar_worker(tar_file: str, prefix: str) -> tuple[str, TarScanResult]:
    patcher = TarPatcher(show_progress=False)
    result = patcher.scan(tar_file, prefix)
    return tar_file, result


def _apply_prefix_worker(tar_file: str, prefix: str) -> str:
    patcher = TarPatcher(show_progress=False)
    patcher.apply_prefix(tar_file, prefix)
    return tar_file


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "dataset_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Only check if in-place renaming is possible; don't modify the file.",
)
def run_dataset(dataset_path: Path, dry_run: bool):
    """
    PREFIX all member names in all tar files in DATASET_PATH in-place, if safely possible.

    Rules:
    - Allowed:
      * Classic/ustar entries.
      * PAX x/g entries that do NOT contain path-related keys.
    - Rejected:
      * Any PAX entry with path/linkpath-style keys.
      * Any GNU longname/longlink (L/K).
      * Any resulting name that doesn't fit fixed-size header fields.
    """
    import time

    patcher = TarPatcher()

    # files = ["shard-0.tar", "audio_0.tar"]
    # for file in files:
    #     print(f"Reading {file}...")
    #     start = time.time()
    #     buf = np.memmap(file, dtype=np.uint8, mode="r")
    #     for i in range(0, buf.size, 65536):
    #         buf[i:i+65536].copy()
    #     # with open(file, "rb") as f:
    #     #     f.read(65536)
    #     end = time.time()
    #     print(f"Read time: {end - start} seconds")

    files = tuple(str(p.relative_to(dataset_path)) for p in dataset_path.rglob("*.tar"))

    click.echo(f"Scanning {dataset_path} for in-place rename feasibility...")
    start = time.time()
    res = patcher.dataset_scan(files, dataset_path)
    end = time.time()
    print(f"Scan time: {end - start} seconds")
    print(
        f"compatible: {res.compatible}, {len(res.duplicates)} duplicates: {[key for _, key in zip(range(10), res.duplicates.items())]}"
    )
    if not res.compatible:
        raise click.ClickException("In-place rename is not possible under current rules.")
    if len(res.duplicates) == 0:
        raise click.ClickException("No duplicates found.")

    click.echo("OK: in-place modification is possible under current rules.")

    if dry_run:
        return

    click.echo("Applying prefix in-place...")
    try:
        start = time.time()
        patcher.dataset_apply_prefix(files, dataset_path)
        end = time.time()
        print(f"Apply time: {end - start} seconds")
    except TarPatcherError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Done. All eligible member names have been updated.")


@cli.command()
@click.argument(
    "tar_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument("prefix", type=str)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Only check if in-place renaming is possible; don't modify the file.",
)
def run_file(tar_file: Path, prefix: str, dry_run: bool):
    """
    PREFIX all member names in TAR_FILE in-place, if safely possible.

    Rules:
    - Allowed:
      * Classic/ustar entries.
      * PAX x/g entries that do NOT contain path-related keys.
    - Rejected:
      * Any PAX entry with path/linkpath-style keys.
      * Any GNU longname/longlink (L/K).
      * Any resulting name that doesn't fit fixed-size header fields.
    """
    import time

    tar_path = Path(tar_file)

    patcher = TarPatcher()

    # files = ["shard-0.tar", "audio_0.tar"]
    # for file in files:
    #     print(f"Reading {file}...")
    #     start = time.time()
    #     buf = np.memmap(file, dtype=np.uint8, mode="r")
    #     for i in range(0, buf.size, 65536):
    #         buf[i:i+65536].copy()
    #     # with open(file, "rb") as f:
    #     #     f.read(65536)
    #     end = time.time()
    #     print(f"Read time: {end - start} seconds")

    click.echo(f"Scanning {tar_path} for in-place rename feasibility...")
    orig_start_method = mp.get_start_method()
    # mp.set_start_method("fork", force=True)
    for i in range(2):
        try:
            print(f"Scanning {i}...")
            start = time.time()
            res = patcher.scan(tar_path, prefix)
            end = time.time()
            print(
                f"compatible: {res.compatible}, {len(res.sample_keys)} keys, keys: {[key for _, key in zip(range(10), res.sample_keys)]}"
            )
            print(f"Scan time: {end - start} seconds")
        except TarPatcherError as exc:
            raise click.ClickException(str(exc)) from exc

    mp.set_start_method(orig_start_method, force=True)

    click.echo("OK: in-place modification is possible under current rules.")

    if dry_run:
        return

    click.echo("Applying prefix in-place...")
    try:
        start = time.time()
        patcher.apply_prefix(tar_path, prefix)
        end = time.time()
        print(f"Apply time: {end - start} seconds")
    except TarPatcherError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Done. All eligible member names have been updated.")


if __name__ == "__main__":
    cli()
