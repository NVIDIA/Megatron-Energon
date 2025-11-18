# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import click
from tqdm import tqdm

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.config import split_name_re

BLOCK_SIZE = 512
NUM_WORKERS = 16


class TarPatcherError(Exception):
    """Raised when a tar patch operation fails."""


@dataclass(slots=True)
class TarScanResult:
    """Result of scanning a tar archive."""

    filenames: list[str] = field(default_factory=list)
    sample_keys: Set[str] = field(default_factory=set)
    compatible: bool = True
    issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DatasetScanResult:
    """Aggregated result of scanning a set of tar archives."""

    compatible: bool
    duplicates: Dict[str, List[str]]
    issues: List[str]
    scan_results: Dict[str, TarScanResult]

    @property
    def has_duplicates(self) -> bool:
        return bool(self.duplicates)


def is_zero_block(block: bytes) -> bool:
    return all(b == 0 for b in block)


def parse_size(size_bytes: bytes) -> int:
    """
    Parse the tar size field (supports standard octal and base-256).
    """
    if not size_bytes:
        return 0

    # Base-256 (binary) encoding (POSIX)
    if size_bytes[0] & 0x80:
        return int.from_bytes(size_bytes, byteorder="big", signed=True)

    # Octal ascii
    s = size_bytes.rstrip(b"\0 ").decode("ascii", "replace").strip() or "0"
    try:
        return int(s, 8)
    except ValueError as e:
        raise ValueError(f"Invalid size field {size_bytes!r}: {e}") from e


def split_ustar_path(path: str):
    """
    Split path into (prefix, name) suitable for ustar:
    - name:  up to 100 bytes
    - prefix: up to 155 bytes
    - both UTF-8 encoded
    Return (prefix_bytes, name_bytes) or (None, None) if it doesn't fit.
    """
    encoded = path.encode("utf-8")

    if len(encoded) <= 100:
        return b"", encoded

    # Try to split at a '/' so prefix <=155 bytes and name <=100 bytes.
    cut = -1
    for i, ch in enumerate(path):
        if ch == "/":
            if len(path[:i].encode("utf-8")) <= 155:
                cut = i

    if cut == -1:
        return None, None

    prefix_str = path[:cut]
    name_str = path[cut + 1 :]

    prefix_b = prefix_str.encode("utf-8")
    name_b = name_str.encode("utf-8")

    if len(prefix_b) > 155 or len(name_b) > 100:
        return None, None

    return prefix_b, name_b


def compute_checksum(header: bytearray) -> int:
    """
    Compute tar header checksum:
    - Treat chksum field (148-155) as spaces (0x20) during calculation.
    """
    tmp = bytearray(header)
    for i in range(148, 156):
        tmp[i] = 32  # ' '
    return sum(tmp)


def format_chksum(val: int) -> bytes:
    """
    Format checksum as 6 octal digits, NUL, space (total 8 bytes).
    """
    return f"{val:06o}\0 ".encode("ascii")


def extract_full_path(hdr: bytes) -> tuple[str, bool]:
    """
    Extract the logical path from a header.
    Returns (full_path, is_ustar_style).
    """
    name = hdr[0:100].rstrip(b"\0").decode("utf-8", "replace")
    magic = hdr[257:263]
    if magic in (b"ustar\0", b"ustar  "):
        prefix_field = hdr[345:500].rstrip(b"\0").decode("utf-8", "replace")
        if prefix_field:
            return f"{prefix_field}/{name}" if name else prefix_field, True
        else:
            return name, True
    else:
        return name, False


def pax_parse(data: bytes):
    """
    Parse PAX extended header into list of (key, value).
    Format lines: "%d key=value\n".
    """
    out = []
    i = 0
    n = len(data)
    while i < n:
        # parse length
        j = i
        while j < n and data[j : j + 1] != b" ":
            j += 1
        if j == n:
            break
        try:
            rec_len = int(data[i:j].decode("ascii", "replace"))
        except ValueError:
            break

        if i + rec_len > n:
            break

        rec = data[i : i + rec_len]
        sp = rec.find(b" ")
        if sp == -1 or not rec.endswith(b"\n"):
            break

        kv = rec[sp + 1 : -1]  # drop trailing '\n'
        eq = kv.find(b"=")
        if eq == -1:
            key = kv.decode("utf-8", "replace")
            val = ""
        else:
            key = kv[:eq].decode("utf-8", "replace")
            val = kv[eq + 1 :].decode("utf-8", "replace")
        out.append((key, val))

        i += rec_len
    return out


# Keys that, if present in PAX, mean path info is controlled by PAX,
# so in-place rename of classic headers is NOT safe.
PAX_PATH_KEYS = {
    "path",
    "linkpath",
    "gnu.path",
    "gnu.linkpath",
    "SCHILY.path",
    "SCHILY.linkpath",
}


class TarPatcher:
    """Utility for scanning and renaming tar archive entries in place."""

    def __init__(self, *, show_progress: bool = True) -> None:
        self._show_progress = show_progress

    def dataset_scan(self, tar_files: Sequence[str], parent_path: EPath) -> DatasetScanResult:
        """Scan multiple tar files, checking compatibility for in-place renaming and for duplicate sample keys.
        Each tar_file string must be a relative or absolute path to a tar file.

        Args:
            tar_files: List of relative or absolute paths to the tar files to scan.
            parent_path: Parent path of the tar files, used if tar_files are relative paths.

        Returns:
            DatasetScanResult: Result of the scan.
        """

        issues: List[str] = []
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
                issues=[],
                scan_results={},
            )

        max_workers = min(len(tasks), NUM_WORKERS)

        with tqdm(
            total=len(tasks),
            desc="Scanning dataset",
            unit="shards",
            disable=not self._show_progress,
        ) as dataset_pbar:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for tar_file, result in executor.map(
                    _scan_tar_worker, tasks, chunksize=(len(tasks) // max_workers) or 1
                ):
                    scan_results[tar_file] = result
                    if not result.compatible:
                        compatible = False
                        issues.extend(f"{tar_file}: {issue}" for issue in result.issues)
                    for sample_key in result.sample_keys:
                        duplicates.setdefault(sample_key, set()).add(tar_file)
                        if len(duplicates[sample_key]) > 1:
                            have_duplicates = True

                    if have_duplicates and not compatible:
                        # Let's stop early if we have duplicates and the dataset is not compatible for fixing
                        break

                    dataset_pbar.update()

        duplicate_map = {key: sorted(paths) for key, paths in duplicates.items() if len(paths) > 1}

        return DatasetScanResult(
            compatible=compatible,
            duplicates=duplicate_map,
            issues=issues,
            scan_results=scan_results,
        )

    def dataset_apply_prefix(
        self,
        tar_files: Sequence[str],
        parent_path: EPath,
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

        max_workers = min(len(tasks), NUM_WORKERS)

        with tqdm(
            total=len(tasks),
            desc="Applying prefixes",
            unit="shards",
            disable=not self._show_progress,
        ) as dataset_pbar:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for _ in executor.map(
                    _apply_prefix_worker, tasks, chunksize=(len(tasks) // max_workers) or 1
                ):
                    dataset_pbar.update()

    def scan(self, tar_path: Path | str, prefix: str, progress: bool = True) -> TarScanResult:
        """Scan *tar_path* and evaluate compatibility for prefixing entries."""

        path = Path(tar_path)
        size = path.stat().st_size
        result = TarScanResult()

        progress_context = self._progress(size, "Scanning", disable=not progress)

        with path.open("rb") as handle, progress_context as pbar:
            while True:
                header = handle.read(BLOCK_SIZE)
                if not header:
                    raise TarPatcherError("Unexpected EOF while reading header.")
                if len(header) < BLOCK_SIZE:
                    raise TarPatcherError("Truncated tar header.")

                pbar.update(BLOCK_SIZE)

                if is_zero_block(header):
                    break

                typeflag = header[156:157]
                size_val = parse_size(header[124:136])

                if typeflag in (b"L", b"K"):
                    result.compatible = False
                    result.issues.append(
                        "Found GNU longname/longlink entry; in-place rename is unsupported."
                    )
                    self._skip_payload(handle, size_val, pbar)
                    continue

                if typeflag in (b"x", b"g"):
                    self._evaluate_pax_header(handle, size_val, pbar, result)
                    continue

                full_path, is_ustar = extract_full_path(header)
                result.filenames.append(full_path)
                name_match = split_name_re.match(full_path)
                if name_match is not None:
                    result.sample_keys.add(name_match.group(1))

                new_path = prefix + full_path
                if is_ustar:
                    new_prefix_b, new_name_b = split_ustar_path(new_path)
                    if new_prefix_b is None:
                        result.compatible = False
                        result.issues.append(f"New name too long for ustar fields: {new_path!r}.")
                else:
                    if len(new_path.encode("utf-8")) > 100:
                        result.compatible = False
                        result.issues.append(f"New name too long for legacy header: {new_path!r}.")

                self._skip_payload(handle, size_val, pbar)

        return result

    def apply_prefix(
        self,
        tar_path: Path | str,
        prefix: str,
        progress: bool = True,
    ) -> None:
        """Apply *prefix* to entries in *tar_path* in place."""

        path = Path(tar_path)
        size = path.stat().st_size

        progress_context = self._progress(size, "Patching", disable=not progress)

        with path.open("r+b") as handle, progress_context as pbar:
            while True:
                position = handle.tell()
                header = handle.read(BLOCK_SIZE)
                if not header:
                    raise TarPatcherError("Unexpected EOF while reading header.")
                if len(header) < BLOCK_SIZE:
                    raise TarPatcherError("Truncated tar header encountered during patch.")

                pbar.update(BLOCK_SIZE)

                if is_zero_block(header):
                    break

                hdr = bytearray(header)
                typeflag = hdr[156:157]
                size_val = parse_size(hdr[124:136])

                if typeflag in (b"L", b"K"):
                    raise TarPatcherError(
                        "Unexpected GNU longname/longlink encountered during patch."
                    )

                if typeflag in (b"x", b"g"):
                    self._skip_payload(handle, size_val, pbar)
                    continue

                full_path, is_ustar = extract_full_path(hdr)
                new_path = prefix + full_path

                if is_ustar:
                    new_prefix_b, new_name_b = split_ustar_path(new_path)
                    if new_prefix_b is None:
                        raise TarPatcherError(
                            f"Internal error: ustar fields don't fit for {new_path!r}."
                        )
                    hdr[0:100] = new_name_b.ljust(100, b"\0")
                    hdr[345:500] = new_prefix_b.ljust(155, b"\0")
                else:
                    new_name_b = new_path.encode("utf-8")
                    if len(new_name_b) > 100:
                        raise TarPatcherError(
                            f"Internal error: legacy name too long for {new_path!r}."
                        )
                    hdr[0:100] = new_name_b.ljust(100, b"\0")

                checksum = compute_checksum(hdr)
                hdr[148:156] = format_chksum(checksum)

                handle.seek(position)
                handle.write(hdr)

                self._skip_payload(handle, size_val, pbar)

    def _evaluate_pax_header(
        self,
        handle: Any,
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
            if key in PAX_PATH_KEYS:
                result.compatible = False
                result.issues.append(
                    f"PAX header contains unsupported key {key!r}; in-place rename is unsafe."
                )

    def _skip_payload(self, handle: Any, size_val: int, pbar: tqdm) -> None:
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


def _scan_tar_worker(task: tuple[str, str]) -> tuple[str, TarScanResult]:
    tar_file, prefix = task
    patcher = TarPatcher(show_progress=False)
    result = patcher.scan(tar_file, prefix, progress=False)
    return tar_file, result


def _apply_prefix_worker(task: tuple[str, str]) -> str:
    tar_file, prefix = task
    patcher = TarPatcher(show_progress=False)
    patcher.apply_prefix(tar_file, prefix, progress=False)
    return tar_file


@click.command()
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
def main(tar_file: Path, prefix: str, dry_run: bool):
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
    tar_path = Path(tar_file)
    patcher = TarPatcher()

    click.echo(f"Scanning {tar_path} for in-place rename feasibility...")
    try:
        scan_result = patcher.scan(tar_path, prefix)
    except TarPatcherError as exc:
        raise click.ClickException(str(exc)) from exc

    if not scan_result.compatible:
        issues = "; ".join(scan_result.issues) or "Tar archive not compatible with in-place rename."
        raise click.ClickException(f"In-place rename not supported: {issues}")

    click.echo("OK: in-place modification is possible under current rules.")

    if dry_run:
        return

    click.echo("Applying prefix in-place...")
    try:
        patcher.apply_prefix(tar_path, prefix, scan_result=scan_result)
    except TarPatcherError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Done. All eligible member names have been updated.")


if __name__ == "__main__":
    main()
