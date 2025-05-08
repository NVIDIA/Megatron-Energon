# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import re
from collections import defaultdict
from typing import List, Tuple

__all__ = ["collapse"]

"""Helper functions for string tokenization and expression building."""

_num_re = re.compile(r"\d+")


def _tokenize(s: str) -> Tuple[List[str], List[Tuple[str, int, int]]]:
    """
    Split the string into literal and numeric parts.
    Always starts with a literal (sometimes empty)

    Example:
        "partition_00/shard_000000.tar" ->
        lits = ["partition_", "/shard_", ".tar"]
        nums = [("00", 0, 2), ("000000", 0, 6)]

    Args:
        s: Input string to tokenize.

    Returns:
        Tuple containing:
            - lits: List of literal pieces, length = #nums + 1
            - nums: List of tuples (raw, value, width) where:
                - raw: original substring (keeps zero-padding)
                - value: int(raw)
                - width: len(raw)
    """
    lits, nums = [], []
    pos = 0
    for m in _num_re.finditer(s):
        lits.append(s[pos : m.start()])
        raw = m.group(0)
        nums.append((raw, int(raw), len(raw)))
        pos = m.end()
    lits.append(s[pos:])
    return lits, nums


def _build_expr(
    lits: List[str],
    nums: List[Tuple[str, int, int]],
    var_idx: int,
    start_raw: str,
    end_raw: str,
) -> str:
    """
    Re-assemble the template, replacing slot with brace expansion syntax.

    Args:
        lits: List of literal pieces of the string.
        nums: List of numeric parts as tuples (raw, value, width).
        var_idx: Index of the numeric slot to replace with range.
        start_raw: Starting value (raw string).
        end_raw: Ending value (raw string).

    Returns:
        String with brace expansion syntax.
    """
    parts: List[str] = []
    for i in range(len(nums)):
        parts.append(lits[i])
        if i == var_idx:
            parts.append(f"{{{start_raw}..{end_raw}}}")
        else:
            parts.append(nums[i][0])
    parts.append(lits[-1])
    return "".join(parts)


def _streaming_mode(strings: List[str]) -> List[str]:
    """
    Compress strings in order-preserving streaming mode.
    Complexity: O(N)

    Args:
        strings: List of strings to compress.

    Returns:
        List of compressed expressions.
    """
    # Result list with brace expressions
    out: List[str] = []

    # Total number of strings
    n = len(strings)

    # Current index
    i = 0

    while i < n:
        lits0, nums0 = _tokenize(strings[i])

        # Strings without numbers can never form a range
        if not nums0:
            out.append(strings[i])
            i += 1
            continue

        # Which numeric slot is changing?
        var_idx: int = -1

        start_raw: str = ""
        prev_nums = nums0

        # Last index in the current candidate range
        run_end = i

        # Starting with string `i` as the template, check subsequent strings `j` as long as they match
        j = i + 1
        while j < n:
            lits1, nums1 = _tokenize(strings[j])

            # Template must be identical (same number of literals and numeric slots)
            if lits1 != lits0 or len(nums1) != len(nums0):
                break

            # Exactly one numeric slot may differ ─ find it
            diff_slots = [k for k, (a, b) in enumerate(zip(prev_nums, nums1)) if a[1] != b[1]]
            if len(diff_slots) != 1:
                break
            k = diff_slots[0]

            # Width must stay the same
            if nums1[k][2] != prev_nums[k][2]:
                break

            # Same changing slot for the whole run
            if var_idx == -1:
                var_idx, start_raw = k, nums0[k][0]
            elif var_idx != k:
                break

            # Contiguous ascending (+1) only
            if nums1[k][1] != prev_nums[k][1] + 1:
                break

            # OK - extend run
            run_end = j
            prev_nums = nums1
            j += 1

        run_len = run_end - i + 1

        if run_len >= 2 and var_idx != -1:
            # Emit range
            end_raw = prev_nums[var_idx][0]
            out.append(_build_expr(lits0, nums0, var_idx, start_raw, end_raw))
            i = run_end + 1
        else:
            # Single string
            out.append(strings[i])
            i += 1
    return out


def _bucket_greedy_mode(strings: List[str]) -> List[str]:
    """
    Compress strings using bucket + greedy algorithm to minimize pattern count.
    Complexity: O(N log N)
    Args:
        strings: List of strings to compress.

    Returns:
        List of compressed expressions (order may change).
    """
    # Tokenize all stringsonce
    tokenized = []
    for s in strings:
        lits, nums = _tokenize(s)
        tokenized.append({"lits": lits, "nums": nums, "orig": s})

    # Build buckets
    buckets: defaultdict = defaultdict(list)
    for idx, t in enumerate(tokenized):
        lits, nums = t["lits"], t["nums"]
        for var_idx, (raw, value, width) in enumerate(nums):
            key_tokens = []
            for k in range(len(nums)):
                key_tokens.append(lits[k])
                key_tokens.append(None if k == var_idx else nums[k][0])
            key_tokens.append(lits[-1])
            key = (var_idx, tuple(key_tokens))
            buckets[key].append((idx, value, raw, width))

    # Find contiguous runs inside every bucket
    # candidate contain tuples (covered_size, indices, expression)
    candidates = []
    for (var_idx, _), entries in buckets.items():
        # Sort by numeric *value*
        entries.sort(key=lambda e: e[1])

        # Start with the first entry
        run = [entries[0]]

        def _flush():
            if len(run) >= 2:
                idxs = [e[0] for e in run]
                start_raw, end_raw = run[0][2], run[-1][2]
                t0 = tokenized[idxs[0]]
                expr = _build_expr(t0["lits"], t0["nums"], var_idx, start_raw, end_raw)
                candidates.append((len(run), idxs, expr))

        # Check subsequent entries
        for e in entries[1:]:
            prev = run[-1]
            if e[1] == prev[1] + 1 and e[3] == prev[3]:  # contiguous, same width
                run.append(e)
            else:
                _flush()
                run = [e]
        _flush()

    # Greedy cover: longest first, no overlaps
    candidates.sort(key=lambda c: (-c[0], c[2]))  # stable order
    covered = [False] * len(strings)
    out: List[str] = []

    for _, idxs, expr in candidates:
        if all(not covered[i] for i in idxs):  # keep only disjoint
            out.append(expr)
            for i in idxs:
                covered[i] = True

    # Leftover single strings
    out.extend(t["orig"] for i, t in enumerate(tokenized) if not covered[i])
    return out


def collapse(strings: List[str], keep_order: bool = False) -> List[str]:
    """
    Reverse-brace-expand a list of strings.

    Args:
        strings: The filenames / words to be compressed.
        keep_order: Whether to preserve original order.
            * False → minimise the **count** of patterns (order may change).
            * True → keep the order of the input in the expanded output.

    Returns:
        List of brace-expressions plus (possibly) single strings.
    """
    return _streaming_mode(strings) if keep_order else _bucket_greedy_mode(strings)


if __name__ == "__main__":
    """Self-test for the module."""
    import time

    ex1 = [
        "/path/to/file001.tar.gz",
        "/path/to/file003.tar.gz",
        "/path/to/file002.tar.gz",
    ]
    ex2 = ["python2", "python3.1", "python3.2", "python3.5"]
    ex3 = ["a2b3c", "a4b3c", "a3b3c", "a4b2c", "a5b2c"]

    for case in (ex1, ex2, ex3):
        print("#", case)
        print("unordered :", collapse(case))
        print("ordered   :", collapse(case, keep_order=True))
        print()

    # ex4 = [f"shard_{x:06d}" for x in range(10_000_000)]
    ex4 = [
        f"partition_{partition:02d}/shard_{x:06d}.tar"
        for partition in range(5)
        for x in range(1_000_000)
    ]
    start = time.perf_counter()
    res = collapse(ex4, keep_order=True)
    print(res)
    print(time.perf_counter() - start, "seconds")
