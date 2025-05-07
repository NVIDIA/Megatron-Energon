from collections import defaultdict
from typing import List, Tuple

from numba import njit, types
from numba.typed import List as NumbaList

__all__ = ["compress"]


digit_tuple_t = types.Tuple((types.unicode_type, types.int64, types.int64))
list_unicode_t = types.ListType(types.unicode_type)
list_tuple_t = types.ListType(digit_tuple_t)


@njit(inline="always")
def str_to_int_digits(s):
    """
    Convert a unicode digit-only string to int.
    Assumes 0-9 only (no sign, no separators).
    """
    val = 0
    for ch in s:
        # ord('0') == 48
        val = val * 10 + (ord(ch) - 48)
    return val


@njit
def nb_split_digit_runs_separate(lst):
    """
    For every string in `lst` return *two* lists:
        1. non-digit substrings   (List[List[str]])
        2. digit tuples (raw, int(raw), len(raw))  (List[List[(str,int,int)]])
    The alternation is assumed to *start* with a non-digit part, so if a
    string begins with digits we add an empty string "" as the first element
    of its non-digit list.
    """
    outer_non = NumbaList.empty_list(list_unicode_t)  # uses pre‑declared type objs
    outer_dig = NumbaList.empty_list(list_tuple_t)

    # ----------------------------------------------------------------
    for s in lst:
        n = len(s)

        inner_non = NumbaList.empty_list(types.unicode_type)
        inner_dig = NumbaList.empty_list(digit_tuple_t)

        i = 0
        # Ensure the non‑digit list always has a *first* slot ----------
        if n == 0 or s[0].isdigit():
            inner_non.append("")  # placeholder if needed

        # Single forward scan -----------------------------------------
        while i < n:
            is_dig = s[i].isdigit()
            j = i + 1
            while j < n and s[j].isdigit() == is_dig:
                j += 1
            chunk = s[i:j]

            if is_dig:
                inner_dig.append((chunk, str_to_int_digits(chunk), len(chunk)))
            else:
                inner_non.append(chunk)

            i = j

        outer_non.append(inner_non)
        outer_dig.append(inner_dig)

    return outer_non, outer_dig


def split_digit_runs_separate(
    lst: List[str],
) -> Tuple[List[List[str]], List[List[Tuple[str, int, int]]]]:
    """Recursively convert a numba.typed.List → plain Python list‑of‑lists."""
    print("Starting split_digit_runs_separate", flush=True)
    outer_non, outer_dig = nb_split_digit_runs_separate(lst)
    print("Finished split_digit_runs_separate", flush=True)
    print("Converting to Python lists", flush=True)
    converted = [list(inner) for inner in outer_non], [list(inner) for inner in outer_dig]
    print("Finished converting to Python lists", flush=True)
    return converted


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
    out: List[str] = []
    n, i = len(strings), 0

    lits, nums = split_digit_runs_separate(strings)

    while i < n:
        # lits0, nums0 = _tokenize(strings[i])
        lits0, nums0 = lits[i], nums[i]

        # strings without numbers can never form a range
        if not nums0:
            out.append(strings[i])
            i += 1
            continue

        var_idx: int = -1  # which numeric slot is changing?
        start_raw: str = ""
        prev_nums = nums0
        run_end = i  # last index in the current candidate range

        j = i + 1
        while j < n:
            lits1, nums1 = lits[j], nums[j]

            # template must be identical
            if lits1 != lits0 or len(nums1) != len(nums0):
                break

            # exactly one numeric slot may differ ─ find it
            diff_slots = [k for k, (a, b) in enumerate(zip(prev_nums, nums1)) if a[1] != b[1]]
            if len(diff_slots) != 1:
                break
            k = diff_slots[0]

            # width must stay the same
            if nums1[k][2] != prev_nums[k][2]:
                break

            # same changing slot for the whole run
            if var_idx == -1:
                var_idx, start_raw = k, nums0[k][0]
            elif var_idx != k:
                break

            # contiguous ascending (+1) only
            if nums1[k][1] != prev_nums[k][1] + 1:
                break

            # OK - extend run
            run_end = j
            prev_nums = nums1
            j += 1

        run_len = run_end - i + 1
        if run_len >= 2 and var_idx != -1:  # emit range
            end_raw = prev_nums[var_idx][0]
            out.append(_build_expr(lits0, nums0, var_idx, start_raw, end_raw))
            i = run_end + 1
        else:  # single string
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
    # ---- tokenise once -----------------------------------------------------
    # tokenized = []
    # for s in strings:
    #     lits, nums = _tokenize(s)
    #     tokenized.append({"lits": lits, "nums": nums, "orig": s})
    lits_all, nums_all = split_digit_runs_separate(strings)

    # ---- build buckets -----------------------------------------------------
    buckets: defaultdict = defaultdict(list)
    for idx in range(len(strings)):
        # lits, nums = t["lits"], t["nums"]
        lits, nums = lits_all[idx], nums_all[idx]
        for var_idx, (raw, value, width) in enumerate(nums):
            key_tokens = []
            for k in range(len(nums)):
                key_tokens.append(lits[k])
                key_tokens.append(None if k == var_idx else nums[k][0])
            key_tokens.append(lits[-1])
            key = (var_idx, tuple(key_tokens))
            buckets[key].append((idx, value, raw, width))

    # ---- find contiguous runs inside every bucket --------------------------
    candidates = []  # (covered_size, indices, expression)
    for (var_idx, _), entries in buckets.items():
        entries.sort(key=lambda e: e[1])  # by numeric *value*
        run = [entries[0]]

        def _flush():
            if len(run) >= 2:
                idxs = [e[0] for e in run]
                start_raw, end_raw = run[0][2], run[-1][2]
                t0_lits, t0_nums = lits_all[idxs[0]], nums_all[idxs[0]]
                expr = _build_expr(t0_lits, t0_nums, var_idx, start_raw, end_raw)
                candidates.append((len(run), idxs, expr))

        for e in entries[1:]:
            prev = run[-1]
            if e[1] == prev[1] + 1 and e[3] == prev[3]:  # contiguous, same width
                run.append(e)
            else:
                _flush()
                run = [e]
        _flush()

    # ---- greedy cover: longest first, no overlaps --------------------------
    candidates.sort(key=lambda c: (-c[0], c[2]))  # stable order
    covered = [False] * len(strings)
    out: List[str] = []

    for _, idxs, expr in candidates:
        if all(not covered[i] for i in idxs):  # keep only disjoint
            out.append(expr)
            for i in idxs:
                covered[i] = True

    # ---- leftover single strings ------------------------------------------
    # out.extend(t["orig"] for i, t in enumerate(tokenized) if not covered[i])
    out.extend(strings[i] for i in range(len(strings)) if not covered[i])
    return out


def compress(strings: List[str], keep_order: bool = False) -> List[str]:
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

    # for case in (ex1, ex2, ex3):
    #     print("#", case)
    #     print("unordered :", compress(case))
    #     print("ordered   :", compress(case, keep_order=True))
    #     print()

    # ex4 = [f"shard_{x:06d}" for x in range(10_000_000)]
    ex4 = [
        f"partition_{partition:02d}/shard_{x:06d}.tar"
        for partition in range(2)
        for x in range(1_000)
    ]
    start = time.perf_counter()
    res = compress(ex4, keep_order=True)
    print(res)
    print(time.perf_counter() - start, "seconds")
