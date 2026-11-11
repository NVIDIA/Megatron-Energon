<!-- Copyright (c) 2026, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# perf(io): reuse local index readers and descriptor sizes

Local POSIX offset indexes use the size of their already-open descriptor. Metadata-provider and other stream types retain their authoritative size lookup. Cached offset readers reuse an existing reader for another offset in the same file, within the existing cache bound. No new configuration or storage-client fork is required.
