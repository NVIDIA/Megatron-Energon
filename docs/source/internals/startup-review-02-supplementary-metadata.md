<!-- Copyright (c) 2026, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# feat(metadata): add opt-in supplementary metadata lookup

Call `megatron.energon.cache.supplementary_metadata.install(preparation_root)` in each loader owner before worker creation. The directory contains a pinned `ready-map.json` and immutable completed SQLite indexes under `databases/`. Payload stores and source identity remain unchanged; missing entries use the existing lookup.

This review branch preserves the measured process-wide adapter. Before opening a merge-ready PR, agree on the permanent store/config API and worker-spawn lifecycle. It does not generate data indexes or silently enable itself. A separate observed-key index is supported by the preserved adapter but was not used in the selected measurement. Preparation is offline and must not mutate published indexes.
