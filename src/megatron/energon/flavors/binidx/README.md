<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->
# Bin-Idx Dataset Support for Megatron-Energon

Adds Megatron-LM's pre-tokenized binary format (`.bin` + `.idx`) as a first-class Energon dataset type, loadable through metadataset YAMLs alongside JSONL and webdataset entries.

## Usage

In a metadataset YAML, reference by `.bin` path (the `.idx` is derived automatically):

```yaml
- path: /data/my_tokens.bin
  subflavors:
    cook: bin_idx
```

No `energon prepare` step needed — bin-idx files already have their own index.

## Architecture

Mirrors the JSONL factory pattern exactly:

| Component | JSONL analog | Bin-Idx |
|---|---|---|
| Reader | `IJsonlReader` | `IBinIdxReader` (`binidx_reader.py`) |
| Factory | `CrudeJsonlDatasetFactory` | `BinIdxDatasetFactory` (`binidx_dataset.py`) |
| Default factory | `DefaultCrudeJsonlDatasetFactory` | `DefaultBinIdxDatasetFactory` |
| File store | `JsonlFileStore` | `BinIdxFileStore` (in `cache/file_store.py`) |
| Detection | `.jsonl` → `JSONL` | `.bin` → `BINIDX` (in `metadata.py`) |
| Dispatch | `dataset_config.py` JSONL branch | `dataset_config.py` BINIDX branch |

## .idx format (from `megatron/core/datasets/indexed_dataset.py`)

```
Header:     b"MMIDIDX\x00\x00"  (9 bytes)
Version:    uint64               (must be 1)
DType:      uint8 code           (1=u8, 2=i8, 3=i16, 4=i32, 5=i64, 6=f64, 7=f32, 8=u16)
SeqCount:   uint64
DocCount:   uint64
Data:       int32[SeqCount]      sequence_lengths
            int64[SeqCount]      sequence_pointers (byte offsets into .bin)
            int64[DocCount]      document_indices
```

The reader mmaps both files. `__getitem__(idx)` reads `sequence_lengths[idx]` tokens at `sequence_pointers[idx]` byte offset from the `.bin` mmap.

## Files changed

**New:**
- `megatron/energon/flavors/binidx/__init__.py`
- `megatron/energon/flavors/binidx/binidx_reader.py`
- `megatron/energon/flavors/binidx/binidx_dataset.py`

**Modified:**
- `megatron/energon/cache/file_store.py` — added `BinIdxFileStore`
- `megatron/energon/flavors/webdataset/metadata.py` — added `BINIDX` enum + `.bin` detection
- `megatron/energon/dataset_config.py` — added BINIDX dispatch
- `megatron/energon/flavors/__init__.py` — exports
- `megatron/energon/__init__.py` — exports
