# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the reader protocols in megatron.energon.flavors.common.reader."""

import inspect
import unittest

from megatron.energon.flavors.common import reader as reader_protocols
from megatron.energon.flavors.common.reader import PartFileReader, SamplePartReader
from megatron.energon.flavors.webdataset.itar_reader import ShardInfosITarReader

PROTOCOLS = (
    reader_protocols.IndexedSampleReader,
    reader_protocols.SamplePartReader,
    reader_protocols.PartFileReader,
    reader_protocols.SamplePartFileReader,
)


class _PartialReader(SamplePartReader):
    """Explicit protocol subclass that only implements the indexed access methods."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        return {"index": index}

    def close(self) -> None:
        pass


class TestReaderProtocols(unittest.TestCase):
    def test_missing_methods_raise(self):
        reader = _PartialReader()
        assert len(reader) == 1
        assert isinstance(reader, SamplePartReader)
        for call in (
            reader.list_all_samples,
            reader.list_all_sample_parts,
            lambda: reader.list_sample_parts("0"),
            reader.get_total_size,
        ):
            with self.assertRaises(NotImplementedError):
                call()

    def test_protocol_methods_raise(self):
        # Readers subclass the protocols explicitly, so any protocol method that a reader does not
        # implement resolves to the protocol's own method, which must fail loudly.
        checked = 0
        for protocol in PROTOCOLS:
            for name, protocol_method in vars(protocol).items():
                if not callable(protocol_method) or (
                    name.startswith("_") and name not in ("__len__", "__getitem__")
                ):
                    continue
                # The protocol methods ignore their arguments, pass dummies
                n_args = len(inspect.signature(protocol_method).parameters)
                with self.subTest(protocol=protocol.__name__, method=name):
                    with self.assertRaises(NotImplementedError):
                        protocol_method(*[None] * n_args)
                checked += 1
        assert checked > 0, "No protocol methods found, test is vacuous"

    def test_reader_inherits_protocol_methods(self):
        # ShardInfosITarReader does not implement the part listing, it resolves to the protocol.
        # Pick another inherited method here if ShardInfosITarReader implements it in the future.
        assert ShardInfosITarReader.list_all_samples is SamplePartReader.list_all_samples
        with self.assertRaises(NotImplementedError):
            ShardInfosITarReader.list_all_samples(None)

    def test_structural_isinstance(self):
        class _Structural:
            def __getitem__(self, key: str):
                return b"", None

            def get_path(self) -> str:
                return "/"

            def close(self) -> None:
                pass

        class _MissingGetPath:
            def __getitem__(self, key: str):
                return b"", None

            def close(self) -> None:
                pass

        assert isinstance(_Structural(), PartFileReader)
        assert not isinstance(_MissingGetPath(), PartFileReader)


if __name__ == "__main__":
    unittest.main()
