# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause


from megatron.energon.cache import FileStore
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.webdataset.itar_reader import SqliteITarEntryReader


class WebdatasetFileStore(SqliteITarEntryReader, FileStore[bytes]):
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset."""

    def __init__(
        self,
        dataset_path: EPath,
    ):
        super().__init__(base_path=dataset_path, key_is_full_entryname=True)

    def get_path(self) -> str:
        return str(self.base_path)
