import io
import sqlite3
import time
from typing import Any, Literal, Optional, Tuple

import webdataset

from megatron.energon.av import AVWebdatasetDecoder
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME, split_name_re
from megatron.energon.flavors.webdataset.default_decoder_webdataset import ImageDecoder
from megatron.energon.flavors.webdataset.fast_read import FileRandomReadPool
from megatron.energon.flavors.webdataset.itar import ITarFile
from megatron.energon.flavors.webdataset.metadata import WebdatasetMeta

STATS_READ_TIME_NS = 0
STATS_DECODE_TIME_NS = 0


class RandomAccessDataset:
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset."""

    dataset_path: EPath
    metadata: WebdatasetMeta
    _read_pool: FileRandomReadPool

    _sqlite_connection: Optional[sqlite3.Connection] = None
    _sqlite_cursor: Optional[sqlite3.Cursor] = None

    def __init__(
        self,
        dataset_path: EPath,
        *,
        split_part: str = "train",
        info_config: str = ".info.yaml",
        split_config: str = "split.yaml",
    ):
        """
        Initialize a direct dataset reader that can read files directly from tar archives.

        Args:
            dataset_path: Path to the dataset directory containing .nv-meta
        """
        self.dataset_path = dataset_path

        # Load metadata (will be used later for direct access)
        self.metadata = WebdatasetMeta.from_config(
            path=dataset_path,
            split_part=split_part,
            info_config=info_config,
            split_config=split_config,
        )

        self._read_pool = FileRandomReadPool()

    def _get_subtar(self, fname: str) -> Tuple[ITarFile, int]:
        m = split_name_re.match(fname)
        if not m:
            raise ValueError(f"Invalid file name: {fname}")
        cur_base_name, _ = m.groups()

        if self._sqlite_connection is None:
            self._sqlite_connection = sqlite3.connect(
                str(self.dataset_path / MAIN_FOLDER_NAME / "index.sqlite")
            )
            self._sqlite_cursor = self._sqlite_connection.cursor()

        self._sqlite_cursor.execute(
            "SELECT tar_file_id, sample_index, byte_offset, byte_size FROM samples WHERE sample_key = ?",
            (cur_base_name,),
        )
        result = self._sqlite_cursor.fetchone()
        if result is None:
            raise ValueError(f"Sample key not found in index: {cur_base_name}")
        tar_file_id, sample_index, byte_offset, byte_size = result
        shard = self.metadata.shards[tar_file_id]

        raw_tar_bytes = self._read_pool.read(str(shard.path), byte_offset, byte_size)
        return ITarFile.open(fileobj=io.BytesIO(raw_tar_bytes), mode="r:"), byte_size

    def __getitem__(self, fname: str) -> bytes:
        global STATS_READ_TIME_NS
        start_time = time.perf_counter_ns()
        tar_file, byte_size = self._get_subtar(fname)

        while tar_file.offset < byte_size:
            tarinfo = tar_file.next()
            if tarinfo is None:
                raise ValueError("Unexpected end of tar file")
            if not tarinfo.isfile() or tarinfo.name is None:
                continue
            if tarinfo.name == fname:
                res = tar_file.extractfile(tarinfo).read()
                STATS_READ_TIME_NS += time.perf_counter_ns() - start_time
                return res
        raise ValueError(f"File not found in tar file: {fname}")


class RandomAccessDecoderDataset:
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset and decode them."""

    def __init__(
        self,
        dataset_path: EPath,
        *,
        split_part: str = "train",
        info_config: str = ".info.yaml",
        split_config: str = "split.yaml",
        image_decode: ImageDecoder = "torchrgb",
        av_decode: Literal["torch", "AVDecoder", "pyav"] = "AVDecoder",
        video_decode_audio: bool = False,
    ):
        self._inner_reader = RandomAccessDataset(
            dataset_path,
            split_part=split_part,
            info_config=info_config,
            split_config=split_config,
        )
        self._decoder = webdataset.autodecode.Decoder(
            [
                webdataset.autodecode.imagehandler(image_decode),
                AVWebdatasetDecoder(
                    video_decode_audio=video_decode_audio,
                    av_decode=av_decode,
                ),
            ]
        )

    def __getitem__(self, fname: str) -> Any:
        global STATS_DECODE_TIME_NS
        m = split_name_re.match(fname)
        if not m:
            raise ValueError(f"Invalid file name: {fname}")
        cur_base_name, ext = m.groups()

        start_time = time.perf_counter_ns()
        res = self._decoder(
            {
                "__key__": cur_base_name,
                ext: self._inner_reader[fname],
            }
        )[ext]
        STATS_DECODE_TIME_NS += time.perf_counter_ns() - start_time

        return res
