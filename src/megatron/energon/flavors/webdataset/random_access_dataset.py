import io
import sqlite3
import threading
import time
from concurrent.futures import Future
from queue import Queue
from typing import Any, Literal, Tuple

import webdataset

from megatron.energon.av import AVWebdatasetDecoder
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import RandomAccessDataset
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME, split_name_re
from megatron.energon.flavors.webdataset.default_decoder_webdataset import ImageDecoder
from megatron.energon.flavors.webdataset.itar import ITarFile
from megatron.energon.flavors.webdataset.metadata import WebdatasetMeta
from megatron.energon.worker import WorkerConfig

STATS_LOAD_QUEUE_SIZE = 0
STATS_DECODE_QUEUE_SIZE = 0
STATS_READ_TIME_NS = 0
STATS_DECODE_TIME_NS = 0


class RandomAccessWebdataset(RandomAccessDataset[bytes]):
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset."""

    dataset_path: EPath
    worker_config: WorkerConfig
    metadata: WebdatasetMeta
    _sqlite_index: sqlite3.Connection | None = None
    _sqlite_cursor: sqlite3.Cursor | None = None

    def __init__(
        self,
        dataset_path: EPath,
        *,
        worker_config: WorkerConfig,
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
        self.worker_config = worker_config

        # Load metadata (will be used later for direct access)
        self.metadata = WebdatasetMeta.from_config(
            path=dataset_path,
            split_part=split_part,
            info_config=info_config,
            split_config=split_config,
        )

    def _get_subtar(self, fname: str) -> Tuple[ITarFile, int]:
        m = split_name_re.match(fname)
        if not m:
            raise ValueError(f"Invalid file name: {fname}")
        cur_base_name, _ = m.groups()

        if self._sqlite_index is None:
            # Lazy initialization of the sqlite index, such that it's in the right process/thread.
            self._sqlite_index = sqlite3.connect(
                str(self.dataset_path / MAIN_FOLDER_NAME / "index.sqlite")
            )
            self._sqlite_cursor = self._sqlite_index.cursor()

        self._sqlite_cursor.execute(
            "SELECT tar_file_id, sample_index, byte_offset, byte_size FROM samples WHERE sample_key = ?",
            (cur_base_name,),
        )
        result = self._sqlite_cursor.fetchone()
        if result is None:
            raise ValueError(f"Sample key not found in index: {cur_base_name}")
        tar_file_id, sample_index, byte_offset, byte_size = result
        shard = self.metadata.shards[tar_file_id]

        with shard.path.open("rb") as f:
            f.seek(byte_offset)
            raw_tar_bytes = f.read(byte_size)
        return ITarFile.open(fileobj=io.BytesIO(raw_tar_bytes), mode="r:"), byte_size

    def __getitem__(self, fname: str) -> bytes:
        tar_file, byte_size = self._get_subtar(fname)

        while tar_file.offset < byte_size:
            tarinfo = tar_file.next()
            if tarinfo is None:
                raise ValueError("Unexpected end of tar file")
            if not tarinfo.isfile() or tarinfo.name is None:
                continue
            if tarinfo.name == fname:
                return tar_file.extractfile(tarinfo).read()
        raise ValueError(f"File not found in tar file: {fname}")


class RandomAccessDecoderWebdataset(RandomAccessDataset[Any]):
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset and decode them."""

    def __init__(
        self,
        dataset_path: EPath,
        *,
        worker_config: WorkerConfig,
        split_part: str = "train",
        info_config: str = ".info.yaml",
        split_config: str = "split.yaml",
        image_decode: ImageDecoder = "torchrgb",
        av_decode: Literal["torch", "AVDecoder", "pyav"] = "AVDecoder",
        video_decode_audio: bool = False,
    ):
        self._inner_reader = RandomAccessWebdataset(
            dataset_path,
            worker_config=worker_config,
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
        m = split_name_re.match(fname)
        if not m:
            raise ValueError(f"Invalid file name: {fname}")
        cur_base_name, ext = m.groups()

        return self._decoder(
            {
                "__key__": cur_base_name,
                ext: self._inner_reader[fname],
            }
        )[ext]


class LazyRandomAccessDecoderWebdataset(RandomAccessDataset[Future[Any]]):
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset and decode them."""

    def __init__(
        self,
        dataset_path: EPath,
        *,
        worker_config: WorkerConfig,
        split_part: str = "train",
        info_config: str = ".info.yaml",
        split_config: str = "split.yaml",
        image_decode: ImageDecoder = "torchrgb",
        av_decode: Literal["torch", "AVDecoder", "pyav"] = "AVDecoder",
        video_decode_audio: bool = False,
        prefetch_queue_size: int = 10,
    ):
        self._inner_reader = RandomAccessWebdataset(
            dataset_path,
            worker_config=worker_config,
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

        self._files_to_read = Queue(maxsize=prefetch_queue_size)
        self._files_to_decode = Queue(maxsize=prefetch_queue_size)

        self._read_thread = threading.Thread(target=self._read_worker, daemon=True)
        self._read_thread.start()

        self._decode_thread = threading.Thread(target=self._decode_worker, daemon=True)
        self._decode_thread.start()

    def _read_worker(self):
        global STATS_READ_TIME_NS
        fname: str
        cur_base_name: str
        ext: str
        future: Future[Any]
        while True:
            fname, cur_base_name, ext, future = self._files_to_read.get()
            try:
                start_time = time.perf_counter_ns()
                raw = self._inner_reader[fname]
                STATS_READ_TIME_NS += time.perf_counter_ns() - start_time
            except Exception as e:
                future.set_exception(e)
            else:
                self._files_to_decode.put((cur_base_name, ext, future, raw))

    def _decode_worker(self):
        global STATS_DECODE_TIME_NS
        cur_base_name: str
        ext: str
        future: Future[Any]
        raw: bytes
        while True:
            cur_base_name, ext, future, raw = self._files_to_decode.get()
            try:
                start_time = time.perf_counter_ns()
                decoded = self._decoder(
                    {
                        "__key__": cur_base_name,
                        ext: raw,
                    }
                )[ext]
                STATS_DECODE_TIME_NS += time.perf_counter_ns() - start_time
            except Exception as e:
                future.set_exception(e)
            else:
                future.set_result(decoded)

    def __getitem__(self, fname: str) -> Future[Any]:
        """Get an item from the dataset.

        Args:
            fname: The name of the file to get.

        Returns:
            A future that will be resolved to the item.
        """
        m = split_name_re.match(fname)
        if not m:
            raise ValueError(f"Invalid file name: {fname}")
        cur_base_name, ext = m.groups()

        future = Future()
        self._files_to_read.put((fname, cur_base_name, ext, future))

        return future
