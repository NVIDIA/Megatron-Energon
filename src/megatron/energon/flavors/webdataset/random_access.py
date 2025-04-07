from concurrent.futures import Future, ThreadPoolExecutor
import io
from queue import Queue
import sqlite3
import threading
import time
from typing import Any, Literal, Tuple
import uuid

import webdataset

from megatron.energon.av import AVWebdatasetDecoder
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME, split_name_re
from megatron.energon.flavors.webdataset.default_decoder_webdataset import ImageDecoder
from megatron.energon.flavors.webdataset.fast_read import FileRandomReadPool
from megatron.energon.flavors.webdataset.itar import ITarFile
from megatron.energon.flavors.webdataset.metadata import WebdatasetMeta


STATS_LOAD_QUEUE_SIZE = 0
STATS_DECODE_QUEUE_SIZE = 0
STATS_READ_TIME_NS = 0
STATS_DECODE_TIME_NS = 0


class ThreadLocalStorage:
    """A class that allows to store data in a thread-local storage."""

    def __init__(self, name: str | None = None):
        if name is None:
            # Random generate
            name = str(uuid.uuid4())
        self.name = name

    def __getattribute__(self, name: str) -> Any:
        if name == "name":
            return object.__getattribute__(self, name)

        storage = getattr(threading.local(), self.name, None)
        if storage is None:
            storage = {}
            print(f"Setting {id(threading.local())}.{self.name} = {id(storage)}")
            setattr(threading.local(), self.name, storage)
        
        if name in storage:
            print(f"Getting {id(threading.local())}.{self.name}.{name} = {id(storage[name])}")
            return storage[name]
        default_value = object.__getattribute__(self, name)
        storage[name] = default_value
        print(f"Getting/Setting {id(threading.local())}.{self.name}.{name} = {id(default_value)}")
        return default_value

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "name":
            object.__setattr__(self, name, value)
            return

        storage = getattr(threading.local(), self.name, None)
        if storage is None:
            storage = {}
            print(f"Setting {id(threading.local())}.{self.name} = {id(storage)}")
            setattr(threading.local(), self.name, storage)
        storage[name] = value
        print(f"Setting {id(threading.local())}.{self.name}.{name} = {id(value)}")


class SqliteThreadLocal(ThreadLocalStorage):
    """A class that allows to store data in a thread-local storage."""
    connection: sqlite3.Connection | None = None
    cursor: sqlite3.Cursor | None = None


class RandomAccessDataset:
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset."""

    dataset_path: EPath
    metadata: WebdatasetMeta
    _read_pool: FileRandomReadPool

    _sqlite: SqliteThreadLocal

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

        self._sqlite = SqliteThreadLocal()

    def _get_subtar(self, fname: str) -> Tuple[ITarFile, int]:
        m = split_name_re.match(fname)
        if not m:
            raise ValueError(f"Invalid file name: {fname}")
        cur_base_name, _ = m.groups()

        if self._sqlite.connection is None:
            self._sqlite.connection = sqlite3.connect(str(self.dataset_path / MAIN_FOLDER_NAME / "index.sqlite"))
            self._sqlite.cursor = self._sqlite.connection.cursor()

        self._sqlite.cursor.execute(
            "SELECT tar_file_id, sample_index, byte_offset, byte_size FROM samples WHERE sample_key = ?",
            (cur_base_name,),
        )
        result = self._sqlite.cursor.fetchone()
        if result is None:
            raise ValueError(f"Sample key not found in index: {cur_base_name}")
        tar_file_id, sample_index, byte_offset, byte_size = result
        shard = self.metadata.shards[tar_file_id]

        raw_tar_bytes = self._read_pool.read(str(shard.path), byte_offset, byte_size)
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


class PipelinedRandomAccessDecoderDataset:
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
        prefetch_queue_size: int = 10,
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

        self._read_executor = ThreadPoolExecutor(max_workers=5)
        self._decode_executor = ThreadPoolExecutor(max_workers=5)

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
                decoded = self._decoder({
                    "__key__": cur_base_name,
                    ext: raw,
                })[ext]
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

        # future = Future()
        # self._files_to_read.put((fname, cur_base_name, ext, future))
        # return future

        def read():
            global STATS_READ_TIME_NS
            start_time = time.perf_counter_ns()
            raw = self._inner_reader[fname]
            STATS_READ_TIME_NS += time.perf_counter_ns() - start_time
            return raw

        read_future = self._read_executor.submit(read)

        def decode(raw: bytes):
            global STATS_DECODE_TIME_NS
            start_time = time.perf_counter_ns()
            decoded = self._decoder({
                "__key__": cur_base_name,
                ext: raw,
            })[ext]
            STATS_DECODE_TIME_NS += time.perf_counter_ns() - start_time
            return decoded
        return self._decode_executor.submit(decode, read_future.result())
