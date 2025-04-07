import threading
import time
from concurrent.futures import Future
from queue import Queue
from typing import Any, Literal

import webdataset

from megatron.energon.av import AVWebdatasetDecoder
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import RandomAccessDataset
from megatron.energon.flavors.webdataset.config import split_name_re
from megatron.energon.flavors.webdataset.default_decoder_webdataset import ImageDecoder
from megatron.energon.flavors.webdataset.itar_reader import SqliteITarEntryReader

STATS_LOAD_QUEUE_SIZE = 0
STATS_DECODE_QUEUE_SIZE = 0
STATS_READ_TIME_NS = 0
STATS_DECODE_TIME_NS = 0


class RandomAccessWebdataset(SqliteITarEntryReader, RandomAccessDataset[bytes]):
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset."""

    def __init__(
        self,
        dataset_path: EPath,
    ):
        super().__init__(base_path=dataset_path, key_is_full_entryname=True)


class RandomAccessDecoderWebdataset(RandomAccessDataset[Any]):
    """This dataset will directly read files from the dataset tar files from a prepared energon dataset and decode them."""

    def __init__(
        self,
        dataset_path: EPath,
        *,
        image_decode: ImageDecoder = "torchrgb",
        av_decode: Literal["torch", "AVDecoder", "pyav"] = "AVDecoder",
        video_decode_audio: bool = False,
    ):
        self._inner_reader = RandomAccessWebdataset(
            dataset_path,
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
        image_decode: ImageDecoder = "torchrgb",
        av_decode: Literal["torch", "AVDecoder", "pyav"] = "AVDecoder",
        video_decode_audio: bool = False,
        prefetch_queue_size: int = 10,
    ):
        self._inner_reader = RandomAccessWebdataset(
            dataset_path,
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
