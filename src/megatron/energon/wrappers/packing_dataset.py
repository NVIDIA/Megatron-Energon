# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

from megatron.energon.errors import SYSTEM_EXCEPTIONS, FatalSampleError
from megatron.energon.flavors.base_dataset import (
    SavableDataset,
    add_sample_restore_key,
    set_sample_restore_key,
)
from megatron.energon.source_info import SourceInfo
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers._log_exception import log_exception
from megatron.energon.wrappers.base import BaseWrapperDataset, SampleIndex, get_sample_restore_key
from megatron.energon.wrappers.buffer import SavableSampleBuffer
from megatron.energon.wrappers.skip import SkipSample

T_sample = TypeVar("T_sample")
T_encoded_sample = TypeVar("T_encoded_sample")
T_batch_sample = TypeVar("T_batch_sample")


class PackingDataset(
    BaseWrapperDataset[T_sample, T_encoded_sample, T_batch_sample],
    Generic[T_sample, T_encoded_sample, T_batch_sample],
):
    """This dataset wrapper transforms samples of a dataset into chunks/packs of samples, which are
    then combined into a batch."""

    buffer_size: int
    pre_packer: Callable[[List[T_sample]], List[List[T_sample]]]
    sample_encoder: Optional[Callable[[T_sample], T_encoded_sample]]
    sample_encoder_stateless: bool
    final_packer: Callable[[List[T_encoded_sample]], T_batch_sample]
    final_packer_stateless: bool
    packer_config: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]]
    error_handler: Callable[[Exception, List[T_sample], list[SourceInfo]], None]

    #: The buffer for collecting the samples that shall be packed.
    _reading_buffer: SavableSampleBuffer

    #: Contains the pre-selected samples to be packed.
    #: The full buffer will be passed to the pre_packer.
    _pre_packing_buffer: SavableSampleBuffer

    #: Lengths of the selected groups of samples to be packed together.
    #: The samples are stored sequentially in the pre_packing_buffer because
    #: SavableSampleBuffer doesn't support nesting. But to keep the groups
    #: separate, we need to store the lengths of the groups here.
    _pre_packing_lengths: List[int]

    #: Sample index for the pre_packer
    _pre_packing_sample_index: SampleIndex

    #: Sample index for the sample_encoder
    _sample_encoder_sample_index: SampleIndex

    #: Sample index for the final_packer
    _final_packing_sample_index: SampleIndex

    _savable_fields = (
        "_reading_buffer",
        "_pre_packing_buffer",
        "_pre_packing_lengths",
        "_pre_packing_sample_index",
        "_sample_encoder_sample_index",
        "_final_packing_sample_index",
    )

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        buffer_size: int,
        pre_packer: Callable[[List[T_sample]], List[List[T_sample]]],
        final_packer: Callable[[List[T_encoded_sample]], T_batch_sample],
        *,
        final_packer_stateless: bool = False,
        sample_encoder: Optional[Callable[[T_sample], T_encoded_sample]] = None,
        sample_encoder_stateless: bool = False,
        packer_config: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]] = None,
        error_handler: Callable[
            [Exception, List[T_sample], list[SourceInfo]], None
        ] = log_exception,
        worker_config: WorkerConfig,
    ):
        """Construct a PackingDataset which is used for sequence packing.
        Using a pre_packer and final_packer, it buffers the incoming samples, groups
        them together based on the logic provided by the pre_packer, and then (using
        the final_packer) combines each group into a packed single sample also called
        a "pack" or a "packed sequence".

        Args:
            dataset: The input dataset to wrap
            buffer_size: The desired size of the input buffer for pre packing. Last buffer of a dataset may be smaller.
            pre_packer: Function which selects samples from the buffer to be packed together.
                May raise :exc:`megatron.energon.SkipSample` to skip a buffer.
            final_packer: Function which combines the selected samples into a single sample.
            final_packer_stateless: If True, the final_packer is stateless, thus samples can be
                stored/restored.
            sample_encoder: Function which encodes the samples.
            sample_encoder_stateless: If True, the sample_encoder is stateless, thus samples can be
                stored/restored.
            packer_config: Configuration for the (pre|final)_packer functions. If callable, it should return the
                configuration. Defaults to None.
            error_handler: Function which handles exceptions raised by the batcher. The default
                implementation logs the exception.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset, worker_config=worker_config)

        assert buffer_size > 0, "Packing buffer size must be greater than 0."

        self.buffer_size = buffer_size
        self.pre_packer = pre_packer
        self.final_packer = final_packer
        self.final_packer_stateless = final_packer_stateless
        self.sample_encoder = sample_encoder
        self.sample_encoder_stateless = True if sample_encoder is None else sample_encoder_stateless
        self.packer_config = packer_config
        self.error_handler = error_handler

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._reading_buffer = SavableSampleBuffer(self.dataset, worker_config=self.worker_config)
        self._pre_packing_buffer = SavableSampleBuffer(
            self.dataset, worker_config=self.worker_config
        )
        self._pre_packing_lengths = []
        self._pre_packing_sample_index = SampleIndex(self.worker_config, src=self)
        self._final_packing_sample_index = SampleIndex(self.worker_config, src=self)
        self._sample_encoder_sample_index = SampleIndex(self.worker_config, src=self)

    def __len__(self):
        """The real length is unknown, since it depends on the packing function.
        We approximate it by the length of the source dataset."""

        return len(self.dataset)

    def _fill_reading_buffer(self, source_iter: Iterator, log_progress: bool = False) -> bool:
        """
        Fill the reading buffer with samples from the dataset source iterator.

        Args:
            source_iter: Iterator of samples from the dataset.
            log_progress: If True, log the progress of the filling.

        Returns:
            True if samples are successfully read into the buffer, False if no more data.
        """

        if log_progress:
            import tqdm

            pbar_ctx = pbar = tqdm.tqdm(total=self.buffer_size, desc="Filling reading buffer")
        else:
            pbar_ctx = contextlib.nullcontext()
            pbar = None

        with pbar_ctx:
            while len(self._reading_buffer) + len(self._pre_packing_buffer) < self.buffer_size:
                try:
                    sample = next(source_iter)
                    self._reading_buffer.append(sample)
                    if pbar is not None:
                        pbar.update(1)
                except StopIteration:
                    return False
        return True

    def __iter__(self) -> Iterator[T_batch_sample]:
        trace_span = self.worker_config.worker_trace_span()
        if self.sample_encoder is not None:
            encode_name = self._function_config(self.sample_encoder)
        pre_packer_name = self._function_config(self.pre_packer)
        final_packer_name = self._function_config(self.final_packer)

        def encode_pack_samples(pack: List[T_sample]) -> List[T_encoded_sample]:
            # Apply the sample encoder to the pack
            if self.sample_encoder is None:
                return pack
            encoded_pack = []
            with trace_span.span(
                "PackingDataset._encode_pack_samples", args={"len": len(pack)}, level=2
            ):
                for sample in pack:
                    try:
                        with (
                            self._sample_encoder_sample_index.ctx() as encode_idx,
                            trace_span.span(encode_name, args={"sample_idx": encode_idx}, level=2),
                        ):
                            encoded_sample = self.sample_encoder(sample)
                        assert not isinstance(encoded_sample, Generator), "Generator not supported"
                        encoded_pack.append(
                            add_sample_restore_key(
                                encoded_sample,
                                encode_idx,
                                src=self,
                            )
                        )
                    except SkipSample:
                        trace_span.instant("PackingDataset._encode_pack_samples.skip", level=2)
                    except SYSTEM_EXCEPTIONS:
                        raise FatalSampleError.from_sample(pack)
                    except Exception as e:
                        self.error_handler(e, [sample])
                        trace_span.instant(
                            "PackingDataset._encode_pack_samples.error/skip",
                            args={"exception": f"{type(e).__name__}: {str(e)}"},
                            level=2,
                        )
            return encoded_pack

        def next_pre_pack():
            """Take the samples from the reading buffer and select groups of samples to be packed
            together."""

            assert len(self._pre_packing_buffer) == 0
            if len(self._reading_buffer) > 0:
                # Take all samples from the reading buffer and pre_pack them
                samples = list(self._reading_buffer)
                # Clear buffer and pre_packing_lengths
                self._reading_buffer.clear()
                self._pre_packing_lengths.clear()
                # Now pre pack the samples
                try:
                    with (
                        self._pre_packing_sample_index.ctx() as pre_pack_idx,
                        trace_span.span(
                            pre_packer_name,
                            args={"pre_pack_idx": pre_pack_idx, "len": len(samples)},
                            level=2,
                        ),
                    ):
                        pre_packs = self.pre_packer(samples)
                except SkipSample:
                    pre_packs = []
                    trace_span.instant("PackingDataset.next_pre_pack.skip", level=2)
                except SYSTEM_EXCEPTIONS:
                    raise FatalSampleError.from_sample(samples)
                except Exception as e:
                    self.error_handler(e, samples)
                    trace_span.instant(
                        "PackingDataset.next_pre_pack.error/skip",
                        args={"exception": f"{type(e).__name__}: {str(e)}"},
                        level=2,
                    )
                    pre_packs = []

                # Put the pre-packed samples into the pre_packing_buffer
                # They will be flattened here to avoid nested buffers
                # But the lengths of the groups are stored in pre_packing_lengths
                # so that the groups can be separated later
                for pre_pack in pre_packs:
                    self._pre_packing_buffer.extend(pre_pack)
                    self._pre_packing_lengths.append(len(pre_pack))

        def next_final_pack() -> Generator[T_batch_sample, None, None]:
            """Yield the next packs from the buffer. The final packer is called on the fly."""

            pack = list(self._pre_packing_buffer[: self._pre_packing_lengths[0]])
            pack = encode_pack_samples(pack)
            if len(pack) == 0:
                # All samples in the pack were skipped
                return

            del self._pre_packing_buffer[: self._pre_packing_lengths[0]]
            del self._pre_packing_lengths[0]
            try:
                pack_restore_keys = tuple(get_sample_restore_key(sample) for sample in pack)
                with (
                    self._final_packing_sample_index.ctx() as pack_idx,
                    trace_span.span(
                        final_packer_name, args={"pack_idx": pack_idx, "len": len(pack)}, level=2
                    ),
                ):
                    final_packed_sample = self.final_packer(pack)
                if isinstance(final_packed_sample, Generator):
                    assert inspect.isgeneratorfunction(self.final_packer), (
                        f"Generator in {self.final_packer} but not marked as such."
                    )
                    for pack_sub_idx, (pack_idx, inner_batch_sample) in trace_span.iterable(
                        enumerate(
                            self._final_packing_sample_index.iter_ctx(final_packed_sample, pack_idx)
                        ),
                        name=f"{final_packer_name}.next",
                        level=2,
                    ):
                        with trace_gen.yield_(
                            last_args={"pack_idx": pack_idx, "pack_sub_idx": pack_sub_idx}
                        ):
                            yield set_sample_restore_key(
                                inner_batch_sample,
                                pack_idx,
                                pack_sub_idx,
                                *pack_restore_keys,
                                src=self,
                            )
                else:
                    with trace_gen.yield_(last_args={"pack_idx": pack_idx}):
                        yield set_sample_restore_key(
                            final_packed_sample,
                            pack_idx,
                            *pack_restore_keys,
                            src=self,
                        )
            except SkipSample:
                trace_span.instant("PackingDataset.next_final_pack.skip", level=2)
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample(pack)
            except Exception as e:
                self.error_handler(e, pack)
                trace_span.instant(
                    "PackingDataset.next_final_pack.error/skip",
                    args={"exception": f"{type(e).__name__}: {str(e)}"},
                    level=2,
                )

        with (
            trace_span.span(
                "PackingDataset.__iter__", args={"config": self._own_config()}, level=1
            ),
            self.worker_config.worker_trace_writer().generator(
                "PackingDataset.__iter__.next", level=2
            ) as trace_gen,
        ):
            # The source dataset
            src_iter = iter(self.dataset)

            try:
                self._pre_packing_buffer.worker_start()
                self._reading_buffer.worker_start()

                is_initial_pack = True

                pre_pack_round = 0
                # Main loop:
                while True:
                    if pre_pack_round > 10:
                        raise RuntimeError("Pre packer did not yield any packs after 10 rounds.")
                    with trace_span.span(
                        "PackingDataset.__iter__.fill_reading_buffer",
                        args={
                            "to_fill": self.buffer_size
                            - len(self._reading_buffer)
                            - len(self._pre_packing_buffer),
                            "reading_buffer": len(self._reading_buffer),
                            "pre_packing_buffer": len(self._pre_packing_buffer),
                            "buffer_size": self.buffer_size,
                        },
                        level=2,
                    ):
                        # Fill a portion of the buffer
                        if not self._fill_reading_buffer(src_iter, log_progress=is_initial_pack):
                            # Break out of the main loop when the source is exhausted.
                            break
                    is_initial_pack = False

                    # Create new pre packs if necessary
                    if len(self._pre_packing_lengths) == 0:
                        with trace_span.span("PackingDataset.__iter__.next_pre_pack", level=1):
                            assert len(self._pre_packing_buffer) == 0
                            assert len(self._reading_buffer) == self.buffer_size
                            next_pre_pack()
                            if len(self._pre_packing_lengths) == 0:
                                # Retry packing, nothing was returned.
                                pre_pack_round += 1
                                continue
                    # Reset the pre pack round counter for failing
                    pre_pack_round = 0

                    with trace_span.span("PackingDataset.__iter__.final_pack", level=2):
                        yield from next_final_pack()

                with trace_span.span("PackingDataset.__iter__.last", level=1):
                    # Yield the remaining packs, flushing the collecting buffer
                    while len(self._pre_packing_lengths) > 0:
                        with trace_span.span("PackingDataset.__iter__.last.final_pack", level=2):
                            yield from next_final_pack()

                    # If there are still samples in the partial reading buffer, pre-pack them and yield the
                    # resulting (partial) packs
                    if len(self._reading_buffer) > 0:
                        with trace_span.span("PackingDataset.__iter__.last.next_pre_pack", level=1):
                            next_pre_pack()

                    # Yield the remaining packs, flushing the collecting buffer
                    while len(self._pre_packing_lengths) > 0:
                        with trace_span.span("PackingDataset.__iter__.last.final_pack", level=2):
                            yield from next_final_pack()
            finally:
                if hasattr(src_iter, "close"):
                    src_iter.close()

    def can_restore_sample(self) -> bool:
        # Cannot really verify if the returned elements contain a __restore_key__.
        # If the user wants to use this, well...
        return (
            super().can_restore_sample()
            and self.final_packer_stateless
            and self.sample_encoder_stateless
        )

    def assert_can_restore(self):
        assert self.final_packer_stateless and self.sample_encoder_stateless, (
            f"Final packer {self.final_packer} and sample encoder {self.sample_encoder} must be stateless to restore samples."
        )
        super().assert_can_restore()

    def restore_sample(self, restore_key: Any) -> T_sample:
        trace_span = self.worker_config.worker_trace_span()
        # We need to store multiple indices to restore a batch.
        self.assert_can_restore()
        with trace_span.span(
            "PackingDataset.restore_sample", args={"restore_key": restore_key}, level=1
        ):
            if inspect.isgeneratorfunction(self.final_packer):
                id, pack_idx, pack_sub_idx, *pack_restore_keys = restore_key
                assert id == type(self).__name__
            else:
                id, pack_idx, *pack_restore_keys = restore_key
                assert id == type(self).__name__

            with trace_span.span(
                "PackingDataset.restore_sample.restore_samples",
                args={"len": len(pack_restore_keys)},
                level=2,
            ):
                pack = []
                for inner_sample_idx, inner_idx in enumerate(pack_restore_keys):
                    if self.sample_encoder is not None:
                        id, sample_idx, *inner_idx = inner_idx
                        assert id == type(self).__name__
                        assert isinstance(sample_idx, int)
                    with trace_span.span(
                        "PackingDataset.restore_sample.dataset",
                        args={"sample_idx": inner_sample_idx},
                        level=2,
                    ):
                        sample = self.dataset.restore_sample(inner_idx)
                    if self.sample_encoder is not None:
                        with (
                            self._sample_encoder_sample_index.ctx(sample_idx),
                            trace_span.span(
                                f"PackingDataset.restore_sample.sample_encoder:{self._function_config(self.sample_encoder)}",
                                args={"sample_idx": sample_idx},
                                level=2,
                            ),
                        ):
                            sample = self.sample_encoder(sample)
                        assert not isinstance(sample, Generator), "Generator not supported"
                        sample = add_sample_restore_key(sample, sample_idx, src=self)
                    pack.append(sample)
            with (
                self._final_packing_sample_index.ctx(pack_idx),
                trace_span.span(
                    f"PackingDataset.restore_sample.final_packer:{self._function_config(self.final_packer)}",
                    args={"pack_idx": pack_idx},
                    level=2,
                ),
            ):
                final_pack = self.final_packer(pack)
            if isinstance(final_pack, Generator):
                assert inspect.isgeneratorfunction(self.final_packer), (
                    f"Generator in {self.final_packer} but not marked as such."
                )
                for cur_batch_sub_idx, (pack_idx, inner_batch_sample) in trace_span.iterable(
                    self._final_packing_sample_index.iter_ctx(final_pack, pack_idx),
                    name=f"PackingDataset.restore_sample.final_packer:{self._function_config(self.final_packer)}.next",
                    level=2,
                ):
                    if cur_batch_sub_idx == pack_sub_idx:
                        return set_sample_restore_key(
                            inner_batch_sample,
                            pack_idx,
                            pack_sub_idx,
                            *pack_restore_keys,
                            src=self,
                        )
                assert False, f"Pack sub-index {pack_sub_idx} not found in pack"
            else:
                return set_sample_restore_key(final_pack, pack_idx, *pack_restore_keys, src=self)

    def _own_config(self) -> Dict[str, Any]:
        return {
            "buffer_size": self.buffer_size,
            "pre_packer": self._function_config(self.pre_packer),
            "final_packer": self._function_config(self.final_packer),
            "final_packer_stateless": self.final_packer_stateless,
            **(
                {
                    "packer_config": (
                        self.packer_config() if callable(self.packer_config) else self.packer_config
                    )
                }
                if self.packer_config
                else {}
            ),
            "error_handler": self._function_config(self.error_handler),
        }

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "buffer_size": self.buffer_size,
            "pre_packer": self._function_config(self.pre_packer),
            "final_packer": self._function_config(self.final_packer),
            "final_packer_stateless": self.final_packer_stateless,
            **(
                {
                    "packer_config": (
                        self.packer_config() if callable(self.packer_config) else self.packer_config
                    )
                }
                if self.packer_config
                else {}
            ),
            "error_handler": self._function_config(self.error_handler),
            "worker_config": self.worker_config.config(),
            "dataset": self.dataset.config(),
        }

    def __str__(self):
        return f"PackingDataset(buffer_size={self.buffer_size}, pre_packer={self.pre_packer}, final_packer={self.final_packer}, dataset={self.dataset})"
