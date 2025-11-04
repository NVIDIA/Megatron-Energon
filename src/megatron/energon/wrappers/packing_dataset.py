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

from megatron.energon.errors import ErrorContext, handle_restore_errors
from megatron.energon.flavors.base_dataset import (
    SavableDataset,
    add_sample_restore_key,
    set_sample_restore_key,
)
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset, SampleIndex, get_sample_restore_key
from megatron.energon.wrappers.buffer import SavableSampleBuffer

T_sample = TypeVar("T_sample")
T_encoded_sample = TypeVar("T_encoded_sample")
T_batch_sample = TypeVar("T_batch_sample")


class PackingDataset(
    BaseWrapperDataset[T_sample, T_batch_sample],
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

    #: Error handlers for tracking failures
    _pre_pack_failure_handler: ErrorContext
    _final_pack_failure_handler: ErrorContext
    _sample_encoder_failure_handler: ErrorContext | None

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
        pre_packer_failure_tolerance: int = 100,
        final_packer_failure_tolerance: int = 100,
        sample_encoder_failure_tolerance: int = 100,
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
            pre_packer_failure_tolerance: Maximum number of pre-packer failures before raising an error. Set to 0 to disable.
            final_packer_failure_tolerance: Maximum number of final-packer failures before raising an error. Set to 0 to disable.
            sample_encoder_failure_tolerance: Maximum number of sample-encoder failures before raising an error. Set to 0 to disable.
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

        self.pre_packer_failure_tolerance = pre_packer_failure_tolerance
        self.final_packer_failure_tolerance = final_packer_failure_tolerance
        self.sample_encoder_failure_tolerance = sample_encoder_failure_tolerance

        self._pre_pack_failure_handler = ErrorContext(
            name=f"PackingDataset.{self.pre_packer}",
            handler=worker_config.global_error_handler,
            tolerance=pre_packer_failure_tolerance,
        )
        self._final_pack_failure_handler = ErrorContext(
            name=f"PackingDataset.{self.final_packer}",
            handler=worker_config.global_error_handler,
            tolerance=final_packer_failure_tolerance,
        )
        if self.sample_encoder is not None:
            self._sample_encoder_failure_handler = ErrorContext(
                name=f"PackingDataset.{self.sample_encoder}",
                handler=worker_config.global_error_handler,
                tolerance=sample_encoder_failure_tolerance,
            )
        else:
            self._sample_encoder_failure_handler = None

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

    def len_worker(self, worker_idx: int | None = None) -> int:
        # The real length is unknown, since it depends on the packing function.
        # We approximate it by the length of the source dataset.
        return self.dataset.len_worker(worker_idx)

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
            while (
                self._reading_buffer.len_worker() + self._pre_packing_buffer.len_worker()
                < self.buffer_size
            ):
                try:
                    sample = next(source_iter)
                    self._reading_buffer.append(sample)
                    if pbar is not None:
                        pbar.update(1)
                except StopIteration:
                    return False
        return True

    def __iter__(self) -> Iterator[T_batch_sample]:
        pre_packing_lengths = self._pre_packing_lengths
        # The source dataset
        src_iter = iter(self.dataset)

        self._pre_packing_buffer.worker_start()
        self._reading_buffer.worker_start()

        is_initial_pack = True

        def encode_pack_samples(pack: List[T_sample]) -> List[T_encoded_sample]:
            """Encode the samples in the pack using the sample encoder."""

            # Apply the sample encoder to the pack
            if self.sample_encoder is None:
                return pack
            encoded_pack = []
            for sample in pack:
                with self._sample_encoder_failure_handler.handle_errors(sample):
                    with self._sample_encoder_sample_index.ctx() as encode_idx:
                        encoded_sample = self.sample_encoder(sample)
                    assert not isinstance(encoded_sample, Generator), "Generator not supported"
                    self._sample_encoder_failure_handler.reset()
                    encoded_pack.append(
                        add_sample_restore_key(
                            encoded_sample,
                            encode_idx,
                            src=self,
                        )
                    )
            return encoded_pack

        def next_pre_pack():
            """Take the samples from the reading buffer and select groups of samples to be packed
            together."""

            assert self._pre_packing_buffer.len_worker() == 0
            if self._reading_buffer.len_worker() > 0:
                # Take all samples from the reading buffer and pre_pack them
                samples = self._reading_buffer.buffer.copy()
                # Clear buffer and pre_packing_lengths
                self._reading_buffer.clear()
                pre_packing_lengths.clear()
                # Now pre pack the samples
                pre_packs = []
                with self._pre_pack_failure_handler.handle_errors(samples):
                    with self._pre_packing_sample_index.ctx():
                        pre_packs = self.pre_packer(samples)

                # Put the pre-packed samples into the pre_packing_buffer
                # They will be flattened here to avoid nested buffers
                # But the lengths of the groups are stored in pre_packing_lengths
                # so that the groups can be separated later
                for pre_pack in pre_packs:
                    if len(pre_pack) > 0:
                        self._pre_packing_buffer.extend(pre_pack)
                        pre_packing_lengths.append(len(pre_pack))

        def next_final_pack() -> Generator[T_batch_sample, None, None]:
            """Yield the next packs from the buffer. The final packer is called on the fly."""

            pack = self._pre_packing_buffer.buffer[: pre_packing_lengths[0]].copy()
            if len(pack) == 0:
                return
            pack = encode_pack_samples(pack)

            del self._pre_packing_buffer[: pre_packing_lengths[0]]
            del pre_packing_lengths[0]
            with self._final_pack_failure_handler.handle_errors(pack):
                pack_restore_keys = tuple(get_sample_restore_key(sample) for sample in pack)
                with self._final_packing_sample_index.ctx() as pack_idx:
                    final_packed_sample = self.final_packer(pack)
                if isinstance(final_packed_sample, Generator):
                    assert inspect.isgeneratorfunction(self.final_packer), (
                        f"Generator in {self.final_packer} but not marked as such."
                    )
                    for pack_sub_idx, (pack_idx, inner_batch_sample) in enumerate(
                        self._final_packing_sample_index.iter_ctx(final_packed_sample, pack_idx)
                    ):
                        self._final_pack_failure_handler.reset()
                        yield set_sample_restore_key(
                            inner_batch_sample,
                            pack_idx,
                            pack_sub_idx,
                            *pack_restore_keys,
                            src=self,
                        )
                else:
                    self._final_pack_failure_handler.reset()
                    yield set_sample_restore_key(
                        final_packed_sample,
                        pack_idx,
                        *pack_restore_keys,
                        src=self,
                    )

        # Main loop:
        pre_pack_round = 0
        while True:
            if (
                self.pre_packer_failure_tolerance > 0
                and pre_pack_round > self.pre_packer_failure_tolerance
            ):
                raise RuntimeError(
                    f"Pre packer {self.pre_packer} did not yield any packs after {pre_pack_round} rounds. Likely your code or dataset are broken."
                )
            # Fill a portion of the buffer
            if not self._fill_reading_buffer(src_iter, log_progress=is_initial_pack):
                # Break out of the main loop when the source is exhausted.
                break
            is_initial_pack = False

            # Create new pre packs if necessary
            if len(pre_packing_lengths) == 0:
                assert self._pre_packing_buffer.len_worker() == 0
                assert self._reading_buffer.len_worker() == self.buffer_size
                next_pre_pack()
                if len(pre_packing_lengths) == 0:
                    # Retry packing, nothing was returned.
                    pre_pack_round += 1
                    continue

            if len(pre_packing_lengths) > 0:
                pre_pack_round = 0

            yield from next_final_pack()

        # Yield the remaining packs, flushing the collecting buffer
        while len(pre_packing_lengths) > 0:
            yield from next_final_pack()

        # If there are still samples in the partial reading buffer, pre-pack them and yield the
        # resulting (partial) packs
        if self._reading_buffer.len_worker() > 0:
            next_pre_pack()

        # Yield the remaining packs, flushing the collecting buffer
        while len(pre_packing_lengths) > 0:
            yield from next_final_pack()

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
        # We need to store multiple indices to restore a batch.
        self.assert_can_restore()
        if inspect.isgeneratorfunction(self.final_packer):
            id, pack_idx, pack_sub_idx, *pack_restore_keys = restore_key
            id, pack_idx, pack_sub_idx, *pack_restore_keys = restore_key
            assert id == type(self).__name__
        else:
            id, pack_idx, *pack_restore_keys = restore_key
            id, pack_idx, *pack_restore_keys = restore_key
            assert id == type(self).__name__

        pack = []
        for inner_idx in pack_restore_keys:
            if self.sample_encoder is not None:
                id, sample_idx, *inner_idx = inner_idx
                assert id == type(self).__name__
                id, sample_idx, *inner_idx = inner_idx
                assert id == type(self).__name__
                assert isinstance(sample_idx, int)
            sample = self.dataset.restore_sample(inner_idx)
            if self.sample_encoder is not None:
                with handle_restore_errors(self.worker_config.restore_error_handler, sample):
                    with self._sample_encoder_sample_index.ctx(sample_idx):
                        sample = self.sample_encoder(sample)
                    assert not isinstance(sample, Generator), "Generator not supported"
                    sample = add_sample_restore_key(sample, sample_idx, src=self)

            pack.append(sample)

        with handle_restore_errors(self.worker_config.restore_error_handler, pack):
            with self._final_packing_sample_index.ctx(pack_idx):
                final_pack = self.final_packer(pack)
            if isinstance(final_pack, Generator):
                assert inspect.isgeneratorfunction(self.final_packer), (
                    f"Generator in {self.final_packer} but not marked as such."
                )
                for cur_batch_sub_idx, (pack_idx, inner_batch_sample) in enumerate(
                    self._final_packing_sample_index.iter_ctx(final_pack, pack_idx)
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
            "worker_config": self.worker_config.config(),
            "dataset": self.dataset.config(),
        }

    def __str__(self):
        return f"PackingDataset(buffer_size={self.buffer_size}, pre_packer={self.pre_packer}, final_packer={self.final_packer}, dataset={self.dataset})"
