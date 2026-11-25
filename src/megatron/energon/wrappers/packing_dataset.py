# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import inspect
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

from megatron.energon.edataclass import edataclass
from megatron.energon.errors import ErrorContext, handle_restore_errors
from megatron.energon.flavors.base_dataset import (
    FlexState,
    RestoreKey,
    SavableDataset,
    set_sample_restore_key,
)
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import (
    BaseWrapperDataset,
    MultiWrappedRestoreKey,
    SampleIndex,
    WrappedRestoreKey,
    get_sample_restore_key,
)
from megatron.energon.wrappers.buffer import SavableSampleBuffer

T_sample = TypeVar("T_sample")
T_encoded_sample = TypeVar("T_encoded_sample")
T_batch_sample = TypeVar("T_batch_sample")
T_slice = TypeVar("T_slice")


@dataclass(frozen=True, slots=True, kw_only=True)
class PartialSampleRestoreKey(WrappedRestoreKey):
    slice: Any


@dataclass(frozen=True, slots=True, kw_only=True)
class EncodePackRestoreKey(WrappedRestoreKey):
    sample_idx: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PackingRestoreKey(MultiWrappedRestoreKey):
    pack_idx: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PackingGenRestoreKey(PackingRestoreKey):
    gen_idx: int


@edataclass
class PartialSample(Generic[T_sample, T_slice]):
    """A view onto part of a sample selected during packing."""

    sample: T_sample
    slice: T_slice
    __restore_key__: PartialSampleRestoreKey | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        inner_restore_key = get_sample_restore_key(self.sample)
        if inner_restore_key is not None:
            self.__restore_key__ = PartialSampleRestoreKey(
                inner=inner_restore_key, slice=self.slice
            )


class SavablePartialSampleBuffer(
    SavableSampleBuffer[T_sample | PartialSample[T_sample, T_slice]],
    Generic[T_sample, T_slice],
):
    """Sample buffer that can save and restore packing-only partial sample wrappers."""

    def restore_sample(
        self, restore_key: RestoreKey
    ) -> T_sample | PartialSample[T_sample, T_slice]:
        if isinstance(restore_key, PartialSampleRestoreKey):
            return PartialSample(
                sample=self.restore_sample(restore_key.inner), slice=restore_key.slice
            )
        return self.dataset.restore_sample(restore_key)


@dataclass(slots=True, frozen=True)
class PackedSamplesOutput(Generic[T_sample]):
    """Return type for :meth:`~megatron.energon.TaskEncoder.select_samples_to_pack` when including
    samples to push back onto the packing reading buffer."""

    #: The packs of samples to be packed together.
    # One entry per pack, each containing the samples to be packed together.
    packs: list[list[T_sample]]

    #: The samples to push back onto the packing reading buffer for the next fill.
    pushback: Sequence[T_sample] = field(default_factory=tuple)


class PackingDataset(
    BaseWrapperDataset[T_sample, T_batch_sample],
    Generic[T_sample, T_encoded_sample, T_batch_sample, T_slice],
):
    """This dataset wrapper transforms samples of a dataset into chunks/packs of samples, which are
    then combined into a batch."""

    _worker_local_fields = {"_state_restored": bool}

    buffer_size: int
    pre_packer: Callable[
        [list[T_sample | PartialSample[T_sample, Any]]],
        list[list[T_sample | PartialSample[T_sample, Any]]]
        | PackedSamplesOutput[T_sample | PartialSample[T_sample, Any]],
    ]
    sample_encoder: Optional[Callable[[T_sample | PartialSample[T_sample, Any]], T_encoded_sample]]
    sample_encoder_stateless: bool
    final_packer: Callable[
        [List[T_encoded_sample | T_sample | PartialSample[T_sample, Any]]], T_batch_sample
    ]
    final_packer_stateless: bool
    packer_config: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]]
    pre_packer_failure_tolerance: int
    sample_encoder_failure_tolerance: int
    final_packer_failure_tolerance: int

    #: The buffer for collecting the samples that shall be packed.
    _reading_buffer: SavablePartialSampleBuffer[T_sample, T_slice]

    #: Contains the pre-selected samples to be packed.
    #: The full buffer will be passed to the pre_packer.
    _pre_packing_buffer: SavablePartialSampleBuffer[T_sample, T_slice]

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

    #: Set once the input dataset is exhausted, so that a restored state does not start a new pass.
    _source_exhausted: bool

    #: The pack whose final packer generator is being emitted, and how many outputs were emitted.
    _generator_pack_idx: Optional[int]
    _generator_sample_keys: tuple[RestoreKey | None, ...] | None
    _generator_offset: Optional[int]

    #: Whether the state was restored since the last iteration started (not saved).
    _state_restored: bool

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
        "_source_exhausted",
        "_generator_pack_idx",
        "_generator_sample_keys",
        "_generator_offset",
    )

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        buffer_size: int,
        pre_packer: Callable[
            [list[T_sample | PartialSample[T_sample, T_slice]]],
            list[list[T_sample | PartialSample[T_sample, T_slice]]]
            | PackedSamplesOutput[T_sample | PartialSample[T_sample, T_slice]],
        ],
        final_packer: Callable[
            [List[T_encoded_sample | T_sample | PartialSample[T_sample, T_slice]]], T_batch_sample
        ],
        *,
        final_packer_stateless: bool = False,
        sample_encoder: Optional[
            Callable[[T_sample | PartialSample[T_sample, T_slice]], T_encoded_sample]
        ] = None,
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
                May return :class:`PackedSamplesOutput` to include a ``pushback`` sequence
                reapplied to the reading buffer. May raise :exc:`megatron.energon.SkipSample` to skip a buffer.
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
        self._reading_buffer = SavablePartialSampleBuffer(
            self.dataset,
            worker_config=self.worker_config,
        )
        self._pre_packing_buffer = SavablePartialSampleBuffer(
            self.dataset,
            worker_config=self.worker_config,
        )
        self._pre_packing_lengths = []
        self._pre_packing_sample_index = SampleIndex(self.worker_config, src=self)
        self._final_packing_sample_index = SampleIndex(self.worker_config, src=self)
        self._sample_encoder_sample_index = SampleIndex(self.worker_config, src=self)
        self._source_exhausted = False
        self._generator_pack_idx = None
        self._generator_sample_keys = None
        self._generator_offset = None
        self._state_restored = False

    def restore_state(self, state: FlexState) -> None:
        super().restore_state(state)
        self._state_restored = True

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

    def _restore_pack(
        self,
        pack_restore_keys: Sequence[RestoreKey | None],
    ) -> List[T_encoded_sample | T_sample | PartialSample[T_sample, Any]]:
        pack = []
        for restore_key in pack_restore_keys:
            if restore_key is None:
                raise ValueError("Cannot restore a pack containing a sample without a restore key")
            if self.sample_encoder is None:
                pack.append(self._reading_buffer.restore_sample(restore_key))
                continue

            key = cast(EncodePackRestoreKey, restore_key)
            sample = self._reading_buffer.restore_sample(key.inner)
            with handle_restore_errors(self.worker_config.restore_error_handler, sample):
                with self._sample_encoder_sample_index.ctx(key.sample_idx):
                    encoded_sample = self.sample_encoder(sample)
                assert not isinstance(encoded_sample, Generator), "Generator not supported"
                pack.append(set_sample_restore_key(encoded_sample, key))
        return pack

    def _finalize_pack(
        self,
        pack: List[T_encoded_sample | T_sample | PartialSample[T_sample, Any]],
        *,
        pack_idx: Optional[int] = None,
        skip_outputs: int = 0,
    ) -> Generator[T_batch_sample, None, None]:
        """Runs the final packer on an encoded pack.

        Args:
            pack: The encoded pack members.
            pack_idx: The final packer sample index, if continuing a restored generator pack.
            skip_outputs: Number of generator outputs that were already emitted before restoring.
        """
        with self._final_pack_failure_handler.handle_errors(pack):
            pack_restore_keys = tuple(get_sample_restore_key(sample) for sample in pack)
            with self._final_packing_sample_index.ctx(pack_idx) as pack_idx:
                final_packed_sample = self.final_packer(pack)
            if isinstance(final_packed_sample, Generator):
                assert inspect.isgeneratorfunction(self.final_packer), (
                    f"Generator in {self.final_packer} but not marked as such."
                )
                # The pack members are already consumed, so save the pack until all its outputs
                # are emitted.
                self._generator_pack_idx = pack_idx
                self._generator_sample_keys = pack_restore_keys
                self._generator_offset = skip_outputs
                for pack_sub_idx, (pack_idx, inner_batch_sample) in enumerate(
                    self._final_packing_sample_index.iter_ctx(final_packed_sample, pack_idx)
                ):
                    if pack_sub_idx < skip_outputs:
                        continue
                    self._generator_offset = pack_sub_idx + 1
                    self._final_pack_failure_handler.reset()
                    yield set_sample_restore_key(
                        inner_batch_sample,
                        PackingGenRestoreKey(
                            pack_idx=pack_idx, gen_idx=pack_sub_idx, inner=pack_restore_keys
                        ),
                    )
            else:
                self._final_pack_failure_handler.reset()
                yield set_sample_restore_key(
                    final_packed_sample,
                    PackingRestoreKey(pack_idx=pack_idx, inner=pack_restore_keys),
                )
        self._generator_pack_idx = None
        self._generator_sample_keys = None
        self._generator_offset = None

    def __iter__(self) -> Iterator[T_batch_sample]:
        pre_packing_lengths = self._pre_packing_lengths
        if not self._state_restored:
            # A new iteration starts a new pass, even if the previous one was not iterated to its
            # end (e.g. stopped right after the last pack).
            self._source_exhausted = False
            self._generator_pack_idx = None
            self._generator_sample_keys = None
            self._generator_offset = None
        self._state_restored = False
        # The source dataset. When restored after it was exhausted, do not start another pass.
        src_iter = iter(()) if self._source_exhausted else iter(self.dataset)

        self._pre_packing_buffer.worker_start()
        self._reading_buffer.worker_start()

        if self._generator_sample_keys is not None:
            # Restored in the middle of a pack: emit its remaining final packer outputs first.
            yield from self._finalize_pack(
                self._restore_pack(self._generator_sample_keys),
                pack_idx=self._generator_pack_idx,
                skip_outputs=self._generator_offset,
            )

        is_initial_pack = True

        def encode_pack_samples(
            pack: List[T_sample | PartialSample[T_sample, Any]],
        ) -> List[T_encoded_sample | T_sample | PartialSample[T_sample, Any]]:
            """Encode the samples in the pack using the sample encoder."""

            # Apply the sample encoder to the pack
            if self.sample_encoder is None:
                return pack
            encoded_pack = []
            for sample in pack:
                input_restore_key = get_sample_restore_key(sample)
                with self._sample_encoder_failure_handler.handle_errors(sample):
                    with self._sample_encoder_sample_index.ctx() as encode_idx:
                        encoded_sample = self.sample_encoder(sample)
                    assert not isinstance(encoded_sample, Generator), "Generator not supported"
                    self._sample_encoder_failure_handler.reset()
                    encoded_pack.append(
                        set_sample_restore_key(
                            encoded_sample,
                            None
                            if input_restore_key is None
                            else EncodePackRestoreKey(
                                sample_idx=encode_idx, inner=input_restore_key
                            ),
                        )
                    )
            return encoded_pack

        def next_pre_pack():
            """Take the samples from the reading buffer and select groups of samples to be packed
            together."""

            assert self._pre_packing_buffer.len_worker() == 0
            if self._reading_buffer.len_worker() == 0:
                return

            # Take all samples from the reading buffer and pre_pack them
            samples = self._reading_buffer.buffer.copy()
            # Clear buffer and pre_packing_lengths
            self._reading_buffer.clear()
            pre_packing_lengths.clear()
            # Now pre pack the samples
            pre_packs_result: list[list[T_sample]] | PackedSamplesOutput[T_sample] | None = None
            with self._pre_pack_failure_handler.handle_errors(samples):
                with self._pre_packing_sample_index.ctx():
                    pre_packs_result = self.pre_packer(samples)
            if pre_packs_result is None:
                # The error handler may actually catch the error
                return
            # Put the pre-packed samples into the pre_packing_buffer
            # They will be flattened here to avoid nested buffers
            # But the lengths of the groups are stored in pre_packing_lengths
            # so that the groups can be separated later
            if isinstance(pre_packs_result, PackedSamplesOutput):
                pre_packs = pre_packs_result.packs
                self._reading_buffer.extend(list(pre_packs_result.pushback))
            else:
                pre_packs = pre_packs_result
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
            yield from self._finalize_pack(pack)

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
                self._source_exhausted = True
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

        # Yield the remaining packs, flushing carryover partials until the source is exhausted.
        flush_round = 0
        while len(pre_packing_lengths) > 0 or self._reading_buffer.len_worker() > 0:
            if len(pre_packing_lengths) > 0:
                flush_round = 0
                yield from next_final_pack()
                continue
            # If there are still samples in the partial reading buffer, pre-pack them and yield the
            # resulting (partial) packs

            next_pre_pack()
            if len(pre_packing_lengths) == 0 and self._reading_buffer.len_worker() > 0:
                flush_round += 1
                if (
                    self.pre_packer_failure_tolerance > 0
                    and flush_round > self.pre_packer_failure_tolerance
                ):
                    raise RuntimeError(
                        f"Pre packer {self.pre_packer} did not yield any packs after {flush_round} flush rounds. Likely your code or dataset are broken."
                    )
        self._source_exhausted = False

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

    def restore_sample(self, restore_key: RestoreKey) -> T_sample:
        self.assert_can_restore()
        key = cast(PackingRestoreKey, restore_key)
        pack = self._restore_pack(key.inner)

        with handle_restore_errors(self.worker_config.restore_error_handler, pack):
            with self._final_packing_sample_index.ctx(key.pack_idx):
                final_pack = self.final_packer(pack)
            if isinstance(key, PackingGenRestoreKey):
                assert isinstance(final_pack, Generator)
                for idx, (_, sample) in enumerate(
                    self._final_packing_sample_index.iter_ctx(final_pack, key.pack_idx)
                ):
                    if idx == key.gen_idx:
                        return set_sample_restore_key(sample, key)
                raise ValueError(f"Pack output {key.gen_idx} is missing for {key!r}")
            return set_sample_restore_key(final_pack, key)

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
