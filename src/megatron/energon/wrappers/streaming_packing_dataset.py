# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

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
    Sequence,
    TypeVar,
    Union,
    cast,
)

from megatron.energon.errors import ErrorContext, handle_restore_errors
from megatron.energon.flavors.base_dataset import (
    SavableDataset,
    add_sample_restore_key,
    set_sample_restore_key,
)
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset, SampleIndex, get_sample_restore_key
from megatron.energon.wrappers.packing_dataset import (
    PackedSamplesOutput,
    PartialSample,
    SavablePartialSampleBuffer,
)

T_sample = TypeVar("T_sample")
T_encoded_sample = TypeVar("T_encoded_sample")
T_batch_sample = TypeVar("T_batch_sample")
T_slice = TypeVar("T_slice")


class StreamingPackingDataset(
    BaseWrapperDataset[T_sample, T_batch_sample],
    Generic[T_sample, T_encoded_sample, T_batch_sample, T_slice],
):
    """Streaming sequence packing wrapper.

    Unlike :class:`PackingDataset`, this wrapper does not fill a fixed-size reading buffer.
    The selector receives an iterator and pulls only the samples it needs for the next pack.
    """

    select_next_pack: Callable[
        [Iterator[T_sample | PartialSample[T_sample, T_slice]]],
        list[list[T_sample | PartialSample[T_sample, T_slice]]]
        | PackedSamplesOutput[T_sample | PartialSample[T_sample, T_slice]],
    ]
    sample_encoder: Optional[
        Callable[[T_sample | PartialSample[T_sample, T_slice]], T_encoded_sample]
    ]
    sample_encoder_stateless: bool
    final_packer: Callable[
        [List[T_encoded_sample | T_sample | PartialSample[T_sample, T_slice]]], T_batch_sample
    ]
    final_packer_stateless: bool
    packer_config: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]]

    _carryover_buffer: SavablePartialSampleBuffer[T_sample, T_slice]
    _select_sample_index: SampleIndex
    _sample_encoder_sample_index: SampleIndex
    _final_packing_sample_index: SampleIndex
    _skip_mode: bool

    _select_failure_handler: ErrorContext
    _final_pack_failure_handler: ErrorContext
    _sample_encoder_failure_handler: ErrorContext | None

    _savable_fields = (
        "_carryover_buffer",
        "_select_sample_index",
        "_sample_encoder_sample_index",
        "_final_packing_sample_index",
    )

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        select_next_pack: Callable[
            [Iterator[T_sample | PartialSample[T_sample, T_slice]]],
            list[list[T_sample | PartialSample[T_sample, T_slice]]]
            | PackedSamplesOutput[T_sample | PartialSample[T_sample, T_slice]],
        ],
        final_packer: Callable[
            [List[T_encoded_sample | T_sample | PartialSample[T_sample, T_slice]]], T_batch_sample
        ],
        *,
        final_packer_stateless: bool = False,
        final_packer_skip_safe: bool = False,
        sample_encoder: Optional[
            Callable[[T_sample | PartialSample[T_sample, T_slice]], T_encoded_sample]
        ] = None,
        sample_encoder_stateless: bool = False,
        sample_encoder_skip_safe: bool = False,
        packer_config: Optional[Union[Dict[str, Any], Callable[[], Dict[str, Any]]]] = None,
        select_failure_tolerance: int = 100,
        final_packer_failure_tolerance: int = 100,
        sample_encoder_failure_tolerance: int = 100,
        worker_config: WorkerConfig,
    ):
        """Construct a streaming packing dataset.

        Args:
            dataset: The input dataset to wrap.
            select_next_pack: Pull-based selector for one pack. It may return a list with at most
                one pack, or :class:`PackedSamplesOutput` with at most one pack plus pushback.
            final_packer: Function which combines the selected samples into a single sample.
            final_packer_stateless: If True, the final packer is stateless and restorable.
            final_packer_skip_safe: If True, the final packer can be elided in skip mode.
            sample_encoder: Optional per-pack-member encoder, usually ``postencode_sample``.
            sample_encoder_stateless: If True, the sample encoder is stateless and restorable.
            sample_encoder_skip_safe: If True, the sample encoder can be elided in skip mode.
            packer_config: Configuration for packer functions.
            select_failure_tolerance: Maximum selector failures or empty retries before raising.
            final_packer_failure_tolerance: Maximum number of final-packer failures.
            sample_encoder_failure_tolerance: Maximum number of sample-encoder failures.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset, worker_config=worker_config)

        self.select_next_pack = select_next_pack
        self.final_packer = final_packer
        self.final_packer_stateless = final_packer_stateless
        self.final_packer_skip_safe = final_packer_skip_safe
        self.sample_encoder = sample_encoder
        self.sample_encoder_stateless = True if sample_encoder is None else sample_encoder_stateless
        self.sample_encoder_skip_safe = True if sample_encoder is None else sample_encoder_skip_safe
        self.packer_config = packer_config

        self.select_failure_tolerance = select_failure_tolerance
        self.final_packer_failure_tolerance = final_packer_failure_tolerance
        self.sample_encoder_failure_tolerance = sample_encoder_failure_tolerance

        self._select_failure_handler = ErrorContext(
            name=f"StreamingPackingDataset.{self.select_next_pack}",
            handler=worker_config.global_error_handler,
            tolerance=select_failure_tolerance,
        )
        self._final_pack_failure_handler = ErrorContext(
            name=f"StreamingPackingDataset.{self.final_packer}",
            handler=worker_config.global_error_handler,
            tolerance=final_packer_failure_tolerance,
        )
        if self.sample_encoder is not None:
            self._sample_encoder_failure_handler = ErrorContext(
                name=f"StreamingPackingDataset.{self.sample_encoder}",
                handler=worker_config.global_error_handler,
                tolerance=sample_encoder_failure_tolerance,
            )
        else:
            self._sample_encoder_failure_handler = None

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._carryover_buffer = SavablePartialSampleBuffer(
            self.dataset,
            worker_config=self.worker_config,
        )
        self._select_sample_index = SampleIndex(self.worker_config, src=self)
        self._final_packing_sample_index = SampleIndex(self.worker_config, src=self)
        self._sample_encoder_sample_index = SampleIndex(self.worker_config, src=self)
        self._skip_mode = False

    def len_worker(self, worker_idx: int | None = None) -> int:
        # The real length depends on pack boundaries and partial carryover.
        return self.dataset.len_worker(worker_idx)

    def set_skip_mode(self, active: bool) -> None:
        self._skip_mode = active

    def _normalize_selection(
        self,
        selected: list[list[T_sample | PartialSample[T_sample, T_slice]]]
        | PackedSamplesOutput[T_sample | PartialSample[T_sample, T_slice]],
    ) -> tuple[
        list[T_sample | PartialSample[T_sample, T_slice]],
        Sequence[T_sample | PartialSample[T_sample, T_slice]],
    ]:
        if isinstance(selected, PackedSamplesOutput):
            packs = selected.packs
            pushback = selected.pushback
        else:
            packs = selected
            pushback = ()

        non_empty_packs = [pack for pack in packs if len(pack) > 0]
        if len(non_empty_packs) > 1:
            raise ValueError(
                "Streaming packing expects select_next_pack to return at most one non-empty pack."
            )
        if len(non_empty_packs) == 0:
            return [], pushback
        return non_empty_packs[0], pushback

    def _encode_pack_samples(
        self,
        pack: List[T_sample | PartialSample[T_sample, Any]],
    ) -> List[T_encoded_sample | T_sample | PartialSample[T_sample, Any]]:
        if self._skip_mode and self.sample_encoder_skip_safe and self.final_packer_skip_safe:
            self._sample_encoder_sample_index.skip(len(pack))
            return len(pack) * [None]

        if self.sample_encoder is None:
            return pack

        encoded_pack = []
        for sample in pack:
            input_restore_key = get_sample_restore_key(sample)
            with self._sample_encoder_failure_handler.handle_errors(sample):
                with self._sample_encoder_sample_index.ctx() as encode_idx:
                    encoded_sample = self.sample_encoder(sample)
                assert not isinstance(encoded_sample, Generator), "Generator not supported"
                if isinstance(sample, PartialSample) and input_restore_key is not None:
                    encoded_sample = set_sample_restore_key(
                        encoded_sample,
                        *input_restore_key[1:],
                        src=sample,
                    )
                self._sample_encoder_failure_handler.reset()
                encoded_pack.append(
                    add_sample_restore_key(
                        encoded_sample,
                        encode_idx,
                        src=self,
                    )
                )
        return encoded_pack

    def _finalize_pack(
        self,
        pack: List[T_sample | PartialSample[T_sample, Any]],
    ) -> Generator[T_batch_sample, None, None]:
        pack = self._encode_pack_samples(pack)

        if self._skip_mode and self.final_packer_skip_safe:
            self._final_packing_sample_index.skip(1)
            yield cast(T_batch_sample, None)
            return

        with self._final_pack_failure_handler.handle_errors(pack):
            pack_restore_keys = tuple(get_sample_restore_key(sample) for sample in pack)
            with self._final_packing_sample_index.ctx() as pack_idx:
                final_packed_sample = self.final_packer(pack)
            if isinstance(final_packed_sample, Generator):
                assert not self.final_packer_skip_safe, "Generator in final_packer but skip_safe"
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

    def __iter__(self) -> Iterator[T_batch_sample]:
        self._carryover_buffer.worker_start()
        src_iter = iter(self.dataset)
        empty_rounds = 0

        while True:
            sample_iter = _StreamingSampleIterator(self._carryover_buffer, src_iter)
            selected = None
            with self._select_failure_handler.handle_errors(self._carryover_buffer.buffer.copy()):
                with self._select_sample_index.ctx():
                    selected = self.select_next_pack(sample_iter)
            if selected is None:
                empty_rounds += 1
            else:
                assert not isinstance(selected, Generator), "Generator not supported"
                pack, pushback = self._normalize_selection(selected)
                self._carryover_buffer.extend(list(pushback))
                if len(pack) > 0:
                    empty_rounds = 0
                    yield from self._finalize_pack(pack)
                    continue
                empty_rounds += 1

            if sample_iter.source_exhausted and self._carryover_buffer.len_worker() == 0:
                break
            if self.select_failure_tolerance > 0 and empty_rounds > self.select_failure_tolerance:
                raise RuntimeError(
                    f"Streaming pack selector {self.select_next_pack} did not yield any packs after {empty_rounds} rounds. Likely your code or dataset are broken."
                )

    def can_restore_sample(self) -> bool:
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

    def restore_sample(self, restore_key: Any) -> T_batch_sample:
        self.assert_can_restore()
        if inspect.isgeneratorfunction(self.final_packer):
            id, pack_idx, pack_sub_idx, *pack_restore_keys = restore_key
            assert id == type(self).__name__
        else:
            id, pack_idx, *pack_restore_keys = restore_key
            assert id == type(self).__name__

        pack = []
        for inner_idx in pack_restore_keys:
            if self.sample_encoder is not None:
                id, sample_idx, *inner_idx = inner_idx
                assert id == type(self).__name__
                assert isinstance(sample_idx, int)
            sample = self._carryover_buffer.restore_sample(inner_idx)
            if self.sample_encoder is not None:
                with handle_restore_errors(self.worker_config.restore_error_handler, sample):
                    input_sample = sample
                    input_restore_key = get_sample_restore_key(input_sample)
                    with self._sample_encoder_sample_index.ctx(sample_idx):
                        sample = self.sample_encoder(sample)
                    assert not isinstance(sample, Generator), "Generator not supported"
                    if isinstance(input_sample, PartialSample) and input_restore_key is not None:
                        sample = set_sample_restore_key(
                            sample,
                            *input_restore_key[1:],
                            src=input_sample,
                        )
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
            "select_next_pack": self._function_config(self.select_next_pack),
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
        return f"StreamingPackingDataset(select_next_pack={self.select_next_pack}, final_packer={self.final_packer}, dataset={self.dataset})"


class _StreamingSampleIterator(Generic[T_sample, T_slice]):
    def __init__(
        self,
        carryover_buffer: SavablePartialSampleBuffer[T_sample, T_slice],
        source_iter: Iterator[T_sample],
    ) -> None:
        self.carryover_buffer = carryover_buffer
        self.source_iter = source_iter
        self.source_exhausted = False

    def __iter__(self) -> "_StreamingSampleIterator[T_sample, T_slice]":
        return self

    def __next__(self) -> T_sample | PartialSample[T_sample, T_slice]:
        if self.carryover_buffer.len_worker() > 0:
            return self.carryover_buffer.pop(0)
        try:
            return next(self.source_iter)
        except StopIteration:
            self.source_exhausted = True
            raise
