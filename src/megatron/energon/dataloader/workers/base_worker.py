# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import dataclass
from typing import Generic, TypeVar

from megatron.energon.cache.base import CachePool
from megatron.energon.dataloader.future import DoneFuture, ExceptionFuture, Future
from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import SavableDataset, set_sample_restore_key
from megatron.energon.rng import SystemRng, SystemRngState
from megatron.energon.state import FlexState
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import WrappedRestoreKey, wrap_sample_restore_key

TSample = TypeVar("TSample", covariant=True)


@dataclass(kw_only=True, slots=True, frozen=True)
class WorkerSampleRestoreKey(WrappedRestoreKey):
    worker_id: int
    sample_idx: int


@edataclass
class WorkerState:
    """
    State of a worker.
    """

    rng: SystemRngState
    dataset: FlexState
    exhausted: bool
    sample_index: int

    def __str__(self):
        from hashlib import sha256

        rng_hash = sha256(str(self.rng).encode()).hexdigest()
        return f"WorkerState(dataset={self.dataset}, exhausted={self.exhausted}, sample_index={self.sample_index}, rng_hash={rng_hash})"


class DataLoaderWorker(Generic[TSample]):
    """
    A worker for a :class:`DataLoader`.

    The basic implementation iterates the dataset.
    The async extension implements the main commands via a command and results queue.
    """

    dataset: SavableDataset[TSample]
    worker_config: WorkerConfig

    _rank_worker_id: int
    _global_worker_id: int
    _seed: int
    _cache_pool: CachePool | None
    _sample_index: int = 0
    _exhausted: bool = True

    def __init__(
        self,
        dataset: SavableDataset[TSample],
        worker_config: WorkerConfig,
        rank_worker_id: int,
        cache_pool: CachePool | None,
    ):
        """
        Initialize the worker.

        Args:
            dataset: The dataset to iterate over.
            worker_config: The worker configuration.
            rank_worker_id: The rank of the worker.
            cache_pool: The cache pool to use.
        """
        self.dataset = dataset
        self.worker_config = worker_config
        self._rank_worker_id = rank_worker_id
        self._global_worker_id = worker_config.global_worker_id(rank_worker_id)
        self._seed = self.worker_config.worker_seed(rank_worker_id)
        self._cache_pool = cache_pool

    # ------------------------------------------------------------------------------------------------
    # Section: Main control methods

    def start(self) -> None:
        """
        Start the worker.
        """
        pass

    def shutdown(self, in_del: bool = False) -> None:
        """
        Shutdown the worker.

        Args:
            in_del: If True, the worker is being deleted.
        """
        self.dataset.worker_close()
        self.dataset.close()

    def running(self) -> bool:
        """
        Check if the worker is running.
        """
        return True

    def _assert_running(self) -> None:
        """
        Assert that the worker is running and alive.
        """
        assert self.running(), "Worker must be running"

    def __del__(self) -> None:
        self.shutdown(in_del=True)

    # ------------------------------------------------------------------------------------------------
    # Section: Worker methods

    def dataset_init(self, state: WorkerState | None) -> None:
        """
        Initialize the worker (may restore the state).
        Calls `new_iter` if the worker is not exhausted and also initially (`state=None`).

        Args:
            state: The state to restore the worker from or None for using the initial state.
        """
        # This is called in the worker context (process/thread).
        assert self._global_worker_id == self.worker_config.global_worker_id(), (
            "Global worker ID mismatch"
        )
        assert self._seed == self.worker_config.worker_seed(self._rank_worker_id), "Seed mismatch"
        print("dataset_init\n", end="")
        self.dataset.reset_state()
        if state is None:
            self._sample_index = 0
            print("dataset_init reset_state_deep\n", end="")
            self.new_iter()
            print("dataset_init new_iter\n", end="")
        else:
            print(f"dataset_init restore_state: {state=}\n", end="")
            self._sample_index = state.sample_index
            SystemRng.restore_state(state.rng)
            self.dataset.restore_state(state.dataset)
            if not state.exhausted:
                self.new_iter()
            assert self._exhausted == state.exhausted, "Exhausted state mismatch"

    def new_iter(self) -> None:
        """
        Start a new iterator of the dataset.
        Called after the dataset is initialized and to start a new epoch (if the dataset is not infinite).
        The iterator is stored in the worker and is used by the `prefetch_next` method, which calls `next` on it.
        Updates the exhausted flag to False.
        """
        # This is called in the worker context (process/thread).
        print("new_iter\n", end="")
        self._dataset_iter = iter(self.dataset)
        self._exhausted = False
        print("new_iter done\n", end="")

    def prefetch_next(self) -> Future[TSample]:
        """
        Fetch the next sample (i.e. call `next` on the iterator) and return a future for getting the result.
        Updates the exhausted flag if the iterator is exhausted.

        Returns:
            A future that will either be resolved to the next sample or raise StopIteration if the iterator is exhausted.
        """
        # This is called in the worker context (process/thread).
        assert self._dataset_iter is not None, "start_iter must be called before prefetch_next"
        if self._exhausted:
            try:
                raise StopIteration()
            except StopIteration as e:
                return ExceptionFuture(e)
        sample_idx = self._sample_index
        self.worker_config.worker_activate(sample_idx, cache_pool=self._cache_pool)
        try:
            next_sample = next(self._dataset_iter)
            self._sample_index += 1
            next_sample = wrap_sample_restore_key(
                next_sample,
                WorkerSampleRestoreKey,
                worker_id=self._global_worker_id,
                sample_idx=sample_idx,
            )
        except StopIteration as e:
            self._exhausted = True
            return ExceptionFuture(e)
        finally:
            self.worker_config.worker_deactivate()
        return DoneFuture(next_sample)

    def restore_sample(self, restore_key: WorkerSampleRestoreKey) -> Future[TSample]:
        """
        Restore a sample from a restore key in the worker.

        Args:
            restore_key: The restore key of the sample to restore.

        Returns:
            A future that will be resolved to the restored sample.
        """
        assert isinstance(restore_key, WorkerSampleRestoreKey)
        assert self._global_worker_id == restore_key.worker_id, "Global worker ID mismatch"
        self.worker_config.worker_activate(
            restore_key.sample_idx,
            cache_pool=self._cache_pool,
        )
        try:
            return DoneFuture(
                set_sample_restore_key(self.dataset.restore_sample(restore_key.inner), restore_key)
            )
        finally:
            self.worker_config.worker_deactivate()

    def save_state(self) -> WorkerState:
        """
        Save the state of the worker.
        """
        # This is called in the worker context (process/thread).
        print(f"save_state: {self._sample_index=}, {self._exhausted=}\n", end="")
        return WorkerState(
            rng=SystemRng.save_state(),
            dataset=self.dataset.save_state(),
            exhausted=self._exhausted,
            sample_index=self._sample_index,
        )


class DataLoaderNoWorker(DataLoaderWorker[TSample], Generic[TSample]):
    """
    DataLoader without async worker.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
