from megatron.energon.sync_end.redistribute import RedistributeLoader, RedistributeDataLoaderState
from megatron.energon.sync_end.stop_first_end import StopFirstLoader, StopFirstDataLoaderState

"""
Provides wrappers for the dataset loaders that allow for synchronization at the end of the dataset.
I.e. if running a training with repeat=False, the loaders will typically exhaust at different times, which may require
synchronization across ranks.

The wrappers are:
- RedistributeLoader: Redistributes the last samples to the ranks that are not exhausted.
- StopFirstLoader: Stops iterating as soon as the first rank is exhausted.
"""

__all__ = ["RedistributeLoader", "RedistributeDataLoaderState", "StopFirstLoader", "StopFirstDataLoaderState"]