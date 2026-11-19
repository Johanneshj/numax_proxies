"""Numba thread control."""
from contextlib import contextmanager

import numba


@contextmanager
def numba_threads(n_threads: int | None):
    """Context manager: use n_threads Numba threads inside the block.

    Args:
        n_threads: number of threads (capped at NUMBA_NUM_THREADS); None = unchanged.
    """
    if n_threads is None:
        yield
        return
    previous = numba.get_num_threads()
    numba.set_num_threads(max(1, min(int(n_threads), numba.config.NUMBA_NUM_THREADS)))
    try:
        yield
    finally:
        numba.set_num_threads(previous)
