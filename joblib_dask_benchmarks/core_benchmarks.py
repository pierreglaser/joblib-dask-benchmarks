import time
import threading

from time import sleep

import numpy as np

from distributed import Client, LocalCluster

from joblib import Parallel, delayed, parallel_backend

from .base import BenchmarkBase


# Any function created inside this folder will not be pickleable unless
# ASV_PYTHONPATH/PYTHONPATH is set correctly, see the README for informations.


def parallel_sum_on_slices(array, backend, release_gil=False):
    slices = [array[i:] for i in range(1)]
    with parallel_backend(backend):
        res = Parallel()(delayed(np.sum)(s) for s in slices)
    return res


lock = threading.Lock()


def simulate_computation(*args, release_gil=False):
    TASK_LENGTH = 0.5
    if release_gil:
        time.sleep(TASK_LENGTH)
    else:
        with lock:
            # any other thread running in the same process will need to wait.
            time.sleep(TASK_LENGTH)
    return


def sleep_and_return(x, sleep_time=1):
    sleep(sleep_time)
    return x


class TimeCoreCrossBackendBenchmarks(BenchmarkBase):
    """Benchmarks of core operations with backend-agnostic parameters

    The benchamrks
    - generally only manipulate numpy arrays
    - are not parametrized using backend-specific options.
    """

    def setup(self, backend, n_workers, threads_per_worker):
        self._setup_backend(backend, n_workers, threads_per_worker)
        self.large_array = (
            np.ones(int(1e8)).astype(np.int8).reshape(int(1e7), 10)
        )

    def time_simple_sleep(self, backend, n_workers, threads_per_worker):
        with parallel_backend(**self.backend_kwargs):
            _ = Parallel()(delayed(sleep)(0.5) for i in range(1))

    def time_heavy_computation_with_no_data_transfer(
        self, backend, n_workers, threads_per_worker
    ):
        # this should generate a maximum speedup, as there is no data transfer.
        with parallel_backend(**self.backend_kwargs):
            _ = Parallel()(
                delayed(simulate_computation)(release_gil=False)
                for _ in range(10)
            )

    def time_run_many_small_tasks(
        self, backend, n_workers, threads_per_worker
    ):
        # serves as a non-regression benchmark: a bug affecting the
        # DaskDistribuedBackend was preventing it from using auto-batching,
        # yielding low performances when submitting a large number of tasks.
        with parallel_backend(**self.backend_kwargs):
            _ = Parallel(verbose=10)(delayed(id)(i) for i in range(100000))

    def time_many_tasks_operating_on_same_data(
        self, backend, n_workers, threads_per_worker
    ):
        # We notice that dask is way slower than loky for this benchmark: this
        # is due to the costly scatter() call.
        # We also notice that auto-scatter provide huge performance
        # improvements as it prevents sending many time large numpy arrays to
        # the dask scheduler.
        with parallel_backend(**self.backend_kwargs):
            _ = Parallel(verbose=10000)(
                delayed(id)(self.large_array) for _ in range(300)
            )

    def time_many_tasks_operating_on_slices_of_same_data(
        self, backend, n_workers, threads_per_worker
    ):
        # In this situation, large_array will be scattered/memmaped many times,
        # which is a waste of computations.
        # The performance of this benchmarks ought to be compared with
        # smart-indexing situations where large_array will be only
        # scattered/memmaped once
        with parallel_backend(**self.backend_kwargs):
            _ = Parallel(verbose=1000)(
                delayed(id)(self.large_array[i:]) for i in range(10)
            )

    def time_nested_calls_with_same_data_transfer_in_each_level(
        self, backend, n_workers, threads_per_worker
    ):
        # This is more of a non-regression test than a benchmark:
        # nested-scattering is partly broken in dask, see dask/distributed#3703
        with parallel_backend(**self.backend_kwargs):
            _ = Parallel()(
                delayed(parallel_sum_on_slices)(self.large_array, backend)
                for _ in range(8)
            )

    def time_slow_input_producer(self, backend, n_workers, threads_per_worker):
        # Simulate the situation where the input generator given to Parallel
        # takes time to produce new inputs. This can happen if this generator
        # is ia LazyLoader reading large data from disk.
        def slow_input_producer():
            for i in range(10):
                time.sleep(1)
                yield np.arange(i, i + 10000).astype(np.int8)

        with parallel_backend(**self.backend_kwargs):
            _ = Parallel()(
                delayed(sleep_and_return)(x) for x in slow_input_producer()
            )
