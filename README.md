Repository containing the benchmarks of `joblib` when relying on the `dask.distributed` package

## How to run the benchmarks

### Using an existing environment

To benchmark your locally installed `joblib` in an existing environment, set ``$PYTHONPATH`` to ``/path/to/joblib-dask-benchmarks``
Example command:
``PYTHONPATH=/home/pierreglaser/repos/joblib-dask-benchmarks asv run -b nested_scatter_inputs --environment=existing:python``
This will run benchmarks containing `nested_scatter_inputs` in their title using the currently activated `Python` virtual environment.

### Using environments specified in `asv.conf.json`
To benchmark `joblib` against configurations specified in `asv.conf.json` (for now: only one configuration), you need to take in account the isolated run feature of asv by setting `$ASV_PYTHONPATH` to `/path/to/joblib/benchmarks`
Example command:
``ASV_PYTHONPATH=/home/pierreglaser/repos/joblib-dask-benchmarks asv run -b nested_scatter_inputs --environment=existing:python``
