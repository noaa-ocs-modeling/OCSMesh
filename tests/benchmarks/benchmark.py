"""
Benchmarks for the meshdata() pipeline.

NOTE: Since this file is named `benchmark.py` (and does not start with `test_`),
pytest won't discover it automatically if you just run `pytest tests/`.
It will only run if you specifically pass the file path: `pytest tests/benchmarks/benchmark.py`.
This is ideal for benchmarks so they don't slow down normal unit test runs.

Uses ``pytest-benchmark`` to measure the full ``hfun.meshdata()``
pipeline under different execution configurations. Results are
parsed by the CI workflow and posted as PR comments.

Run locally::

    pytest tests/benchmarks/benchmark.py --benchmark-only -v

Generate JSON for CI::

    pytest tests/benchmarks/benchmark.py --benchmark-only --benchmark-json output.json
"""

import numpy as np
import numpy.testing as npt
import pytest

from ocsmesh import Hfun


# ─── Benchmark Configuration ────────────────────────────────────────
# Edit rounds here to control how many times each mode is measured.
# Higher rounds = more stable statistics but slower CI.
BENCHMARK_ROUNDS = 3

# Dictionary to cache the first result to assert numerical equivalence
_CACHED_RESULTS = {}


# ─── Test Case Definitions ──────────────────────────────────────────
BENCHMARK_CASES = ["serial", "parallel"]


# ─── Helpers ─────────────────────────────────────────────────────────

def _build_hfun(raster_list, execution_mode):
    """Create an Hfun with 3 refinements and the given execution_mode."""

    hfun = Hfun(raster_list, nprocs=4, hmin=10, hmax=1000)
    hfun.execution_mode = execution_mode
    hfun.add_topo_bound_constraint(
        value=100, upper_bound=5, lower_bound=-5, value_type='min')
    hfun.add_topo_bound_constraint(
        value=200, upper_bound=8, lower_bound=2, value_type='max')
    hfun.add_constant_value(value=500, lower_bound=-10, upper_bound=-5)
    return hfun


# ─── Benchmarks ──────────────────────────────────────────────────────

@pytest.mark.benchmark(group="meshdata_pipeline")
@pytest.mark.parametrize("exec_mode", BENCHMARK_CASES)
def test_meshdata_pipeline(benchmark, benchmark_raster_list, exec_mode):
    """Benchmark hfun.meshdata() under a specific execution configuration.

    The ``benchmark_raster_list`` fixture (from conftest.py) provides
    4 tiled rasters with 20% overlap, created once per module.
    """

    def run_meshdata():
        hfun = _build_hfun(benchmark_raster_list, exec_mode)
        return hfun.meshdata()

    meshdata = benchmark.pedantic(
        run_meshdata,
        rounds=BENCHMARK_ROUNDS,
    )

    # Sanity checks
    assert meshdata is not None
    assert len(meshdata.coords) > 0
    assert np.all(np.isfinite(meshdata.values))

    # Equivalence checks across modes
    mean_val = float(np.mean(meshdata.values))
    
    if not _CACHED_RESULTS:
        _CACHED_RESULTS["mean_val"] = mean_val
    else:
        npt.assert_allclose(
            _CACHED_RESULTS["mean_val"], 
            mean_val, 
            rtol=1e-5,
            err_msg=f"Mean value mismatch in {exec_mode} mode"
        )
