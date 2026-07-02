"""
Benchmarks for the meshdata() pipeline.

NOTE: Since this file is named `benchmark.py` (and does not start with `test_`),
pytest won't discover it automatically if you just run `pytest tests/`.

Uses ``pytest-benchmark`` to measure the full ``hfun.meshdata()``
pipeline under different execution configurations. Results are
parsed by the CI workflow and posted as PR comments.

Run locally::

    pytest tests/benchmarks/benchmark.py --benchmark-only -v

Generate JSON for CI::

    pytest tests/benchmarks/benchmark.py --benchmark-only --benchmark-json output.json
"""

import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from ocsmesh import Hfun


# ─── Benchmark Configuration ────────────────────────────────────────
# Edit rounds here to control how many times each mode is measured.
BENCHMARK_ROUNDS = 3

# Relative tolerance for the numerical equivalence check
EQUIVALENCE_RTOL = 1e-5

# Dictionary to cache the first result to assert numerical equivalence
_CACHED_RESULTS = {}

# Path to write equivalence check results for CI reporting
_EQUIVALENCE_OUTPUT = Path("equivalence_result.json")


# ─── Test Case Definitions ──────────────────────────────────────────
BENCHMARK_CASES = ["serial", "parallel"]


# ─── Helpers ─────────────────────────────────────────────────────────
# TODO: this can be extended to cover more refinement scenarios.

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


def _write_equivalence_result(status, first_mode, second_mode, checks, rtol):
    """Write equivalence check outcome to a JSON file for CI reporting.

    Parameters
    ----------
    status : str
        'pass' or 'fail'.
    first_mode, second_mode : str
        The execution modes being compared.
    checks : dict
        Per-metric comparison results, e.g.
        {"num_nodes": {"first": 100, "second": 100, "match": True}, ...}
    rtol : float
        Relative tolerance used for floating-point comparisons.
    """

    result = {
        "status": status,
        "first_mode": first_mode,
        "second_mode": second_mode,
        "rtol": rtol,
        "checks": checks,
    }
    _EQUIVALENCE_OUTPUT.write_text(json.dumps(result, indent=2))


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

    # Cache results from the first mode for cross-mode comparison
    stats = {
        "num_nodes": len(meshdata.coords),
        "min": float(np.min(meshdata.values)),
        "max": float(np.max(meshdata.values)),
        "mean": float(np.mean(meshdata.values)),
        "coords": meshdata.coords,
    }

    if not _CACHED_RESULTS:
        _CACHED_RESULTS["mode"] = exec_mode
        _CACHED_RESULTS["stats"] = stats
    else:
        cached = _CACHED_RESULTS["stats"]

        # Build per-metric check results for CI reporting
        checks = {
            "num_nodes": {
                "first": cached["num_nodes"],
                "second": stats["num_nodes"],
                "match": cached["num_nodes"] == stats["num_nodes"],
            },
            "min": {
                "first": cached["min"],
                "second": stats["min"],
                "rel_diff": abs(cached["min"] - stats["min"]) / abs(cached["min"]) if cached["min"] != 0 else 0.0,
            },
            "max": {
                "first": cached["max"],
                "second": stats["max"],
                "rel_diff": abs(cached["max"] - stats["max"]) / abs(cached["max"]) if cached["max"] != 0 else 0.0,
            },
            "mean": {
                "first": cached["mean"],
                "second": stats["mean"],
                "rel_diff": abs(cached["mean"] - stats["mean"]) / abs(cached["mean"]) if cached["mean"] != 0 else 0.0,
            },
        }

        _write_equivalence_result(
            status="pass",
            first_mode=_CACHED_RESULTS["mode"],
            second_mode=exec_mode,
            checks=checks,
            rtol=EQUIVALENCE_RTOL,
        )

        # Node count must match exactly
        assert cached["num_nodes"] == stats["num_nodes"], (
            f"Node count mismatch: {cached['num_nodes']} vs {stats['num_nodes']}"
        )

        # Coordinates must match within tolerance
        npt.assert_allclose(
            cached["coords"], stats["coords"],
            rtol=EQUIVALENCE_RTOL,
            err_msg=f"Coordinates mismatch in {exec_mode} mode",
        )

        # Value statistics must match within tolerance
        npt.assert_allclose(
            cached["min"], stats["min"],
            rtol=EQUIVALENCE_RTOL,
            err_msg=f"Min value mismatch in {exec_mode} mode",
        )
        npt.assert_allclose(
            cached["max"], stats["max"],
            rtol=EQUIVALENCE_RTOL,
            err_msg=f"Max value mismatch in {exec_mode} mode",
        )
        npt.assert_allclose(
            cached["mean"], stats["mean"],
            rtol=EQUIVALENCE_RTOL,
            err_msg=f"Mean value mismatch in {exec_mode} mode",
        )

