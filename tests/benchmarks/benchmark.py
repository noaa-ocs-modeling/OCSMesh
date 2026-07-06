"""
Benchmarks for the meshdata() pipeline.

NOTE: Since this file is named `benchmark.py` (and does not start with `test_`),
pytest won't discover it automatically if you just run `pytest tests/`.

Uses ``pytest-benchmark`` to measure the full ``hfun.meshdata()``
pipeline under different execution configurations. Results are
parsed by the CI workflow and posted as PR comments.

Run locally::

    pytest tests/benchmarks/benchmark.py -v

Generate JSON for CI::

    pytest tests/benchmarks/benchmark.py --benchmark-json output.json
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
# TODO: Make this configurable via pytest command line option
# Path to write equivalence check results for CI reporting
_EQUIVALENCE_OUTPUT = Path("equivalence_result.json")


# ─── Test Case Definitions ──────────────────────────────────────────
# The first entry is used as the
# baseline for all equivalence comparisons.
BENCHMARK_CASES = ["serial", "parallel"]


# ─── Collected Results ──────────────────────────────────────────────
_RESULTS = {}


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


def _extract_stats(meshdata):
    """Extract comparable statistics from a meshdata result."""

    return {
        "num_nodes": len(meshdata.coords),
        "min": float(np.min(meshdata.values)),
        "max": float(np.max(meshdata.values)),
        "mean": float(np.mean(meshdata.values)),
        "coords": meshdata.coords,
    }


def _rel_diff(a, b):
    """Compute the relative difference between two scalars."""

    return abs(a - b) / abs(a) if a != 0 else 0.0


def _compare_stats(baseline, current):
    """Compare two stat dicts and return a per-metric check summary.

    Parameters
    ----------
    baseline, current : dict
        Output of ``_extract_stats``.

    Returns
    -------
    dict
        Per-metric check results suitable for JSON serialization.
    """

    return {
        "num_nodes": {
            "baseline": baseline["num_nodes"],
            "current": current["num_nodes"],
            "match": baseline["num_nodes"] == current["num_nodes"],
        },
        "min": {
            "baseline": baseline["min"],
            "current": current["min"],
            "rel_diff": _rel_diff(baseline["min"], current["min"]),
        },
        "max": {
            "baseline": baseline["max"],
            "current": current["max"],
            "rel_diff": _rel_diff(baseline["max"], current["max"]),
        },
        "mean": {
            "baseline": baseline["mean"],
            "current": current["mean"],
            "rel_diff": _rel_diff(baseline["mean"], current["mean"]),
        },
    }


def _assert_stats_equal(baseline, current, mode_label, rtol):
    """Assert that two stat dicts are numerically equivalent."""

    assert baseline["num_nodes"] == current["num_nodes"], (
        f"[{mode_label}] Node count mismatch: "
        f"{baseline['num_nodes']} vs {current['num_nodes']}"
    )

    npt.assert_allclose(
        baseline["coords"], current["coords"],
        rtol=rtol,
        err_msg=f"[{mode_label}] Coordinates mismatch",
    )

    for metric in ("min", "max", "mean"):
        npt.assert_allclose(
            baseline[metric], current[metric],
            rtol=rtol,
            err_msg=f"[{mode_label}] {metric} value mismatch",
        )


# ─── Benchmarks ──────────────────────────────────────────────────────

@pytest.mark.benchmark(group="meshdata_pipeline")
@pytest.mark.parametrize("exec_mode", BENCHMARK_CASES)
def test_meshdata_pipeline(benchmark, benchmark_raster_list, exec_mode):
    """Benchmark hfun.meshdata() under a specific execution configuration.

    The ``benchmark_raster_list`` fixture (from conftest.py) provides
    4 tiled rasters with 20% overlap, created once per module.

    Results are stored in ``_RESULTS`` for the equivalence test below.
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

    # Store for cross-mode comparison
    _RESULTS[exec_mode] = _extract_stats(meshdata)


# ─── Equivalence ─────────────────────────────────────────────────────

def test_numerical_equivalence():
    """Verify that all execution modes produce identical meshdata.

    Runs after all parametrized benchmarks. Uses the first entry in
    ``BENCHMARK_CASES`` as the baseline and compares every other mode
    against it. Writes a detailed JSON report for the CI workflow.
    """

    assert len(_RESULTS) == len(BENCHMARK_CASES), (
        f"Expected results for {BENCHMARK_CASES}, "
        f"got results for {list(_RESULTS.keys())}"
    )

    baseline_mode = BENCHMARK_CASES[0]
    baseline = _RESULTS[baseline_mode]

    comparisons = {}
    for mode in BENCHMARK_CASES[1:]:
        current = _RESULTS[mode]
        comparisons[mode] = _compare_stats(baseline, current)
        _assert_stats_equal(baseline, current, mode, EQUIVALENCE_RTOL)

    # Write report for CI
    report = {
        "status": "pass",
        "baseline": baseline_mode,
        "rtol": EQUIVALENCE_RTOL,
        "comparisons": comparisons,
    }
    _EQUIVALENCE_OUTPUT.write_text(json.dumps(report, indent=2))
