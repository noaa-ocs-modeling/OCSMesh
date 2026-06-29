"""
Profiling test for _calculate_and_write_hfun_to_disk().

Runs the full meshdata() pipeline under 4 configurations to isolate
the impact of parallelizing _calculate_and_write_hfun_to_disk:

  Mode 1 — ALL PARALLEL:
      apply_constraints = parallel, _calculate_and_write_hfun_to_disk = parallel

  Mode 2 — ALL SERIAL:
      apply_constraints = serial,   _calculate_and_write_hfun_to_disk = serial

  Mode 3 — PARALLEL EXCEPT WRITE HFUN:
      apply_constraints = parallel, _calculate_and_write_hfun_to_disk = serial  (monkey-patched)

  Mode 4 — SERIAL EXCEPT WRITE HFUN:
      apply_constraints = serial,   _calculate_and_write_hfun_to_disk = parallel (monkey-patched)

By comparing Mode 1 vs Mode 3, you see the speedup from parallelizing
_calculate_and_write_hfun_to_disk alone (everything else is the same).

By comparing Mode 2 vs Mode 4, you see the same thing but with the
rest of the pipeline serial.
"""

import unittest
import tempfile
import platform
import shutil
import gc
import time as time_mod
import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from ocsmesh import Hfun, Raster
from ocsmesh.hfun.collector import HfunCollector
from ocsmesh.utils import raster_from_numpy


IS_WINDOWS = platform.system() == 'Windows'
NUM_ITERATIONS = 3  # Number of times to run each mode for averaging


class TestCalculateAndWriteHfunProfiling(unittest.TestCase):
    """Profile _calculate_and_write_hfun_to_disk under 4 execution configurations."""

    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())

        grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
        dem_data = (grid_x * 20) - 10  # Values from -10 to 10

        self.dem_paths = []
        self.raster_list = []
        for i in range(4):
            p = self.tdir / f'dem_{i}.tif'
            raster_from_numpy(p, dem_data.copy(), (grid_x, grid_y), 4326)
            self.dem_paths.append(p)
            self.raster_list.append(Raster(p))

    def tearDown(self):
        self.raster_list = None
        gc.collect()
        try:
            shutil.rmtree(self.tdir)
        except PermissionError:
            pass

    def _build_hfun(self, execution_mode):
        """Create an Hfun with 3 refinements. Sets execution_mode."""
        hfun = Hfun(self.raster_list, nprocs=4, hmin=10, hmax=1000)
        hfun.execution_mode = execution_mode
        hfun.add_topo_bound_constraint(
            value=100, upper_bound=5, lower_bound=-5, value_type='min')
        hfun.add_topo_bound_constraint(
            value=200, upper_bound=8, lower_bound=2, value_type='max')
        hfun.add_constant_value(value=500, lower_bound=-10, upper_bound=-5)
        return hfun

    def _run_mode(self, label, execution_mode, calculate_and_write_hfun_override=None):
        """
        Run meshdata() under a specific configuration.

        Parameters
        ----------
        label : str
            Human-readable mode name.
        execution_mode : str
            'serial' or 'parallel' — controls apply_constraints etc.
        calculate_and_write_hfun_override : str or None
            If 'serial', monkey-patch _calculate_and_write_hfun_to_disk → serial.
            If 'parallel', monkey-patch _calculate_and_write_hfun_to_disk → parallel.
            If None, use whatever execution_mode dictates.

        Returns
        -------
        dict with elapsed time, mean value, and label.
        """
        elapsed_times = []
        meshdata = None
        for i in range(NUM_ITERATIONS):
            hfun = self._build_hfun(execution_mode)

            if calculate_and_write_hfun_override == 'serial':
                ctx = patch.object(
                    HfunCollector, '_calculate_and_write_hfun_to_disk',
                    HfunCollector._calculate_and_write_hfun_to_disk_serial)
            elif calculate_and_write_hfun_override == 'parallel':
                ctx = patch.object(
                    HfunCollector, '_calculate_and_write_hfun_to_disk',
                    HfunCollector._calculate_and_write_hfun_to_disk_parallel)
            else:
                # No patching — use the default dispatcher
                from contextlib import nullcontext
                ctx = nullcontext()

            with ctx:
                t0 = time_mod.perf_counter()
                meshdata = hfun.meshdata()
                elapsed_times.append(time_mod.perf_counter() - t0)

        avg_elapsed = sum(elapsed_times) / len(elapsed_times)

        return {
            'label': label,
            'elapsed': avg_elapsed,
            'mean_val': float(np.mean(meshdata.values)),
            'meshdata': meshdata,
        }

    # TODO: Check if windows won't cause issues.
    # @unittest.skipIf(IS_WINDOWS, 'Profiling not supported on Windows')
    def test_calculate_and_write_hfun_profiling(self):
        """Run all 4 modes, print progress, write a .txt report."""

        n_rasters = len(self.raster_list)
        nprocs = 4
        report_path = Path.cwd() / 'timing_report_calculate_and_write_hfun_profiling.txt'

        # ─── Mode definitions ───────────────────────────────────────
        #   (label, execution_mode, calculate_and_write_hfun_override)
        modes = [
            {
                'label':     'MODE 1: ALL PARALLEL',
                'exec_mode': 'parallel',
                'override':  None,
                'details': (
                    '  apply_constraints    : PARALLEL\n'
                    '  apply_features       : PARALLEL\n'
                    '  apply_contours       : PARALLEL\n'
                    '  _calculate_and_write_hfun_to_disk  : PARALLEL\n'
                ),
            },
            {
                'label':     'MODE 2: ALL SERIAL',
                'exec_mode': 'serial',
                'override':  None,
                'details': (
                    '  apply_constraints    : SERIAL\n'
                    '  apply_features       : SERIAL\n'
                    '  apply_contours       : SERIAL\n'
                    '  _calculate_and_write_hfun_to_disk  : SERIAL\n'
                ),
            },
            {
                'label':     'MODE 3: PARALLEL EXCEPT _calculate_and_write_hfun_to_disk',
                'exec_mode': 'parallel',
                'override':  'serial',
                'details': (
                    '  apply_constraints    : PARALLEL\n'
                    '  apply_features       : PARALLEL\n'
                    '  apply_contours       : PARALLEL\n'
                    '  _calculate_and_write_hfun_to_disk  : SERIAL  (monkey-patched)\n'
                ),
            },
            {
                'label':     'MODE 4: SERIAL EXCEPT _calculate_and_write_hfun_to_disk',
                'exec_mode': 'serial',
                'override':  'parallel',
                'details': (
                    '  apply_constraints    : SERIAL\n'
                    '  apply_features       : SERIAL\n'
                    '  apply_contours       : SERIAL\n'
                    '  _calculate_and_write_hfun_to_disk  : PARALLEL (monkey-patched)\n'
                ),
            },
        ]

        # ─── Run all modes ──────────────────────────────────────────
        results = []
        total = len(modes)

        for i, mode in enumerate(modes):
            pct = int(i / total * 100)
            print(f"\n[{pct:3d}%] Running {mode['label']}...")

            result = self._run_mode(
                mode['label'], mode['exec_mode'], mode['override'])
            results.append(result)

            print(f"       Done in {result['elapsed']:.2f}s (avg of {NUM_ITERATIONS})  "
                  f"(mean={result['mean_val']:.4f})")

        print(f"[100%] All modes complete.\n")

        # ─── Numerical sanity check ─────────────────────────────────
        # All 4 modes must produce the same mean value
        for r in results[1:]:
            npt.assert_allclose(
                results[0]['mean_val'], r['mean_val'], rtol=1e-5,
                err_msg=f"{results[0]['label']} vs {r['label']}")

        # ─── Build report ───────────────────────────────────────────
        lines = []
        lines.append('=' * 65)
        lines.append('  _calculate_and_write_hfun_to_disk() Profiling Report')
        lines.append('=' * 65)
        lines.append(f'Date            : {datetime.datetime.now().isoformat()}')
        lines.append(f'Rasters         : {n_rasters}')
        lines.append(f'nprocs          : {nprocs}')
        lines.append(f'Grid size       : 100 x 100 per DEM')
        lines.append(f'Iterations      : {NUM_ITERATIONS}')
        lines.append(f'Refinements     : 3 (topo_bound_min, topo_bound_max, constant_value)')
        lines.append('')

        for i, (mode, result) in enumerate(zip(modes, results)):
            lines.append('-' * 65)
            lines.append(f'  {mode["label"]}')
            lines.append('-' * 65)
            lines.append(mode['details'].rstrip())
            lines.append(f'  Time  : {result["elapsed"]:.4f} s')
            lines.append(f'  Mean  : {result["mean_val"]:.6f}')
            lines.append('')

        # ─── Comparison table ───────────────────────────────────────
        lines.append('=' * 65)
        lines.append('  COMPARISON')
        lines.append('=' * 65)
        lines.append(f'  {"Mode":<45} {"Time":>8}')
        lines.append(f'  {"-"*45} {"-"*8}')
        for result in results:
            lines.append(f'  {result["label"]:<45} {result["elapsed"]:>7.2f}s')
        lines.append('')

        # Speedup: Mode 1 vs Mode 3 (isolates _calculate_and_write_hfun_to_disk parallel)
        if results[2]['elapsed'] > 0:
            speedup_write = results[2]['elapsed'] / results[0]['elapsed']
            lines.append(f'  Speedup from parallel _calculate_and_write_hfun_to_disk:')
            lines.append(f'    Mode 3 / Mode 1 = {speedup_write:.2f}x '
                         f'({results[2]["elapsed"]:.2f}s → {results[0]["elapsed"]:.2f}s)')

        # Speedup: Mode 2 vs Mode 4 (same isolation, serial baseline)
        if results[1]['elapsed'] > 0:
            speedup_write2 = results[1]['elapsed'] / results[3]['elapsed']
            lines.append(f'    Mode 2 / Mode 4 = {speedup_write2:.2f}x '
                         f'({results[1]["elapsed"]:.2f}s → {results[3]["elapsed"]:.2f}s)')

        # Overall speedup: Mode 2 vs Mode 1
        if results[1]['elapsed'] > 0:
            speedup_total = results[1]['elapsed'] / results[0]['elapsed']
            lines.append(f'  Overall speedup (all serial → all parallel):')
            lines.append(f'    Mode 2 / Mode 1 = {speedup_total:.2f}x '
                         f'({results[1]["elapsed"]:.2f}s → {results[0]["elapsed"]:.2f}s)')

        lines.append('')
        lines.append(f'  Values match  : YES')
        lines.append('=' * 65)

        report_text = '\n'.join(lines) + '\n'

        with open(report_path, 'w') as f:
            f.write(report_text)

        self.assertTrue(report_path.exists())
        print(f"--- Report saved to: {report_path} ---\n")
        print(report_text)


if __name__ == '__main__':
    unittest.main()
