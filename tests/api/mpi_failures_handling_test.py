"""MPI integration tests for failure handling and soft-fail recovery (requires mpiexec -n 2)."""

import gc
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ocsmesh import Raster
from ocsmesh.hfun.collector import MPITaskRunner
from ocsmesh.hfun.raster import HfunRaster
from ocsmesh.utils import raster_from_numpy

try:
    from mpi4py import MPI

    IS_UNDER_MPIEXEC = MPI.COMM_WORLD.Get_size() > 1
except ImportError:
    IS_UNDER_MPIEXEC = False


def _create_test_rasters(base_dir):
    """Create synthetic DEM rasters for testing."""
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    dem_data = (grid_x * 20) - 10

    dem1_path = base_dir / "dem1.tif"
    dem2_path = base_dir / "dem2.tif"
    raster_from_numpy(dem1_path, dem_data, (grid_x, grid_y), 4326)
    raster_from_numpy(dem2_path, dem_data.copy(), (grid_x, grid_y), 4326)

    return [Raster(dem1_path), Raster(dem2_path)]


@unittest.skipUnless(IS_UNDER_MPIEXEC, "Requires mpiexec with >1 rank")
class TestMPIFailuresHandling(unittest.TestCase):
    """Worker soft-fail and recovery tests — require exactly 2 MPI ranks (-n 2).

    With exactly 1 worker (n=2), dispatching [bad_task, good_task] in a
    single runner.dispatch() call forces the same worker to handle both.
    If the good task succeeds, the worker survived the failure.

    If executed with n != 2 under mpiexec, setUp() fails explicitly with a
    misconfiguration error message to avoid false confidence from skipping.
    """

    def setUp(self):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()

        if comm.Get_size() != 2:
            self.fail(
                f"[MISCONFIGURATION] TestMPIFailuresHandling requires "
                f"exactly 2 MPI ranks (1 manager + 1 worker via '-n 2'), "
                f"but was executed with {comm.Get_size()} ranks. "
                f"Please run with: mpiexec -n 2 python -m pytest "
                f"tests/api/test_mpi_failures_handling.py"
            )

        if self.rank == 0:
            self.tdir = Path(tempfile.mkdtemp())
            self.raster_list = _create_test_rasters(self.tdir)
        else:
            self.tdir = None
            self.raster_list = None

        self.tdir = comm.bcast(self.tdir, root=0)

    def tearDown(self):
        comm = MPI.COMM_WORLD
        comm.Barrier()
        self.raster_list = None
        gc.collect()
        if self.rank == 0:
            try:
                shutil.rmtree(self.tdir)
            except (PermissionError, FileNotFoundError):
                pass

    def test_task_exception_recovery(self):
        """Worker survives a task exception and processes the next task."""
        runner = MPITaskRunner()
        out_path = str(self.tdir / "recovery_test")

        def main():
            raster = self.raster_list[0]

            # ── BAD TASK ────────────────────────────────────────────
            # Definition: Uses a valid/registered operation ('op': 'meshdata'),
            # but passes non-existent file paths.
            bad_task = {
                'op': 'meshdata',
                'type': 'raster',
                'original_index': 0,
                'topo_path': '/non/existent/path/dem.tif',
                'hfun_input_path': '/non/existent/path/tmp.hfun',
                'output_path': '/non/existent/path/out',
                'hmin': 10,
                'hmax': 1000,
                'meshdata_kwargs': {}
            }

            valid_hfun = HfunRaster(
                raster=raster, hmin=10, hmax=1000
            )
            good_task = {
                'op': 'meshdata',
                'type': 'raster',
                'original_index': 1,
                'topo_path': raster.path,
                'hfun_input_path': valid_hfun.tmpfile,
                'output_path': out_path,
                'hmin': 10,
                'hmax': 1000,
                'meshdata_kwargs': {}
            }

            # Single dispatch — 1 worker handles both sequentially
            results = runner.dispatch([bad_task, good_task])

            self.assertEqual(len(results), 2)

            errors = [
                r for r in results if r.get('status') == 'error'
            ]
            successes = [
                r for r in results if r.get('status') != 'error'
            ]

            # Bad task returned a structured error
            self.assertEqual(len(errors), 1)
            self.assertIn('error', errors[0])
            self.assertIn('worker_rank', errors[0])
            self.assertEqual(errors[0]['original_index'], 0)

            # Good task succeeded — same worker was still alive
            self.assertEqual(len(successes), 1)
            self.assertEqual(successes[0]['original_index'], 1)

        runner.run(main)
        MPI.COMM_WORLD.Barrier()

    def test_unregistered_op_recovery(self):
        """Worker survives an unregistered op and processes the next task."""
        runner = MPITaskRunner()
        out_path = str(self.tdir / "unreg_recovery_test")

        def main():
            raster = self.raster_list[0]

            # ── INVALID TASK ────────────────────────────────────────
            # Definition: Uses an operation name ('op': 'nonexistent_operation')
            # that is NOT registered in MPITaskRunner._worker_registry().
            invalid_task = {
                'op': 'nonexistent_operation',
                'original_index': 0,
            }

            valid_hfun = HfunRaster(
                raster=raster, hmin=10, hmax=1000
            )
            good_task = {
                'op': 'meshdata',
                'type': 'raster',
                'original_index': 1,
                'topo_path': raster.path,
                'hfun_input_path': valid_hfun.tmpfile,
                'output_path': out_path,
                'hmin': 10,
                'hmax': 1000,
                'meshdata_kwargs': {}
            }

            results = runner.dispatch([invalid_task, good_task])

            self.assertEqual(len(results), 2)

            errors = [
                r for r in results if r.get('status') == 'error'
            ]
            successes = [
                r for r in results if r.get('status') != 'error'
            ]

            self.assertEqual(len(errors), 1)
            self.assertIn(
                "No worker registered for op",
                errors[0]['error']
            )

            self.assertEqual(len(successes), 1)
            self.assertEqual(successes[0]['original_index'], 1)

        runner.run(main)
        MPI.COMM_WORLD.Barrier()


if __name__ == "__main__":
    unittest.main()
