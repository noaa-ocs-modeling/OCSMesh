"""MPI integration tests for Hfun write path (requires mpiexec)."""

# pylint: disable=c-extension-no-member

import gc
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

from ocsmesh import Hfun, Raster
from ocsmesh.mpi import MPIExecutor
from ocsmesh.hfun.raster import HfunRaster
from ocsmesh.utils import raster_from_numpy

from ocsmesh.mpi import _is_mpi_env_detected

IS_UNDER_MPIEXEC = _is_mpi_env_detected()
if IS_UNDER_MPIEXEC:
    try:
        from mpi4py import MPI
    except ImportError:
        IS_UNDER_MPIEXEC = False


def _create_test_rasters(base_dir):
    """Create two synthetic DEM rasters for testing."""
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    dem_data = (grid_x * 20) - 10  # Values from -10 to 10

    dem1_path = base_dir / "dem1.tif"
    dem2_path = base_dir / "dem2.tif"
    raster_from_numpy(dem1_path, dem_data, (grid_x, grid_y), 4326)
    raster_from_numpy(dem2_path, dem_data.copy(), (grid_x, grid_y), 4326)


@unittest.skipUnless(IS_UNDER_MPIEXEC, "Requires mpiexec with >1 rank")
class TestMPIWriteHfun(unittest.TestCase):
    """Test meshdata() write path under MPI.

    Run with: mpiexec -n 2 python -m pytest tests/api/mpi_write_hfun_test.py -v -s
    """

    def setUp(self):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()

        if self.rank == 0:
            self.tdir = Path(tempfile.mkdtemp())
            _create_test_rasters(self.tdir)
        else:
            self.tdir = None

        # bcast is collective — all ranks are synchronized when it returns.
        self.tdir = comm.bcast(self.tdir, root=0)

        dem1_path = self.tdir / "dem1.tif"
        dem2_path = self.tdir / "dem2.tif"
        self.raster_list = [Raster(dem1_path), Raster(dem2_path)]

    def tearDown(self):
        # No barrier needed — MPIExecutor.run() already synchronized
        # all ranks before returning (workers exit recv loop on STOP).
        self.raster_list = None
        gc.collect()
        if self.rank == 0:
            try:
                shutil.rmtree(self.tdir)
            except (PermissionError, FileNotFoundError):
                pass

    def test_mpi_write_hfun_basic(self):
        """Basic smoke test: MPI meshdata() produces valid output."""
        hfun = Hfun(self.raster_list, nprocs=1, hmin=10, hmax=1000)
        hfun.execution_mode = "mpi"
        meshdata = hfun.meshdata()

        if self.rank == 0:
            self.assertIsNotNone(meshdata)
            self.assertTrue(len(meshdata.values) > 0)
            self.assertTrue(len(meshdata.coords) > 0)
        else:
            self.assertIsNone(meshdata)

    def test_serial_vs_mpi_write_hfun_equivalence(self):
        """Numerical equivalence: serial meshdata == MPI meshdata."""
        # ── SERIAL baseline ──
        hfun_serial = Hfun(
            self.raster_list, nprocs=2, hmin=10, hmax=1000
        )
        hfun_serial.execution_mode = "serial"
        hfun_serial.add_subtidal_flow_limiter(
            hmin=50, lower_bound=-5, upper_bound=5
        )
        hfun_serial.add_constant_value(
            value=200, lower_bound=5, upper_bound=10
        )
        meshdata_serial = hfun_serial.meshdata()

        # ── MPI execution ──
        hfun_mpi = Hfun(
            self.raster_list, nprocs=2, hmin=10, hmax=1000
        )
        hfun_mpi.execution_mode = "mpi"
        hfun_mpi.add_subtidal_flow_limiter(
            hmin=50, lower_bound=-5, upper_bound=5
        )
        hfun_mpi.add_constant_value(
            value=200, lower_bound=5, upper_bound=10
        )
        meshdata_mpi = hfun_mpi.meshdata()

        if self.rank == 0:
            values_serial = meshdata_serial.values
            values_mpi = meshdata_mpi.values
            self.assertAlmostEqual(
                len(values_serial), len(values_mpi),
                delta=len(values_serial) * 0.01,
            )
            npt.assert_allclose(
                np.min(values_serial), np.min(values_mpi), rtol=1e-5
            )
            npt.assert_allclose(
                np.max(values_serial), np.max(values_mpi), rtol=1e-5
            )
            npt.assert_allclose(
                np.mean(values_serial), np.mean(values_mpi), rtol=1e-5
            )

    def test_run_classmethod_workers_in_recv_loop(self):
        """Simple test to verify that workers are in their recv loop when run() is called. "No deadlock"
        """
        raster = self.raster_list[0] if self.raster_list else None

        if MPIExecutor.is_manager() and raster is not None:
            valid_hfun = HfunRaster(
                raster=raster, hmin=10, hmax=1000
            )
            out_path = str(self.tdir / "run_recv_test")
            tasks = [{
                'op': 'meshdata',
                'type': 'raster',
                'original_index': 0,
                'topo_path': raster.path,
                'hfun_input_path': valid_hfun.tmpfile,
                'output_path': out_path,
                'hmin': 10,
                'hmax': 1000,
                'meshdata_kwargs': {}
            }]
        else:
            tasks = [{'op': 'meshdata', 'original_index': 0}]

        # No work_dir → submit() skips verify_shared_filesystem(),
        # goes straight to _dispatch().
        result = MPIExecutor.run(tasks)

        if MPIExecutor.is_manager():
            self.assertIsNotNone(result)
            self.assertIn(0, result)
        else:
            self.assertIsNone(result)

    def test_verify_shared_fs_inside_run(self):
        """Simple test to verify that workers are in their recv loop when run() is called with work_dir. "NO Deadlock"
        """
        raster = self.raster_list[0] if self.raster_list else None

        if MPIExecutor.is_manager() and raster is not None:
            valid_hfun = HfunRaster(
                raster=raster, hmin=10, hmax=1000
            )
            out_path = str(self.tdir / "fs_check_test")
            tasks = [{
                'op': 'meshdata',
                'type': 'raster',
                'original_index': 0,
                'topo_path': raster.path,
                'hfun_input_path': valid_hfun.tmpfile,
                'output_path': out_path,
                'hmin': 10,
                'hmax': 1000,
                'meshdata_kwargs': {}
            }]
        else:
            tasks = [{'op': 'meshdata', 'original_index': 0}]

        # work_dir=self.tdir → submit() calls verify_shared_filesystem()
        # which fires its own _dispatch(check_tasks) BEFORE the main dispatch.
        result = MPIExecutor.run(tasks, work_dir=self.tdir)

        if MPIExecutor.is_manager():
            self.assertIsNotNone(result)
            self.assertIn(0, result)
        else:
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()