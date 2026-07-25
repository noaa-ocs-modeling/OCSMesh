"""MPI integration tests for Hfun write path (requires mpiexec)."""

import gc
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

from ocsmesh import Hfun, Raster
from ocsmesh.hfun.collector import MPITaskRunner
from ocsmesh.utils import raster_from_numpy

try:
    from mpi4py import MPI

    IS_UNDER_MPIEXEC = MPI.COMM_WORLD.Get_size() > 1
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

    return [Raster(dem1_path), Raster(dem2_path)]


@unittest.skipUnless(IS_UNDER_MPIEXEC, "Requires mpiexec with >1 rank")
class TestMPIWriteHfun(unittest.TestCase):
    """Test meshdata() write path under MPI using MPITaskRunner.

    Run with: mpiexec -n 2 python -m pytest tests/api/test_mpi_write_hfun.py -v -s
    """

    def setUp(self):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()

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

    def test_mpi_write_hfun_basic(self):
        """Basic smoke test: MPI meshdata() produces valid output."""
        runner = MPITaskRunner()

        def main():
            hfun = Hfun(self.raster_list, nprocs=1, hmin=10, hmax=1000)
            hfun.execution_mode = "mpi"
            meshdata = hfun.meshdata()

            self.assertIsNotNone(meshdata)
            self.assertTrue(len(meshdata.values) > 0)
            self.assertTrue(len(meshdata.coords) > 0)
            return meshdata

        result = runner.run(main)

        if self.rank == 0:
            self.assertIsNotNone(result)
        else:
            self.assertIsNone(result)

        MPI.COMM_WORLD.Barrier()

    def test_serial_vs_mpi_write_hfun_equivalence(self):
        """Numerical equivalence: serial meshdata == MPI meshdata."""
        runner = MPITaskRunner()

        def main():
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
            values_serial = meshdata_serial.values

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
            values_mpi = meshdata_mpi.values

            # ── COMPARE ──
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

        runner.run(main)
        MPI.COMM_WORLD.Barrier()


if __name__ == "__main__":
    unittest.main()