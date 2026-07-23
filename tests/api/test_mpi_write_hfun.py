"""MPI integration tests for Hfun write path (requires mpiexec with >1 rank)."""

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

    HAS_MPI = True
except ImportError:
    HAS_MPI = False


def _is_under_mpiexec():
    """True only when running under mpiexec with >1 rank."""
    if not HAS_MPI:
        return False
    return MPI.COMM_WORLD.Get_size() > 1


def _create_test_rasters(base_dir):
    """Create two synthetic DEM rasters for testing."""
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    dem_data = (grid_x * 20) - 10  # Values from -10 to 10

    dem1_path = base_dir / "dem1.tif"
    dem2_path = base_dir / "dem2.tif"
    raster_from_numpy(dem1_path, dem_data, (grid_x, grid_y), 4326)
    raster_from_numpy(dem2_path, dem_data.copy(), (grid_x, grid_y), 4326)

    return [Raster(dem1_path), Raster(dem2_path)]


@unittest.skipUnless(HAS_MPI and _is_under_mpiexec(), "Requires mpiexec with >1 rank")
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

    # ────────────────────────────────────────────────────────────────
    # Send/recv tests — all ranks call MPITaskRunner.run()
    #
    # Rank 0 runs user_fn (which calls hfun.meshdata() internally).
    # Workers enter the point-to-point recv/execute/send loop and
    # exit when Rank 0 finishes (via TAG_STOP in finally block).
    # ────────────────────────────────────────────────────────────────

    def test_mpi_write_hfun_basic(self):
        """Basic smoke test: MPI meshdata() produces valid output."""
        runner = MPITaskRunner()

        def main():
            print("\n[Rank 0] Creating Hfun and calling meshdata()...", flush=True)
            hfun = Hfun(self.raster_list, nprocs=1, hmin=10, hmax=1000)
            hfun.execution_mode = "mpi"
            meshdata = hfun.meshdata()

            print(f"[Rank 0] meshdata() returned {len(meshdata.values)} nodes", flush=True)
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
            print("\n[Rank 0] Running serial baseline...", flush=True)
            hfun_serial = Hfun(self.raster_list, nprocs=2, hmin=10, hmax=1000)
            hfun_serial.execution_mode = "serial"
            hfun_serial.add_subtidal_flow_limiter(hmin=50, lower_bound=-5, upper_bound=5)
            hfun_serial.add_constant_value(value=200, lower_bound=5, upper_bound=10)
            meshdata_serial = hfun_serial.meshdata()
            values_serial = meshdata_serial.values
            print(f"[Rank 0] Serial: {len(values_serial)} nodes", flush=True)

            # ── MPI execution ──
            print("[Rank 0] Running MPI execution...", flush=True)
            hfun_mpi = Hfun(self.raster_list, nprocs=2, hmin=10, hmax=1000)
            hfun_mpi.execution_mode = "mpi"
            hfun_mpi.add_subtidal_flow_limiter(hmin=50, lower_bound=-5, upper_bound=5)
            hfun_mpi.add_constant_value(value=200, lower_bound=5, upper_bound=10)
            meshdata_mpi = hfun_mpi.meshdata()
            values_mpi = meshdata_mpi.values
            print(f"[Rank 0] MPI: {len(values_mpi)} nodes", flush=True)

            # ── COMPARE ──
            self.assertAlmostEqual(
                len(values_serial), len(values_mpi), delta=len(values_serial) * 0.01
            )
            npt.assert_allclose(np.min(values_serial), np.min(values_mpi), rtol=1e-5)
            npt.assert_allclose(np.max(values_serial), np.max(values_mpi), rtol=1e-5)
            npt.assert_allclose(np.mean(values_serial), np.mean(values_mpi), rtol=1e-5)
            print("[Rank 0] Serial and MPI results match!", flush=True)

        runner.run(main)

        MPI.COMM_WORLD.Barrier()


if __name__ == "__main__":
    unittest.main()
