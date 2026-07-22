"""
MPI tests for HfunCollector (scatter/gather with self-describing tasks).

Tests that do NOT require mpiexec are in TestMPIModeProperty.
Tests that REQUIRE mpiexec are in TestMPIWriteHfun.

Running instructions:

    # Unit tests (no mpiexec needed)
    PYTHONPATH=. python -m pytest tests/api/test_mpi_integration.py::TestMPIModeProperty -v

    # MPI integration tests (require mpiexec; all ranks compute,
    # so use n>=2 for actual parallelism)
    PYTHONPATH=. mpiexec -n 2 python -m pytest tests/api/test_mpi_integration.py::TestMPIWriteHfun -v -s
    PYTHONPATH=. mpiexec -n 4 python -m pytest tests/api/test_mpi_integration.py::TestMPIWriteHfun -v -s
"""

import unittest
import tempfile
import shutil
import gc
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from ocsmesh import Hfun, Raster
from ocsmesh.utils import raster_from_numpy
from ocsmesh.hfun.collector import (
    mpi_worker_loop
)

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
    """Create two synthetic DEM rasters for testing.

    Returns list of Raster objects.
    """
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    dem_data = (grid_x * 20) - 10  # Values from -10 to 10

    dem1_path = base_dir / 'dem1.tif'
    dem2_path = base_dir / 'dem2.tif'
    raster_from_numpy(dem1_path, dem_data, (grid_x, grid_y), 4326)
    raster_from_numpy(dem2_path, dem_data.copy(), (grid_x, grid_y), 4326)

    return [Raster(dem1_path), Raster(dem2_path)]


# ════════════════════════════════════════════════════════════════════
# Unit tests — no mpiexec needed
# ════════════════════════════════════════════════════════════════════

class TestMPIModeProperty(unittest.TestCase):
    """Test execution_mode='mpi' property behavior without mpiexec."""

    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())
        self.raster_list = _create_test_rasters(self.tdir)

    def tearDown(self):
        self.raster_list = None
        gc.collect()
        try:
            shutil.rmtree(self.tdir)
        except PermissionError:
            pass

    def test_execution_mode_accepts_mpi_string(self):
        """'mpi' is a valid mode string (even if it falls back)."""
        hfun = Hfun(self.raster_list, nprocs=2)
        # When not under mpiexec, should warn and fall back
        with self.assertWarns(UserWarning):
            hfun.execution_mode = 'mpi'
        # Should have fallen back to 'parallel'
        self.assertEqual(hfun.execution_mode, 'parallel')

    def test_mpi_mode_fallback_not_under_mpiexec(self):
        """Setting mode='mpi' outside mpiexec falls back with warning."""
        hfun = Hfun(self.raster_list, nprocs=2)
        with self.assertWarns(UserWarning) as cm:
            hfun.execution_mode = 'mpi'
        self.assertIn('Falling back', str(cm.warning))

    @patch('ocsmesh.hfun.collector._get_mpi', return_value=None)
    def test_mpi_mode_fallback_no_mpi4py(self, mock_get_mpi):
        """Setting mode='mpi' without mpi4py falls back with warning."""
        hfun = Hfun(self.raster_list, nprocs=2)
        with self.assertWarns(UserWarning) as cm:
            hfun.execution_mode = 'mpi'
        self.assertIn('mpi4py is not installed', str(cm.warning))
        self.assertEqual(hfun.execution_mode, 'parallel')

    def test_invalid_mode_raises(self):
        """Invalid mode string raises ValueError."""
        hfun = Hfun(self.raster_list, nprocs=2)
        with self.assertRaises(ValueError):
            hfun.execution_mode = 'distributed'


# ════════════════════════════════════════════════════════════════════
# MPI integration tests — require mpiexec
# ════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_MPI and _is_under_mpiexec(),
                     "Requires mpiexec with >1 rank")
class TestMPIWriteHfun(unittest.TestCase):
    """Test _calculate_and_write_hfun_to_disk under MPI.

    Run with: mpiexec -n 2 python -m pytest tests/api/test_mpi_integration.py::TestMPIWriteHfun -v
    """

    def setUp(self):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()

        # Only rank 0 creates test data — broadcast path to all ranks
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
    # Scatter/gather tests — workers use mpi_worker_loop()
    #
    # Rank 0 calls normal methods (hfun.meshdata()), which internally
    # scatter tasks to all ranks (including itself) and gather results.
    # Workers just call mpi_worker_loop() and return once Rank 0 sends
    # the None sentinel via mpi_stop_workers().
    # ────────────────────────────────────────────────────────────────

    def test_mpi_write_hfun_basic(self):
        """Basic smoke test: MPI meshdata() produces valid output.

        Workers use mpi_worker_loop() — no manual dispatch needed.
        """
        comm = MPI.COMM_WORLD

        if self.rank == 0:
            print(f"\n[Rank 0] Creating Hfun and calling meshdata()...",
                  flush=True)
            hfun = Hfun(self.raster_list, nprocs=1,
                        hmin=10, hmax=1000)
            hfun.execution_mode = 'mpi'
            meshdata = hfun.meshdata()

            print(f"[Rank 0] meshdata() returned {len(meshdata.values)} "
                  f"nodes", flush=True)
            self.assertIsNotNone(meshdata)
            self.assertTrue(len(meshdata.values) > 0)
            self.assertTrue(len(meshdata.coords) > 0)
        else:
            # Workers just call mpi_worker_loop() — it handles
            # scatter/gather rounds automatically.
            print(f"[Rank {self.rank}] Entering mpi_worker_loop()...",
                  flush=True)
            mpi_worker_loop()
            print(f"[Rank {self.rank}] mpi_worker_loop() returned.",
                  flush=True)

        comm.Barrier()

    def test_serial_vs_mpi_write_hfun_equivalence(self):
        """Numerical equivalence: serial meshdata == MPI meshdata.

        Workers use mpi_worker_loop() — no manual dispatch needed.
        """
        comm = MPI.COMM_WORLD

        if self.rank == 0:
            # ── SERIAL baseline ──
            print(f"\n[Rank 0] Running serial baseline...", flush=True)
            hfun_serial = Hfun(self.raster_list, nprocs=2,
                               hmin=10, hmax=1000)
            hfun_serial.execution_mode = 'serial'
            hfun_serial.add_subtidal_flow_limiter(
                hmin=50, lower_bound=-5, upper_bound=5)
            hfun_serial.add_constant_value(
                value=200, lower_bound=5, upper_bound=10)
            meshdata_serial = hfun_serial.meshdata()
            values_serial = meshdata_serial.values
            print(f"[Rank 0] Serial: {len(values_serial)} nodes", flush=True)

            # ── MPI execution ──
            print(f"[Rank 0] Running MPI execution...", flush=True)
            hfun_mpi = Hfun(self.raster_list, nprocs=2,
                            hmin=10, hmax=1000)
            hfun_mpi.execution_mode = 'mpi'
            hfun_mpi.add_subtidal_flow_limiter(
                hmin=50, lower_bound=-5, upper_bound=5)
            hfun_mpi.add_constant_value(
                value=200, lower_bound=5, upper_bound=10)
            meshdata_mpi = hfun_mpi.meshdata()
            values_mpi = meshdata_mpi.values
            print(f"[Rank 0] MPI: {len(values_mpi)} nodes", flush=True)

            # ── COMPARE ──
            self.assertAlmostEqual(
                len(values_serial), len(values_mpi),
                delta=len(values_serial) * 0.01)
            npt.assert_allclose(
                np.min(values_serial), np.min(values_mpi), rtol=1e-5)
            npt.assert_allclose(
                np.max(values_serial), np.max(values_mpi), rtol=1e-5)
            npt.assert_allclose(
                np.mean(values_serial), np.mean(values_mpi), rtol=1e-5)
            print(f"[Rank 0] Serial and MPI results match!", flush=True)
        else:
            # Workers participate in MPI scatter/gather rounds
            # automatically via the worker loop.
            print(f"[Rank {self.rank}] Entering mpi_worker_loop()...",
                  flush=True)
            mpi_worker_loop()
            print(f"[Rank {self.rank}] mpi_worker_loop() returned.",
                  flush=True)

        comm.Barrier()


if __name__ == '__main__':
    unittest.main()
