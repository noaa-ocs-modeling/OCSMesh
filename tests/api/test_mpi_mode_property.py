"""Unit tests for MPI execution mode property behavior (no mpiexec required)."""

import gc
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ocsmesh import Hfun, Raster
from ocsmesh.utils import raster_from_numpy


def _create_test_rasters(base_dir):
    """Create two synthetic DEM rasters for testing."""
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    dem_data = (grid_x * 20) - 10  # Values from -10 to 10

    dem1_path = base_dir / "dem1.tif"
    dem2_path = base_dir / "dem2.tif"
    raster_from_numpy(dem1_path, dem_data, (grid_x, grid_y), 4326)
    raster_from_numpy(dem2_path, dem_data.copy(), (grid_x, grid_y), 4326)

    return [Raster(dem1_path), Raster(dem2_path)]


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
        with self.assertWarns(UserWarning):
            hfun.execution_mode = "mpi"
        self.assertEqual(hfun.execution_mode, "parallel")

    def test_mpi_mode_fallback_not_under_mpiexec(self):
        """Setting mode='mpi' outside mpiexec falls back with warning."""
        hfun = Hfun(self.raster_list, nprocs=2)
        with self.assertWarns(UserWarning) as cm:
            hfun.execution_mode = "mpi"
        self.assertIn("Falling back", str(cm.warning))

    @patch("ocsmesh.hfun.collector._get_mpi", return_value=None)
    def test_mpi_mode_fallback_no_mpi4py(self, mock_get_mpi):
        """Setting mode='mpi' without mpi4py falls back with warning."""
        hfun = Hfun(self.raster_list, nprocs=2)
        with self.assertWarns(UserWarning) as cm:
            hfun.execution_mode = "mpi"
        self.assertIn("mpi4py is not installed", str(cm.warning))
        self.assertEqual(hfun.execution_mode, "parallel")

    def test_invalid_mode_raises(self):
        """Invalid mode string raises ValueError."""
        hfun = Hfun(self.raster_list, nprocs=2)
        with self.assertRaises(ValueError):
            hfun.execution_mode = "distributed"


if __name__ == "__main__":
    unittest.main()
