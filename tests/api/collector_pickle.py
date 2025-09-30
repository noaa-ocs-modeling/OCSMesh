import unittest
import tempfile
import platform
import shutil
import gc
import os
from pathlib import Path
import numpy as np
import numpy.testing as npt

from ocsmesh import Hfun, Raster
from ocsmesh.hfun.raster import HfunRaster
from ocsmesh.utils import raster_from_numpy



IS_WINDOWS = platform.system() == 'Windows'


class TestHfunCollectorExecution(unittest.TestCase):
    """
    Test the advanced execution features of HfunCollector, including
    serial vs parallel modes and resource cleanup.
    """


    def setUp(self):
        """Create a temporary directory and sample DEM files for testing."""
        self.tdir = Path(tempfile.mkdtemp())
        self.dem1_path = self.tdir / 'dem1.tif'
        self.dem2_path = self.tdir / 'dem2.tif'

        # Create a simple raster with a linear slope from -10 to 10
        # This makes it easy to verify the effects of refinements.
        grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
        dem_data = (grid_x * 20) - 10  # Values from -10 to 10

        # Create two identical DEM files for multi-raster tests
        raster_from_numpy(self.dem1_path, dem_data, (grid_x, grid_y), 4326)
        raster_from_numpy(self.dem2_path, dem_data.copy(), (grid_x, grid_y), 4326)

        # Create a list of raster objects for HfunCollector
        self.raster_list = [Raster(self.dem1_path), Raster(self.dem2_path)]


    def tearDown(self):
        """Remove the temporary directory and all its contents."""
        shutil.rmtree(self.tdir)


    @unittest.skipIf(IS_WINDOWS, 'Pickle tests not guaranteed stable on Windows due to I/O issues')
    def test_hfunraster_initial_value_logic(self):
        """
        Verify that HfunRaster correctly initializes from a file path.
        This is critical for the worker functions.
        """
        # 1. Create a raster with a known, unique value
        initial_hfun_path = self.tdir / 'initial_hfun.tif'
        initial_data = np.full((10, 10), 777.0, dtype=np.float32)
        grid_x, grid_y = np.mgrid[0:1:10j, 0:1:10j]
        raster_from_numpy(initial_hfun_path, initial_data, (grid_x, grid_y), 4326)

        # 2. Create an HfunRaster using this file as the initial_value
        base_raster = Raster(self.dem1_path)
        hfun = HfunRaster(
            raster=base_raster,
            initial_value=initial_hfun_path
        )

        # 3. Read the values and assert they match the initial file, not the default.
        loaded_values = hfun.get_values()
        self.assertEqual(loaded_values.shape, initial_data.shape)
        npt.assert_allclose(loaded_values, initial_data)
        # Ensure it's not the default "blank" value
        self.assertNotEqual(loaded_values[0, 0], np.finfo(np.float32).max)
        print("\nSUCCESS: HfunRaster correctly initializes from `initial_value`.")


    @unittest.skipIf(IS_WINDOWS, 'Pickle tests not guaranteed stable on Windows due to I/O issues')
    def test_execution_mode_property(self):
        """Test the behavior of the `execution_mode` property."""
        hfun = Hfun(self.raster_list, nprocs=2)

        # 1. Test that the default mode is 'serial' (lazy initialization)
        self.assertEqual(hfun.execution_mode, 'serial')

        # 2. Test setting the mode to 'parallel'
        hfun.execution_mode = 'parallel'
        self.assertEqual(hfun.execution_mode, 'parallel')

        # 3. Test that setting an invalid mode raises a ValueError
        with self.assertRaises(ValueError):
            hfun.execution_mode = 'invalid_mode'

        # 4. Test that a warning is issued if setting parallel with nprocs=1
        hfun_single_core = Hfun(self.raster_list, nprocs=1)
        with self.assertWarns(UserWarning):
            hfun_single_core.execution_mode = 'parallel'

    
    @unittest.skipIf(IS_WINDOWS, 'Pickle tests not guaranteed stable on Windows due to I/O issues')
    def test_work_dir_cleanup(self):
        """Verify that the temporary _work_dir is deleted when the object is destroyed."""
        hfun = Hfun(self.raster_list, nprocs=2)
        # Get the path to the temporary directory
        work_dir_path = hfun._work_dir

        # Assert that the directory exists after creation
        self.assertTrue(os.path.exists(work_dir_path))

        # Explicitly delete the object to trigger __del__
        del hfun
        # Encourage the garbage collector to run
        gc.collect()

        # Assert that the directory no longer exists
        self.assertFalse(os.path.exists(work_dir_path))

    
    @unittest.skipIf(IS_WINDOWS, 'Pickle tests not guaranteed stable on Windows due to I/O issues')
    def test_serial_vs_parallel_equivalence(self):
        """
        The main test: ensure that running refinements in serial and parallel
        modes produces numerically equivalent results.
        """
        nprocs = 2  # Use 2 cores for a simple parallel test

        # --- SERIAL EXECUTION ---
        hfun_serial = Hfun(self.raster_list, nprocs=nprocs, hmin=10, hmax=1000)
        # The default execution_mode is 'serial'
        
        # Add a chain of refinements
        hfun_serial.add_subtidal_flow_limiter(hmin=50, lower_bound=-5, upper_bound=5)
        hfun_serial.add_constant_value(value=200, lower_bound=5, upper_bound=10)
        
        print("\nRunning serial execution...")
        msh_t_serial = hfun_serial.msh_t()
        values_serial = msh_t_serial.value
        print("Serial execution finished.")

        # --- PARALLEL EXECUTION ---
        hfun_parallel = Hfun(self.raster_list, nprocs=nprocs, hmin=10, hmax=1000)
        hfun_parallel.execution_mode = 'parallel'  # Explicitly set to parallel
        
        # Add the exact same chain of refinements
        hfun_parallel.add_subtidal_flow_limiter(hmin=50, lower_bound=-5, upper_bound=5)
        hfun_parallel.add_constant_value(value=200, lower_bound=5, upper_bound=10)

        print("Running parallel execution...")
        msh_t_parallel = hfun_parallel.msh_t()
        values_parallel = msh_t_parallel.value
        print("Parallel execution finished.")
        
        # --- COMPARISON ---
        # NOTE: Due to minor floating point differences in meshing algorithms,
        # the vertex count and values might not be bit-for-bit identical.
        # The most robust comparison is to check key statistical properties.

        # 1. Check if the number of mesh nodes is very similar
        self.assertAlmostEqual(len(values_serial), len(values_parallel), delta=len(values_serial) * 0.01)

        # 2. Check if the min, max, and mean of the size function are almost identical
        npt.assert_allclose(np.min(values_serial), np.min(values_parallel), rtol=1e-5)
        npt.assert_allclose(np.max(values_serial), np.max(values_parallel), rtol=1e-5)
        npt.assert_allclose(np.mean(values_serial), np.mean(values_parallel), rtol=1e-5)



if __name__ == '__main__':
    unittest.main()