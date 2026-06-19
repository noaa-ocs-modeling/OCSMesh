import unittest
import tempfile
import platform
import shutil
import gc
import os
from multiprocessing.pool import Pool
from pathlib import Path
import numpy as np
import numpy.testing as npt
from shapely import geometry

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
        # Fix for Windows: rasterio/GDAL keeps file handles open as long as
        # Raster objects are in memory. Windows prevents deleting open files.
        # We explicitly destroy the raster objects to release the file locks.
        self.raster_list = None
        gc.collect()
        
        try:
            shutil.rmtree(self.tdir)
        except PermissionError:
            # Even after garbage collection, the Windows filesystem is sometimes
            # too slow to release the lock before rmtree executes. Since this is 
            # just a temporary test directory, it is safe to ignore.
            pass


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
        
        meshdata_serial = hfun_serial.meshdata()
        values_serial = meshdata_serial.values

        # --- PARALLEL EXECUTION ---
        hfun_parallel = Hfun(self.raster_list, nprocs=nprocs, hmin=10, hmax=1000)
        hfun_parallel.execution_mode = 'parallel'  # Explicitly set to parallel
        
        # Add the exact same chain of refinements
        hfun_parallel.add_subtidal_flow_limiter(hmin=50, lower_bound=-5, upper_bound=5)
        hfun_parallel.add_constant_value(value=200, lower_bound=5, upper_bound=10)

        meshdata_parallel = hfun_parallel.meshdata()
        values_parallel = meshdata_parallel.values
        
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


    @unittest.skipIf(IS_WINDOWS, 'Pickle tests not guaranteed stable on Windows due to I/O issues')
    def test_serial_vs_parallel_constraints_equivalence(self):
        """
        Verify that _apply_constraints() produces numerically equivalent
        results when run in serial vs parallel mode.

        Uses TopoConstConstraint (pickleable) with a min and max constraint
        to exercise both value_type paths.
        """
        nprocs = 2

        # --- SERIAL EXECUTION ---
        hfun_serial = Hfun(
            self.raster_list, nprocs=nprocs, hmin=10, hmax=1000)
        # Default execution_mode is 'serial'
        hfun_serial.add_topo_bound_constraint(
            value=100, upper_bound=5, lower_bound=-5, value_type='min')
        hfun_serial.add_topo_bound_constraint(
            value=500, upper_bound=0, value_type='max')

        meshdata_serial = hfun_serial.meshdata()
        values_serial = meshdata_serial.values

        # --- PARALLEL EXECUTION ---
        hfun_parallel = Hfun(
            self.raster_list, nprocs=nprocs, hmin=10, hmax=1000)
        hfun_parallel.execution_mode = 'parallel'
        hfun_parallel.add_topo_bound_constraint(
            value=100, upper_bound=5, lower_bound=-5, value_type='min')
        hfun_parallel.add_topo_bound_constraint(
            value=500, upper_bound=0, value_type='max')

        meshdata_parallel = hfun_parallel.meshdata()
        values_parallel = meshdata_parallel.values

        # --- COMPARISON ---
        # Node count should be very close (meshing is non-deterministic)
        self.assertAlmostEqual(
            len(values_serial), len(values_parallel),
            delta=len(values_serial) * 0.01)

        # Statistical properties must be nearly identical
        npt.assert_allclose(
            np.min(values_serial), np.min(values_parallel), rtol=1e-5)
        npt.assert_allclose(
            np.max(values_serial), np.max(values_parallel), rtol=1e-5)
        npt.assert_allclose(
            np.mean(values_serial), np.mean(values_parallel), rtol=1e-5)


    @unittest.skipIf(IS_WINDOWS, 'Pickle tests not guaranteed stable on Windows due to I/O issues')
    def test_parallel_falls_back_for_func_constraint(self):
        """
        Verify that when a TopoFuncConstraint (which stores a lambda)
        is present, parallel mode gracefully falls back to serial
        without raising a pickling error.
        """
        hfun = Hfun(self.raster_list, nprocs=2, hmin=10, hmax=1000)
        hfun.execution_mode = 'parallel'

        # TopoFuncConstraint uses a lambda — not pickleable by default.
        # The dispatcher should detect this and fall back to serial.
        hfun.add_topo_func_constraint(
            func=lambda i: abs(i) / 2.0,
            upper_bound=-10,
            value_type='min',
        )

        # Should NOT raise PicklingError — falls back to serial and emits a warning
        with self.assertWarns(UserWarning):
            meshdata = hfun.meshdata()
        
        self.assertIsNotNone(meshdata)
        self.assertTrue(len(meshdata.values) > 0)


    def test_mixed_raster_mesh_constraint_filtering(self):
        """
        Verify that constraints are correctly filtered when applied to a mix of
        rasters and meshes, using selective source indices.
        """
        from ocsmesh.hfun.collector import _ConstraintInfoCollector
        from ocsmesh.features.constraint import TopoConstConstraint
        from unittest.mock import MagicMock
        from ocsmesh.hfun.raster import HfunRaster
        from ocsmesh.hfun.mesh import HfunMesh

        coll = _ConstraintInfoCollector()
        
        # Add constraint 1: applies to indices 1, 2, 3
        c1 = TopoConstConstraint(10, value_type='min')
        coll.add([1, 2, 3], c1)
        
        # Add constraint 2: applies to indices 1, 7, 8
        c2 = TopoConstConstraint(20, value_type='min')
        coll.add([1, 7, 8], c2)
        
        # Add constraint 3: applies to all (no source_index)
        c3 = TopoConstConstraint(30, value_type='min')
        coll.add(None, c3)
        
        mock_raster = MagicMock(spec=HfunRaster)
        mock_mesh = MagicMock(spec=HfunMesh)
        
        # Test all indices based on the mixed scenario:
        # 1: raster, 2: raster, 3: raster
        # 4: mesh, 5: mesh, 6: mesh
        # 7: raster, 8: raster, 9: raster
        
        # Index 1 (Raster) -> Expected: c1, c2, c3
        self.assertEqual(coll.get_constraints(mock_raster, 1), [c1, c2, c3])
        
        # Index 2 & 3 (Raster) -> Expected: c1, c3
        for idx in [2 , 3]:
            self.assertEqual(coll.get_constraints(mock_raster, idx), [c1, c3])
                
        # Index 4 & 5 & 6 (Mesh) -> Expected: []
        for idx in [4 , 5 , 6]:
            self.assertEqual(coll.get_constraints(mock_mesh, idx), [])
        
        # Index 7 & 8 (Raster) -> Expected: c2, c3
        for idx in [7 , 8]:
            self.assertEqual(coll.get_constraints(mock_raster, idx), [c2, c3])
        
        # Index 9 (Raster) -> Expected: c3
        self.assertEqual(coll.get_constraints(mock_raster, 9), [c3])
        

    def test_serial_vs_parallel_patch_equivalence(self):
        """
        Verify that add_patch() without expansion_rate produces
        equivalent results in serial vs parallel modes.

        Exercises: _apply_patch() -> hfun.add_patch(pool=p)
        """
        nprocs = 2
        bx = geometry.box(0.2, 0.2, 0.8, 0.8)

        # --- SERIAL ---
        hfun_serial = Hfun(
            self.raster_list, nprocs=nprocs, hmin=10, hmax=1000)
        hfun_serial.add_patch(shape=bx, target_size=50)

        meshdata_serial = hfun_serial.meshdata()
        values_serial = meshdata_serial.values

        # --- PARALLEL ---
        hfun_parallel = Hfun(
            self.raster_list, nprocs=nprocs, hmin=10, hmax=1000)
        hfun_parallel.execution_mode = 'parallel'
        hfun_parallel.add_patch(shape=bx, target_size=50)

        meshdata_parallel = hfun_parallel.meshdata()
        values_parallel = meshdata_parallel.values

        # --- COMPARISON ---
        self.assertAlmostEqual(
            len(values_serial), len(values_parallel),
            delta=len(values_serial) * 0.01)
        npt.assert_allclose(
            np.min(values_serial), np.min(values_parallel), rtol=1e-5)
        npt.assert_allclose(
            np.max(values_serial), np.max(values_parallel), rtol=1e-5)
        npt.assert_allclose(
            np.mean(values_serial), np.mean(values_parallel), rtol=1e-5)


    def test_serial_vs_parallel_patch_with_expansion_equivalence(self):
        """
        Verify that add_patch() WITH expansion_rate produces
        equivalent results in serial vs parallel modes.

        This is the critical path: add_patch(pool=p) internally
        calls add_feature(pool=pool), exercising the shared-pool
        forwarding chain.

        Exercises: _apply_patch() -> hfun.add_patch(pool=p)
                                       -> hfun.add_feature(pool=pool)
        """
        nprocs = 2
        bx = geometry.box(0.2, 0.2, 0.8, 0.8)

        # --- SERIAL ---
        hfun_serial = Hfun(
            self.raster_list, nprocs=nprocs, hmin=10, hmax=1000)
        hfun_serial.add_patch(
            shape=bx, target_size=200, expansion_rate=0.1)

        meshdata_serial = hfun_serial.meshdata()
        values_serial = meshdata_serial.values

        # --- PARALLEL ---
        hfun_parallel = Hfun(
            self.raster_list, nprocs=nprocs, hmin=10, hmax=1000)
        hfun_parallel.execution_mode = 'parallel'
        hfun_parallel.add_patch(
            shape=bx, target_size=200, expansion_rate=0.1)

        meshdata_parallel = hfun_parallel.meshdata()
        values_parallel = meshdata_parallel.values

        # --- COMPARISON ---
        self.assertAlmostEqual(
            len(values_serial), len(values_parallel),
            delta=len(values_serial) * 0.01)
        npt.assert_allclose(
            np.min(values_serial), np.min(values_parallel), rtol=1e-5)
        npt.assert_allclose(
            np.max(values_serial), np.max(values_parallel), rtol=1e-5)
        npt.assert_allclose(
            np.mean(values_serial), np.mean(values_parallel), rtol=1e-5)


    def test_serial_vs_parallel_channel_equivalence(self):
        """
        Verify that add_channel() produces equivalent results in
        serial vs parallel modes.

        Exercises: _apply_channels() -> hfun.add_patch(pool=p)
                                          -> hfun.add_feature(pool=pool)
        """
        nprocs = 2

        # --- SERIAL ---
        hfun_serial = Hfun(
            self.raster_list, nprocs=nprocs, hmin=10, hmax=1000)
        hfun_serial.add_channel(
            level=0, width=200, target_size=100, expansion_rate=0.1)

        meshdata_serial = hfun_serial.meshdata()
        values_serial = meshdata_serial.values

        # --- PARALLEL ---
        hfun_parallel = Hfun(
            self.raster_list, nprocs=nprocs, hmin=10, hmax=1000)
        hfun_parallel.execution_mode = 'parallel'
        hfun_parallel.add_channel(
            level=0, width=200, target_size=100, expansion_rate=0.1)

        meshdata_parallel = hfun_parallel.meshdata()
        values_parallel = meshdata_parallel.values

        # --- COMPARISON ---
        self.assertAlmostEqual(
            len(values_serial), len(values_parallel),
            delta=len(values_serial) * 0.01)
        npt.assert_allclose(
            np.min(values_serial), np.min(values_parallel), rtol=1e-5)
        npt.assert_allclose(
            np.max(values_serial), np.max(values_parallel), rtol=1e-5)
        npt.assert_allclose(
            np.mean(values_serial), np.mean(values_parallel), rtol=1e-5)


    def test_add_patch_backward_compat_nprocs(self):
        """
        Verify that calling HfunRaster.add_patch() with the old
        nprocs= kwarg still works via @add_pool_args decorator
        auto-spawn. This ensures external code using the old API
        is not broken.
        """
        rast = Raster(self.dem1_path)
        hfun = HfunRaster(rast, hmin=10, hmax=1000)
        bx = geometry.box(0.2, 0.2, 0.8, 0.8)

        # Old API: nprocs= should be translated to pool= by decorator
        hfun.add_patch(
            multipolygon=bx, target_size=200, nprocs=2)

        # Verify values were actually modified
        values = hfun.get_values()
        self.assertTrue(np.any(values <= 200),
            "Patch target_size was not applied to raster values")

    def test_add_patch_with_expansion_backward_compat_nprocs(self):
        """
        Verify that calling HfunRaster.add_patch() with expansion_rate
        and the old nprocs= kwarg still works. This exercises the
        decorator -> add_feature forwarding path via nprocs.
        """
        rast = Raster(self.dem1_path)
        hfun = HfunRaster(rast, hmin=10, hmax=1000)
        bx = geometry.box(0.2, 0.2, 0.8, 0.8)

        # Old API with expansion_rate: nprocs= triggers decorator,
        # which creates a pool and passes it to add_feature internally
        hfun.add_patch(
            multipolygon=bx, target_size=200,
            expansion_rate=0.1, nprocs=2)

        values = hfun.get_values()
        self.assertTrue(np.any(values <= 200),
            "Patch target_size was not applied to raster values")

    def test_add_channel_with_pool(self):
        """
        Verify that calling HfunRaster.add_channel() with an explicit
        pool works and exercises the pool-forwarding path used by the
        add_pool_args decorator.
        """
        rast = Raster(self.dem1_path)
        hfun = HfunRaster(rast, hmin=10, hmax=1000)

        with Pool(processes=2) as pool:
            hfun.add_channel(
                level=0, width=200, target_size=100, pool=pool)

        values = hfun.get_values()
        self.assertIsNotNone(values)

    def test_add_channel_backward_compat_nprocs(self):
        """
        Verify that calling HfunRaster.add_channel() with the old
        nprocs= kwarg still works via @add_pool_args decorator.
        This keeps compatibility with existing callers.
        """
        rast = Raster(self.dem1_path)
        hfun = HfunRaster(rast, hmin=10, hmax=1000)

        # Old API: nprocs= should be translated to pool= by decorator.
        # pylint: disable=unexpected-keyword-arg,missing-kwoa
        hfun.add_channel(
            level=0, width=200, target_size=100, nprocs=2)

        values = hfun.get_values()
        self.assertIsNotNone(values)


if __name__ == '__main__':
    unittest.main()
