"""Flow limiter ``flat_areas`` option.

On flat cells (zero topography gradient) the limiter formula
``|z| / (3 |grad z|)`` is undefined. ``flat_areas='hmin'`` (default)
gives them the limiter's ``hmin``, as earlier versions did;
``flat_areas='keep'`` leaves their size unchanged.

Run serially (serial, parallel) and under ``mpiexec -n 2`` or more (mpi).
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ocsmesh import Raster
from ocsmesh.hfun.raster import HfunRaster
from tests.api import multitile

LIMITER = {'hmin': 100.0, 'hmax': 3000.0,
           'lower_bound': -30.0, 'upper_bound': 0.0}
BASE = 2000.0
SHELF = (0.25, 0.45)   # flat at -10 m between these longitudes


def shelf(x, y):
    """The fixture bathymetry with a flat shelf at -10 m."""
    z = multitile.bathymetry(x, y)
    return np.where((x > SHELF[0]) & (x < SHELF[1]), -10.0, z).astype('float32')


def _limit(flat_areas=None):
    kwargs = dict(LIMITER)
    if flat_areas is not None:
        kwargs['flat_areas'] = flat_areas

    def configure(hfun):
        hfun.add_constant_value(BASE)
        hfun.add_subtidal_flow_limiter(**kwargs)
    return configure


class TestRasterFlatAreas(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # pylint: disable=consider-using-with
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dem = multitile.write_tile(
            Path(cls._tmp.name) / 'shelf.tif',
            multitile.TileSpec('shelf', 0.004, (0.0, 0.0, 0.6, 0.3)), shelf)
        topo = Raster(str(cls.dem))
        cls.z = topo.get_values()
        x = topo.get_x()
        # flat: inside the shelf, away from its edges (one-sided gradients)
        cls.flat = np.broadcast_to(
            (x > SHELF[0] + 0.01) & (x < SHELF[1] - 0.01), cls.z.shape)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _refine(self, flat_areas=None):
        hfun = HfunRaster(Raster(str(self.dem)),
                          hmin=multitile.HMIN, hmax=multitile.HMAX)
        _limit(flat_areas)(hfun)
        return hfun.get_values()

    def test_default_is_hmin(self):
        values = self._refine()
        np.testing.assert_array_equal(values[self.flat], LIMITER['hmin'])
        np.testing.assert_array_equal(self._refine('hmin'), values)

    def test_keep_leaves_flat_cells_unchanged(self):
        values = self._refine('keep')
        np.testing.assert_array_equal(values[self.flat], BASE)

    def test_only_flat_cells_differ(self):
        differ = self._refine('hmin') != self._refine('keep')
        self.assertTrue(differ[self.flat].all())
        # sloped cells, and everything outside the bounds, are the same
        sloped = ~np.broadcast_to(
            (Raster(str(self.dem)).get_x() > SHELF[0])
            & (Raster(str(self.dem)).get_x() < SHELF[1]), self.z.shape)
        self.assertFalse(differ[sloped].any())

    def test_invalid_value(self):
        hfun = HfunRaster(Raster(str(self.dem)),
                          hmin=multitile.HMIN, hmax=multitile.HMAX)
        with self.assertRaises(ValueError):
            hfun.add_subtidal_flow_limiter(**LIMITER, flat_areas='zero')


class TestCollectorFlatAreas(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(scale=2.0, include_empty=False,
                                          depth_fn=shelf)

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def _rasters(self, mode, flat_areas=None):
        return multitile.refined_rasters(
            self.stack.paths, mode, aoi=self.stack.aoi,
            configure=_limit(flat_areas))

    def test_modes_agree(self):
        for flat_areas in (None, 'hmin', 'keep'):
            reference = self._rasters('serial', flat_areas)
            for mode in multitile.available_modes()[1:]:
                other = self._rasters(mode, flat_areas)
                if not multitile.is_manager():
                    continue
                with self.subTest(flat_areas=flat_areas, mode=mode):
                    self.assertEqual(
                        multitile.compare_rasters(reference, other), '')

    def test_option_reaches_the_tiles(self):
        default = self._rasters('serial')
        keep = self._rasters('serial', 'keep')
        if multitile.is_manager():
            self.assertNotEqual(multitile.compare_rasters(default, keep), '')
            # 'hmin' only lowers sizes, and on many more cells
            for d, k in zip(default, keep):
                self.assertTrue(np.all(d <= k))
            n_hmin = [sum(int(np.sum(r == LIMITER['hmin'])) for r in rs)
                      for rs in (default, keep)]
            self.assertGreater(n_hmin[0], 2 * n_hmin[1])

    def test_fast_method(self):
        values = {}
        for flat_areas in ('hmin', 'keep'):
            stats = multitile.meshdata_stats(
                self.stack.paths, 'serial', aoi=self.stack.aoi,
                configure=_limit(flat_areas), method='fast')
            if multitile.is_manager():
                values[flat_areas] = stats['mean']
        if multitile.is_manager():
            self.assertLess(values['hmin'], values['keep'])

    def test_invalid_value(self):
        hfun = multitile.make_hfun(self.stack.paths, 'serial')
        with self.assertRaises(ValueError):
            hfun.add_subtidal_flow_limiter(**LIMITER, flat_areas='zero')


if __name__ == '__main__':
    unittest.main()
