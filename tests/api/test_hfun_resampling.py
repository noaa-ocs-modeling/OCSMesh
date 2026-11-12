"""DEM resampling for raster size functions (``resampling_factor``).

The collector's factor sets the grid the sizes are computed on; a
refinement's own factor sets the DEM it is computed from (detection for
contours and channels, depths and slopes for constant values and flow
limiters), with ``'native'`` for the DEM's own pixels.

In the multi-tile stack (hmin = 50 m) the fine tiles have ~44 m pixels,
the medium ones ~111 m and the background ~445 m, so a factor of 2.0
(100 m) resamples only the fine tiles.

Run serially (serial, parallel) and under ``mpiexec -n 2`` or more (mpi).
"""

import os
import tempfile
import unittest
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from ocsmesh import Hfun
from ocsmesh.raster import Raster
from ocsmesh.hfun.raster import HfunRaster
from ocsmesh.hfun import resampling
from tests.api import multitile


FACTOR = 2.0          # 100 m: only the ~44 m fine tiles are resampled


def _write(path, z, res=0.25, nodata=-99999.0):
    with rasterio.open(
            path, 'w', driver='GTiff', width=z.shape[1], height=z.shape[0],
            count=1, dtype='float32', crs='EPSG:4326',
            transform=from_origin(0, 1, res, res), nodata=nodata) as dst:
        dst.write(z.astype('float32'), 1)
    return path


class TestRasterResample(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        z = np.arange(16, dtype='float32').reshape(4, 4)
        z[0, 0] = -99999.0
        self.path = _write(os.path.join(self.tmpdir, 'dem.tif'), z)

    def test_methods_skip_nodata(self):
        avg = Raster(self.path)
        avg.resample(0.5, 'average')
        np.testing.assert_allclose(
            avg.get_values(), [[(1 + 4 + 5) / 3, 4.5], [10.5, 12.5]],
            rtol=1e-6)
        low = Raster(self.path)
        low.resample(0.5, Resampling.min)
        np.testing.assert_array_equal(low.get_values(), [[1, 2], [8, 10]])
        self.assertEqual(low.shape, (2, 2))
        self.assertAlmostEqual(low.dx, 0.5)

    def test_bad_method(self):
        with self.assertRaises(ValueError):
            Raster(self.path).resample(0.5, 'bogus')
        with self.assertRaises(ValueError):
            Raster(self.path).resample(0.0)

    def test_pixel_size_m(self):
        px, py = Raster(self.path).pixel_size_m()
        self.assertAlmostEqual(py, 0.25 * 110_540.0)
        self.assertLess(px, 0.25 * 111_320.0)   # cos(latitude) < 1

    def test_resampled_copy_never_finer(self):
        raster = Raster(self.path)            # ~27.7 km pixels
        self.assertIs(resampling.resampled_copy(raster, 1000.0, 'average'),
                      raster)
        coarse = resampling.resampled_copy(raster, 55_000.0, 'average')
        self.assertIsNot(coarse, raster)
        self.assertEqual(coarse.shape, (2, 2))
        self.assertEqual(raster.shape, (4, 4))  # the input is untouched


class TestRefineOn(unittest.TestCase):

    def test_reduces_finer_sizes_with_min(self):
        tmpdir = tempfile.mkdtemp()
        z = np.linspace(-40, 5, 64, dtype='float32').reshape(8, 8)
        fine_path = _write(os.path.join(tmpdir, 'fine.tif'), z, res=0.01)
        fine = Raster(fine_path)
        coarse = Raster(fine_path)
        coarse.resample(0.5, 'average')

        grid = HfunRaster(coarse, hmin=50.0, hmax=3000.0)
        grid.add_constant_value(2500.0)
        before = grid.get_values().copy()

        def depth_based(hfun):
            hfun.add_constant_value(700.0, upper_bound=-20.0)
        grid.refine_on(fine, depth_based)

        on_fine = HfunRaster(fine, hmin=50.0, hmax=3000.0)
        depth_based(on_fine)
        expected = on_fine.get_values().reshape(4, 2, 4, 2).min(axis=(1, 3))
        np.testing.assert_array_equal(
            grid.get_values(), np.minimum(before, expected))

    def test_same_grid_applies_directly(self):
        tmpdir = tempfile.mkdtemp()
        z = np.linspace(-40, 5, 16, dtype='float32').reshape(4, 4)
        raster = Raster(_write(os.path.join(tmpdir, 'dem.tif'), z))
        a = HfunRaster(raster, hmin=50.0, hmax=3000.0)
        b = HfunRaster(raster, hmin=50.0, hmax=3000.0)
        a.add_constant_value(700.0, upper_bound=-20.0)
        b.refine_on(None, lambda h: h.add_constant_value(
            700.0, upper_bound=-20.0))
        np.testing.assert_array_equal(a.get_values(), b.get_values())


class TestArguments(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(include_empty=False)

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def test_invalid_values(self):
        paths = self.stack.paths[:1]
        with self.assertRaises(ValueError):
            Hfun(paths, hmax=3000.0, resampling_factor=1.0)    # no hmin
        for bad in (0, -1, 'native', 'coarse'):
            with self.subTest(factor=bad), self.assertRaises(ValueError):
                Hfun(paths, hmin=50.0, hmax=3000.0, resampling_factor=bad)
        with self.assertRaises(ValueError):
            Hfun(paths, hmin=50.0, hmax=3000.0, resampling_method='bogus')
        hfun = Hfun(paths, hmin=50.0, hmax=3000.0)
        with self.assertRaises(ValueError):
            hfun.add_channel(resampling_factor='coarse')
        with self.assertRaises(ValueError):
            hfun.add_contour(level=0.0, resampling_factor=-0.5)
        with self.assertRaises(ValueError):
            hfun.add_subtidal_flow_limiter(resampling_method='bogus')
        hfun.add_constant_value(500.0, resampling_factor='native')


def _shapes(hfun):
    return [h.raster.shape for h in hfun._hfun_list  # pylint: disable=W0212
            if isinstance(h, HfunRaster)]


class TestGrid(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(include_empty=False)

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def _hfun(self, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return Hfun(list(self.stack.paths), hmin=multitile.HMIN,
                        hmax=multitile.HMAX, base_shape=self.stack.aoi,
                        base_shape_crs='EPSG:4326', **kwargs)

    def test_only_finer_tiles_are_resampled(self):
        if not multitile.is_manager():
            return
        native = _shapes(self._hfun())
        coarse = _shapes(self._hfun(resampling_factor=FACTOR))
        for spec, n, c in zip(self.stack.specs, native, coarse):
            with self.subTest(tile=spec.name):
                if spec.name.startswith('fine'):
                    self.assertLess(c[0], n[0])
                    self.assertLess(c[1], n[1])
                else:
                    self.assertEqual(c, n)

    def test_factor_finer_than_every_tile_changes_nothing(self):
        kwargs = {'aoi': self.stack.aoi,
                  'configure': multitile.add_depth_bands}
        reference = multitile.refined_rasters(
            self.stack.paths, 'serial', **kwargs)
        fine_factor = multitile.refined_rasters(
            self.stack.paths, 'serial',
            hfun_kwargs={'resampling_factor': 0.5}, **kwargs)
        if multitile.is_manager():
            self.assertEqual(
                multitile.compare_rasters(reference, fine_factor), '')

    def test_detection_sources(self):
        if not multitile.is_manager():
            return
        hfun = self._hfun(resampling_factor=FACTOR)
        rasters = [h for h in hfun._hfun_list   # pylint: disable=W0212
                   if isinstance(h, HfunRaster)]
        sources_for = hfun._detection_sources(rasters)  # pylint: disable=W0212
        inherit = sources_for({'resampling_factor': None})
        self.assertTrue(all(s is h for s, h in zip(inherit, rasters)))
        native = sources_for({'resampling_factor': 'native'})
        for spec, src, hfun_r in zip(self.stack.specs, native, rasters):
            with self.subTest(tile=spec.name):
                if spec.name.startswith('fine'):
                    self.assertIsInstance(src, Raster)
                    self.assertLess(src.pixel_size_m()[1], 50.0)
                else:
                    self.assertIs(src, hfun_r)
        coarser = sources_for({'resampling_factor': 4.0,
                               'resampling_method': 'min'})
        for spec, src in zip(self.stack.specs, coarser):
            if spec.name.startswith(('fine', 'medium')):
                self.assertGreater(src.pixel_size_m()[1], 190.0)


def _flow_limiter(factor):
    def configure(hfun):
        hfun.add_constant_value(3000.0)
        hfun.add_subtidal_flow_limiter(
            hmin=100.0, hmax=3000.0, lower_bound=-30.0, upper_bound=0.0,
            resampling_factor=factor)
    return configure


class TestNativeRefinements(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(include_empty=False)

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def _rasters(self, factor, collector_factor=FACTOR):
        return multitile.refined_rasters(
            self.stack.paths, 'serial', aoi=self.stack.aoi,
            configure=_flow_limiter(factor),
            hfun_kwargs={'resampling_factor': collector_factor})

    def test_native_differs_from_grid_on_resampled_tiles_only(self):
        """fine_a (sea, resampled) changes; unresampled tiles do not."""
        on_grid = self._rasters(None)
        native = self._rasters('native')
        if not multitile.is_manager():
            return
        for spec, a, b in zip(self.stack.specs, on_grid, native):
            with self.subTest(tile=spec.name):
                self.assertEqual(a.shape, b.shape)
                if spec.name == 'fine_a':
                    self.assertFalse(np.array_equal(a, b))
                elif not spec.name.startswith('fine'):
                    np.testing.assert_array_equal(a, b)
                # fine_b is land above the limiter's bounds: no change

    def test_native_without_collector_factor_is_unchanged(self):
        plain = self._rasters(None, collector_factor=None)
        native = self._rasters('native', collector_factor=None)
        if multitile.is_manager():
            self.assertEqual(multitile.compare_rasters(plain, native), '')


def _everything(hfun):
    multitile.add_depth_bands(hfun)
    hfun.add_constant_value(800.0, lower_bound=-5.0, upper_bound=0.0,
                            resampling_factor='native')
    hfun.add_subtidal_flow_limiter(hmin=100.0, hmax=3000.0,
                                   lower_bound=-30.0, upper_bound=0.0,
                                   resampling_factor='native')
    hfun.add_contour(level=0.0, expansion_rate=0.01, target_size=500.0,
                     resampling_factor='native', resampling_method='min')
    hfun.add_channel(level=0.0, width=2000.0, target_size=500.0,
                     expansion_rate=0.01, tolerance=50,
                     resampling_factor=4.0)


class TestModeEquivalence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(scale=1.0, include_empty=True)

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def test_everything_resampled(self):
        kwargs = {'aoi': self.stack.aoi, 'configure': _everything,
                  'hfun_kwargs': {'resampling_factor': FACTOR}}
        reference = multitile.refined_rasters(
            self.stack.paths, 'serial', **kwargs)
        ref_stats = multitile.meshdata_stats(
            self.stack.paths, 'serial', **kwargs)
        for mode in multitile.available_modes()[1:]:
            rasters = multitile.refined_rasters(
                self.stack.paths, mode, **kwargs)
            stats = multitile.meshdata_stats(self.stack.paths, mode, **kwargs)
            if not multitile.is_manager():
                continue
            with self.subTest(mode=mode):
                self.assertEqual(
                    multitile.compare_rasters(reference, rasters), '')
                self.assertEqual(stats['nodes'], ref_stats['nodes'])
                np.testing.assert_allclose(stats['mean'], ref_stats['mean'],
                                           rtol=1e-9)


if __name__ == '__main__':
    unittest.main()
