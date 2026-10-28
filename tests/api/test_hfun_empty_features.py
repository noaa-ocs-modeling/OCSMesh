"""Empty refinements and all-nodata tiles (NJ issue N1).

``add_feature`` and ``add_patch`` write a new raster window by window.
Windows they skipped (no overlap, nothing to refine) were never written,
read back as 0 and were clamped to ``hmin``: one all-nodata tile with a
channel or contour refinement (nothing detected) turned the whole NJ
size function into a uniform 50 m. Such tiles are now also dropped by
the collector, since they carry no sizes.
"""

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, box

from ocsmesh import Raster
from ocsmesh.hfun.raster import HfunRaster
from tests.api import multitile


def _contour(hfun):
    hfun.add_contour(level=0.0, expansion_rate=0.01, target_size=500.0)


def _channel(hfun):
    hfun.add_channel(level=0.0, width=2000.0, target_size=500.0,
                     expansion_rate=0.01, tolerance=50)


class TestHfunRasterEmptyInputs(unittest.TestCase):

    def setUp(self):
        # pylint: disable=consider-using-with
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / 'tile.tif'
        multitile.write_tile(
            path, multitile.TileSpec('tile', 0.002, (0.0, 0.0, 0.1, 0.1)))
        self.hfun = HfunRaster(Raster(str(path)),
                               hmin=multitile.HMIN, hmax=multitile.HMAX)
        self.hfun.add_constant_value(1000.0)
        self.before = self.hfun.get_values().copy()

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_feature(self):
        self.hfun.add_feature(feature=MultiLineString([]), expansion_rate=0.01,
                              target_size=500.0)
        np.testing.assert_array_equal(self.hfun.get_values(), self.before)

    def test_feature_of_empty_lines(self):
        self.hfun.add_feature(feature=LineString(), expansion_rate=0.01,
                              target_size=500.0)
        np.testing.assert_array_equal(self.hfun.get_values(), self.before)

    def test_empty_patch(self):
        self.hfun.add_patch(MultiPolygon([]), target_size=500.0)
        np.testing.assert_array_equal(self.hfun.get_values(), self.before)

    def test_patch_outside_raster(self):
        # non-empty but disjoint: every window takes the no-overlap branch
        self.hfun.add_patch(box(5.0, 5.0, 6.0, 6.0), target_size=500.0)
        np.testing.assert_array_equal(self.hfun.get_values(), self.before)


class TestAllNodataTileInCollector(unittest.TestCase):
    """A clipped tile with no valid data is dropped and changes nothing."""

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(scale=4.0, include_empty=True)

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def test_dropped_with_warning(self):
        # Construction is collective under MPI: every rank builds it, and
        # only rank 0 clips (and warns).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            hfun = multitile.make_hfun(self.stack.paths, 'serial',
                                       aoi=self.stack.aoi)
        if not multitile.is_manager():
            self.skipTest('collector construction checked on rank 0')
        self.assertTrue(any('no valid data' in str(w.message) for w in caught))
        # pylint: disable=protected-access
        self.assertEqual(len(hfun._hfun_list), len(self.stack.paths) - 1)

    def _check(self, configure):
        without = multitile.refined_rasters(
            self.stack.without('empty'), 'serial', aoi=self.stack.aoi,
            configure=configure)
        for mode in multitile.available_modes():
            with_empty = multitile.refined_rasters(
                self.stack.paths, mode, aoi=self.stack.aoi,
                configure=configure)
            if multitile.is_manager():
                with self.subTest(mode=mode):
                    self.assertEqual(
                        multitile.compare_rasters(without, with_empty), '')

    def test_contour(self):
        self._check(_contour)

    def test_channel(self):
        self._check(_channel)


if __name__ == '__main__':
    unittest.main()
