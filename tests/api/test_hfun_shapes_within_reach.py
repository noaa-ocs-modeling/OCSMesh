"""Shapes beyond reach are left out of add_feature and add_patch.

A size growing from a shape at ``expansion_rate`` reaches ``hmax`` at a
finite distance, so a shape farther than that from a raster (or one of
its windows) cannot change its sizes. Every tile used to reproject and
resample every detected channel and contour of every other tile; it now
skips the far ones. The sizes must not change.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, box

from ocsmesh import Raster
from ocsmesh.hfun import raster as hfun_raster
from ocsmesh.hfun.raster import HfunRaster, _reach_m, _within_reach
from tests.api import multitile


def _keep_all(geoms, *_args, **_kwargs):
    return geoms


class TestWithinReach(unittest.TestCase):

    def test_projected(self):
        near = box(1050.0, 0.0, 1060.0, 10.0)    # 50 m east
        far = box(1200.0, 0.0, 1210.0, 10.0)     # 200 m east
        kept = _within_reach([near, far], (0.0, 0.0, 1000.0, 1000.0),
                             'EPSG:32618', 100.0)
        self.assertEqual(kept, [near])

    def test_geographic(self):
        # 0.01 deg of longitude at 40 N is about 850 m
        bounds = (-74.0, 40.0, -73.9, 40.1)
        near = LineString([(-73.895, 40.05), (-73.894, 40.05)])
        far = LineString([(-73.88, 40.05), (-73.87, 40.05)])
        kept = _within_reach([near, far], bounds, 'EPSG:4326', 500.0)
        self.assertEqual(kept, [near])

    def test_unbounded_keeps_all(self):
        geoms = [box(1e6, 1e6, 1e6 + 1, 1e6 + 1)]
        self.assertEqual(
            _within_reach(geoms, (0, 0, 1, 1), 'EPSG:32618', None), geoms)

    def test_reach(self):
        self.assertAlmostEqual(_reach_m(3000.0, 500.0, 0.01), 500.0)
        self.assertEqual(_reach_m(3000.0, 500.0, None), 0.0)
        self.assertIsNone(_reach_m(None, 500.0, 0.01))


class TestSameSizes(unittest.TestCase):
    """Same raster with and without leaving the far shapes out."""

    def setUp(self):
        # pylint: disable=consider-using-with
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = Path(self._tmp.name) / 'tile.tif'
        multitile.write_tile(
            self.path, multitile.TileSpec('tile', 0.002, (0.0, 0.0, 0.1, 0.1)))
        # Lines inside, near (within the 500 m reach) and far away
        self.lines = MultiLineString([
            [(0.02, 0.02), (0.05, 0.06)],
            [(0.103, 0.0), (0.103, 0.1)],
            [(0.5, 0.5), (0.6, 0.6)],
            [(-1.0, 0.0), (-1.0, 0.1)],
        ])
        self.polys = MultiPolygon([
            box(0.03, 0.03, 0.04, 0.05),
            box(0.102, 0.02, 0.104, 0.03),
            box(0.5, 0.5, 0.52, 0.52),
        ])

    def tearDown(self):
        self._tmp.cleanup()

    def _hfun(self, chunk_size=None):
        return HfunRaster(Raster(str(self.path), chunk_size=chunk_size),
                          hmin=50.0, hmax=3000.0)

    def _both(self, refine, chunk_size=None):
        new = self._hfun(chunk_size)
        refine(new)
        old = self._hfun(chunk_size)
        with mock.patch.object(hfun_raster, '_within_reach', _keep_all):
            refine(old)
        return new.get_values(), old.get_values()

    def test_feature(self):
        for chunk in (None, 13):
            new, old = self._both(lambda h: h.add_feature(
                feature=self.lines, expansion_rate=0.01, target_size=500.0),
                chunk_size=chunk)
            np.testing.assert_array_equal(new, old)

    def test_feature_all_far(self):
        far = MultiLineString([[(0.5, 0.5), (0.6, 0.6)]])
        new, old = self._both(lambda h: h.add_feature(
            feature=far, expansion_rate=0.01, target_size=500.0))
        np.testing.assert_array_equal(new, old)

    def test_patch(self):
        for rate in (None, 0.01):
            for chunk in (None, 13):
                new, old = self._both(lambda h, r=rate: h.add_patch(
                    self.polys, expansion_rate=r, target_size=500.0),
                    chunk_size=chunk)
                np.testing.assert_array_equal(new, old)

    def test_patch_all_far(self):
        far = MultiPolygon([box(0.5, 0.5, 0.52, 0.52)])
        for rate in (None, 0.01):
            new, old = self._both(lambda h, r=rate: h.add_patch(
                far, expansion_rate=r, target_size=500.0))
            np.testing.assert_array_equal(new, old)


if __name__ == '__main__':
    unittest.main()
