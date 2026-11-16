"""_Footprint.covers skips rasters whose bounds are far from the points.

Compositing checks every tile's nodes against the footprint of every
higher-priority tile; reading the raster mask for each pair made the
cost grow with the square of the tile count (about 430 s for 199 tiles
on a shared filesystem). Points outside a footprint's padded bounds are
now rejected without opening the file, with the same result.
"""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from ocsmesh.hfun import collector
from ocsmesh.hfun.collector import _Footprint


def _write(path, crs, x0, y1, res, n=20, nodata_rows=3):
    z = np.ones((n, n), dtype='float32')
    z[:nodata_rows, :] = -99999.0
    with rasterio.open(path, 'w', driver='GTiff', width=n, height=n,
                       count=1, dtype='float32', crs=crs,
                       transform=from_origin(x0, y1, res, res),
                       nodata=-99999.0) as dst:
        dst.write(z, 1)
    return path


def _reference_covers(footprint, lonlat):
    """covers() without the bounds test (the previous implementation)."""
    return footprint._covers_raster(lonlat)  # pylint: disable=W0212


class TestFootprintBounds(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        self.points = np.column_stack([rng.uniform(-75.3, -74.3, 4000),
                                       rng.uniform(39.2, 40.2, 4000)])

    def _check_same(self, footprint):
        got = footprint.covers(self.points)
        expected = _reference_covers(footprint, self.points)
        np.testing.assert_array_equal(got, expected)
        self.assertTrue(got.any())          # the test points do reach it
        self.assertFalse(got.all())

    def test_geographic(self):
        path = _write(os.path.join(self.tmpdir, 'geo.tif'), 'EPSG:4326',
                      -75.0, 40.0, 0.02)
        self._check_same(_Footprint(topo_path=path))

    def test_projected(self):
        x0, y1 = Transformer.from_crs(
            'EPSG:4326', 'EPSG:32618', always_xy=True).transform(-75.0, 40.0)
        path = _write(os.path.join(self.tmpdir, 'utm.tif'), 'EPSG:32618',
                      x0, y1, 2000.0)
        self._check_same(_Footprint(topo_path=path))

    def test_edge_nodes_still_covered(self):
        path = _write(os.path.join(self.tmpdir, 'geo.tif'), 'EPSG:4326',
                      -75.0, 40.0, 0.02, nodata_rows=0)
        footprint = _Footprint(topo_path=path)
        # on the edges and a quarter pixel outside them
        edge = np.array([[-75.0, 39.8], [-74.6, 39.6], [-75.005, 39.7],
                         [-74.595, 39.9], [-74.8, 40.005], [-74.8, 39.595]])
        self.assertTrue(footprint.covers(edge).all())

    def test_far_points_do_not_open_the_raster(self):
        path = _write(os.path.join(self.tmpdir, 'geo.tif'), 'EPSG:4326',
                      -75.0, 40.0, 0.02)
        footprint = _Footprint(topo_path=path)
        far = np.array([[-70.0, 35.0], [-80.0, 45.0]])
        with mock.patch.object(collector.rasterio, 'open',
                               side_effect=AssertionError('opened')):
            self.assertFalse(footprint.covers(far).any())


if __name__ == '__main__':
    unittest.main()
