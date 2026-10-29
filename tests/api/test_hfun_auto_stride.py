"""Automatic stride of HfunRaster.meshdata() for projected DEMs.

The stride assumed the DEM was in degrees (``dx * 111000``): a projected
DEM looked ~100,000 times coarser than it is, the stride came out as 1
and the whole tile went to Gmsh, which is what the stride is there to
prevent (fine lidar tiles ran out of memory).
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import rasterio
from rasterio.transform import from_origin

from ocsmesh import Raster
from ocsmesh.engines.gmsh import GmshEngine
from ocsmesh.hfun.raster import HfunRaster

FT = 0.3048006096012192   # US survey foot


def _write(path, crs, origin, res, n=200):
    x = np.linspace(-30.0, 10.0, n)
    with rasterio.open(
            path, 'w', driver='GTiff', height=n, width=n, count=1,
            dtype='float32', crs=crs, nodata=-99999.0,
            transform=from_origin(origin[0], origin[1], res, res)) as dst:
        dst.write(np.tile(x, (n, 1)).astype('float32'), 1)
    return str(path)


class TestAutoStride(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # pylint: disable=consider-using-with
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.utm = _write(tmp / 'utm.tif', 'EPSG:32618', (580000, 4500000), 10.0)
        cls.feet = _write(tmp / 'ft.tif', 'EPSG:2263', (980000, 200000), 10.0)
        cls.geo = _write(tmp / 'geo.tif', 'EPSG:4326', (-74.0, 40.0), 0.0001)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _hfun(self, path, hmin=100.0):
        hfun = HfunRaster(Raster(path), hmin=hmin, hmax=1000.0)
        hfun.add_constant_value(500.0)
        return hfun

    def test_resolution_in_metres(self):
        # pylint: disable=protected-access
        self.assertAlmostEqual(self._hfun(self.utm)._resolution_m(), 10.0)
        self.assertAlmostEqual(self._hfun(self.feet)._resolution_m(),
                               10.0 * FT, places=6)
        self.assertAlmostEqual(self._hfun(self.geo)._resolution_m(), 11.1)

    def _sizing_points(self, path, **kwargs):
        """Number of sizing points HfunRaster.meshdata() passes to Gmsh."""
        with mock.patch.object(GmshEngine, 'generate', autospec=True,
                               side_effect=GmshEngine.generate) as spy:
            self._hfun(path).meshdata(**kwargs)
        return sum(len(c.args[2].coords) for c in spy.call_args_list)

    def test_projected_auto_stride_subsamples(self):
        # hmin 100 m on 10 m pixels: stride 5 (40 x 40 of 200 x 200)
        self.assertEqual(self._sizing_points(self.utm),
                         self._sizing_points(self.utm, stride=5))
        self.assertEqual(self._sizing_points(self.utm), 40 * 40)
        self.assertEqual(self._sizing_points(self.utm, stride=1), 200 * 200)

if __name__ == '__main__':
    unittest.main()
