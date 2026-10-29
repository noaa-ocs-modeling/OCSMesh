"""Mesh.interpolate and raster nodata.

``method='spline'`` fitted the spline to the raw DEM values, nodata
included: nodes in (or near) a nodata hole of a higher-priority raster
got values around -99999 instead of falling back to the raster below,
as ``linear`` and ``nearest`` already do. With ``chunk_size`` the
results of the windows were concatenated instead of merged, and the
node coordinates were reprojected again for every window.
"""

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from scipy.spatial import Delaunay

from ocsmesh import Mesh, Raster
from ocsmesh.internal import MeshData

NODATA = -99999.0
RES = 0.01
METHODS = ('spline', 'linear', 'nearest')


def _write(path, array):
    with rasterio.open(
            path, 'w', driver='GTiff', height=array.shape[0],
            width=array.shape[1], count=1, dtype='float64', crs='EPSG:4326',
            transform=from_origin(0.0, array.shape[0] * RES, RES, RES),
            nodata=NODATA) as dst:
        dst.write(array, 1)
    return str(path)


def _background(x):
    return -5.0 - 195.0 * x


def _mesh(points, crs='EPSG:4326'):
    coords = points.copy()
    if crs != 'EPSG:4326':
        tr = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
        coords[:, 0], coords[:, 1] = tr.transform(coords[:, 0], coords[:, 1])
    return Mesh(MeshData(coords=coords, tria=Delaunay(points).simplices,
                         values=np.zeros((len(points), 1)),
                         crs=CRS.from_user_input(crs)))


class TestInterpolateNodata(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # pylint: disable=consider-using-with
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        xs = (np.arange(100) + 0.5) * RES
        cls.bg = _write(tmp / 'bg.tif', np.tile(_background(xs), (100, 1)))
        hi = np.full((100, 100), -50.0)
        hi[40:60, 40:60] = NODATA
        cls.hi = _write(tmp / 'hi.tif', hi)
        grid = np.linspace(0.05, 0.95, 19)
        xx, yy = np.meshgrid(grid, grid)
        cls.points = np.c_[xx.ravel(), yy.ravel()]
        x, y = cls.points[:, 0], cls.points[:, 1]
        cls.hole = (x > 0.42) & (x < 0.58) & (y > 0.42) & (y < 0.58)
        # a cubic stencil reaches 2 pixels: clear of the hole by 3
        cls.clear = ((x < 0.37) | (x > 0.63) | (y < 0.37) | (y > 0.63))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _interpolate(self, method, chunk_size=None, crs='EPSG:4326'):
        mesh = _mesh(self.points, crs)
        rasters = [Raster(self.bg), Raster(self.hi)]
        for raster in rasters:
            raster.chunk_size = chunk_size
        mesh.interpolate(rasters, method=method, nprocs=1)
        return mesh.meshdata.values.ravel()

    def _check(self, values):
        np.testing.assert_allclose(
            values[self.hole], _background(self.points[self.hole, 0]),
            atol=2.0)
        np.testing.assert_allclose(values[self.clear], -50.0, atol=1e-6)
        self.assertTrue(np.all(values > -1000.0))

    def test_hole_falls_back_to_lower_raster(self):
        for method in METHODS:
            with self.subTest(method=method):
                self._check(self._interpolate(method))

    def test_chunked_windows(self):
        for method in METHODS:
            with self.subTest(method=method):
                self._check(self._interpolate(method, chunk_size=37))

    def test_projected_mesh_chunked(self):
        for method in METHODS:
            with self.subTest(method=method):
                self._check(self._interpolate(
                    method, chunk_size=37, crs='EPSG:32631'))

    def test_spline_matches_linear_on_smooth_data(self):
        spline = self._interpolate('spline')
        linear = self._interpolate('linear')
        np.testing.assert_allclose(spline, linear, atol=1e-6)

    def test_warns_about_nodes_without_data(self):
        mesh = _mesh(self.points)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            mesh.interpolate(Raster(self.hi), method='linear', nprocs=1)
        messages = [str(w.message) for w in caught]
        self.assertTrue(any('no valid raster data' in m for m in messages),
                        messages)
        values = mesh.meshdata.values.ravel()
        np.testing.assert_array_equal(values[self.hole], 0.0)


if __name__ == '__main__':
    unittest.main()
