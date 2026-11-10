"""utils.clip_elevation clamps node elevations."""

import unittest

import numpy as np
from pyproj import CRS
from scipy.spatial import Delaunay

from ocsmesh import Mesh, utils
from ocsmesh.internal import MeshData

VALUES = [-7000.0, -100.0, 5.0, 20.0, np.nan]


def _meshdata():
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float)
    return MeshData(coords=pts, tria=Delaunay(pts).simplices,
                    values=np.array(VALUES).reshape(-1, 1),
                    crs=CRS.from_epsg(4326))


class TestClipElevation(unittest.TestCase):

    def test_both_bounds_on_mesh(self):
        mesh = Mesh(_meshdata())
        self.assertIsNone(utils.clip_elevation(mesh, -6000.0, 10.0))
        np.testing.assert_array_equal(
            mesh.meshdata.values.ravel(),
            [-6000.0, -100.0, 5.0, 10.0, np.nan])
        self.assertEqual(mesh.meshdata.values.shape, (5, 1))

    def test_one_bound_on_meshdata(self):
        md = _meshdata()
        utils.clip_elevation(md, max_elevation=10.0)
        np.testing.assert_array_equal(
            md.values.ravel(), [-7000.0, -100.0, 5.0, 10.0, np.nan])
        md = _meshdata()
        utils.clip_elevation(md, min_elevation=-6000.0)
        np.testing.assert_array_equal(
            md.values.ravel(), [-6000.0, -100.0, 5.0, 20.0, np.nan])

    def test_invalid_bounds(self):
        with self.assertRaises(ValueError):
            utils.clip_elevation(_meshdata())
        with self.assertRaises(ValueError):
            utils.clip_elevation(_meshdata(), 10.0, -10.0)


if __name__ == '__main__':
    unittest.main()
