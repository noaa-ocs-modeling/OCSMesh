#! python
import os
import tempfile
import unittest
import warnings
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
from pyproj import CRS
from shapely import geometry

from ocsmesh import utils
from ocsmesh.mesh.mesh import Mesh, Raster



def edge_at (x, y):
    return geometry.Point(x, y).buffer(0.05)

class BoundaryExtraction(unittest.TestCase):

    def setUp(self):
        """
        Note:
            x = x-index
            y = y-index

            node-index(node-id)

              25(26)             29(30)
          5     *---*---*---*---*
                | \ | \ | \ | \ |
          4     *---*---*---*---*
                | \ | \ | \ | \ |
          3     *---*---*---*---*
                | \ |   | \ | \ |
          2     *---*---*---*---*
                | \ | \ | \ | \ |
          1     *---*---*---*---*
                | \ | \ | \ | \ |
          0     *---*---*---*---*
              0(1)               4(5)

                0   1   2   3   4
        """

        # x and y coords are the same as index (in value)
        mesh_meshdata = utils.create_rectangle_mesh(nx=5, ny=6, holes=[10])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, message='Input mesh has no CRS information'
            )

            self.mesh = Mesh(mesh_meshdata)

    def _chkEqualPermutations(self, a, b):
        # Make sure there are rings!
        assert a[0] == a[-1] and b[0] == b[-1]

        a = np.array(a[:-1])
        b = np.array(b[:-1])
        idx = np.nonzero(a == b[0])
        if len(idx) == 0:
            raise ValueError("One item in arg2 is not found in arg1")
        shift = -idx[0].item()
        self.assertTrue(np.all(np.roll(a, shift=shift) == np.array(b)))

    def test_auto_boundary_fails_if_na_elev(self):
        # Set one node to nan value
        self.mesh.meshdata.values = np.nan
        with self.assertRaises(ValueError):
            self.mesh.boundaries.auto_generate()


    def test_auto_boundary_1open_correctness(self):
        # Set right boundary to be open
        self.mesh.meshdata.values[self.mesh.meshdata.coords[:, 0] > 3] = -10

        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        # Mesh has one segment of each boundary type
        self.assertEqual(len(bdry.open()), 1)
        self.assertEqual(len(bdry.land()), 1)
        self.assertEqual(len(bdry.interior()), 1)

        # Boundaries node ID list
        self.assertEqual(bdry.open().iloc[0]['index_id'], [30, 25, 20, 15, 10, 5])
        self.assertEqual(
            bdry.land().iloc[0]['index_id'],
            [5, 4, 3, 2, 1, 6, 11, 16, 21, 26, 27, 28, 29, 30]
        )
        self._chkEqualPermutations(bdry.interior().iloc[0]['index_id'], [12, 13, 18, 17, 12])


    def test_auto_boundary_2open_correctness(self):
        # Set left and right boundary to be open
        self.mesh.meshdata.values[self.mesh.meshdata.coords[:, 0] > 3] = -10
        self.mesh.meshdata.values[self.mesh.meshdata.coords[:, 0] < 1] = -10

        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        self.assertEqual(len(bdry.open()), 2)
        self.assertEqual(len(bdry.land()), 2)
        self.assertEqual(len(bdry.interior()), 1)

        # Boundaries node ID list
        self.assertEqual(bdry.open().iloc[0]['index_id'], [1, 6, 11, 16, 21, 26])
        self.assertEqual(bdry.open().iloc[1]['index_id'], [30, 25, 20, 15, 10, 5])
        self.assertEqual(bdry.land().iloc[0]['index_id'], [5, 4, 3, 2, 1])
        self.assertEqual(bdry.land().iloc[1]['index_id'], [26, 27, 28, 29, 30])
        self._chkEqualPermutations(bdry.interior().iloc[0]['index_id'], [12, 13, 18, 17, 12])


    def test_manual_boundary_specification_correctness(self):
        # Shape for wrapping bottom boundary
        shape1 = geometry.box(0.5, -0.5, 3.5, 0.5)
        shape2 = geometry.box(1.5, -0.5, 2.5, 0.5)

        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        bdry.set_open(region=shape1)
        bdry.set_land(region=shape2)

        self.assertEqual(len(bdry.open()), 2)
        self.assertEqual(len(bdry.land()), 2)
        self.assertEqual(len(bdry.interior()), 1)

        # Boundaries node ID list
        self.assertEqual(bdry.open().iloc[0]['index_id'], [1, 2])
        self.assertEqual(bdry.open().iloc[1]['index_id'], [4, 5])
        self.assertEqual(
            bdry.land().iloc[0]['index_id'],
            [1, 6, 11, 16, 21, 26, 27, 28, 29, 30, 25, 20, 15, 10, 5]
        )
        self.assertEqual(bdry.land().iloc[1]['index_id'], [2, 3, 4])
        self._chkEqualPermutations(bdry.interior().iloc[0]['index_id'], [12, 13, 18, 17, 12])


    def test_manual_boundary_notaffect_interior(self):
        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        bdry.set_open(region=geometry.box(0.5, 1.5, 2.5, 3.5))

        self.assertEqual(len(bdry.open()), 0)
        self.assertEqual(len(bdry.land()), 1)
        self.assertEqual(len(bdry.interior()), 1)


    def test_manual_boundary_convex_region(self):
        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        bdry.set_open(region=geometry.Polygon([
            (-1, 4.5),
            (0.5, 4.5),
            (0.5, 3.5),
            (-0.5, 3.5),
            (-0.5, 1.5),
            (0.5, 1.5),
            (0.5, 0.5),
            (-1, 0.5),
        ]))

        self.assertEqual(len(bdry.open()), 2)
        self.assertEqual(len(bdry.land()), 2)
        self.assertEqual(len(bdry.interior()), 1)

        self.assertEqual(bdry.open().iloc[0]['index_id'], [1, 6, 11])
        self.assertEqual(bdry.open().iloc[1]['index_id'], [16, 21, 26])
        self.assertEqual(bdry.land().iloc[0]['index_id'], [11, 16])
        self.assertEqual(
            bdry.land().iloc[1]['index_id'],
            [26, 27, 28, 29, 30, 25, 20, 15, 10, 5, 4, 3, 2, 1]
        )


    def test_specified_boundary_order_nomerge(self):
        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        bdry.set_open(region=edge_at(1, 0))
        bdry.set_open(region=edge_at(4, 1))
        bdry.set_open(region=edge_at(0, 5))
        bdry.set_open(region=edge_at(4, 5))

        self.assertEqual(len(bdry.open()), 4)
        self.assertEqual(len(bdry.land()), 4)
        self.assertEqual(len(bdry.interior()), 1)

        self.assertEqual(bdry.open().iloc[0]['index_id'], [1, 2, 3])
        self.assertEqual(bdry.open().iloc[1]['index_id'], [5, 10, 15])
        self.assertEqual(bdry.open().iloc[2]['index_id'], [21, 26, 27])
        self.assertEqual(bdry.open().iloc[3]['index_id'], [29, 30, 25])


    def test_manual_boundary_brokenring_stillconnected(self):
        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        bdry.set_open(region=edge_at(4, 5))

        # Mesh has one segment of each boundary type
        self.assertEqual(len(bdry.open()), 1)
        self.assertEqual(len(bdry.land()), 1)
        self.assertEqual(len(bdry.interior()), 1)


    def test_manual_boundary_merge_sametype(self):
        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        bdry.set_land(region=edge_at(1, 0), merge=True)
        bdry.set_open(region=edge_at(4, 3), merge=True)
        bdry.set_open(region=edge_at(4, 4), merge=True)


        # Mesh has one segment of each boundary type
        self.assertEqual(len(bdry.open()), 1)
        self.assertEqual(len(bdry.land()), 1)
        self.assertEqual(len(bdry.interior()), 1)


    def test_manual_boundary_connect_two_separate_segments(self):
        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        bdry.set_open(region=edge_at(0, 0), merge=True)
        bdry.set_open(region=edge_at(4, 0), merge=True)

        self.assertEqual(len(bdry.open()), 2)
        self.assertEqual(len(bdry.land()), 2)
        self.assertEqual(len(bdry.interior()), 1)

        self.mesh.boundaries.set_open(region=edge_at(2, 0), merge=True)

        self.assertEqual(len(bdry.open()), 1)
        self.assertEqual(len(bdry.land()), 1)
        self.assertEqual(len(bdry.interior()), 1)


    def test_specified_boundary_order_withmerge(self):
        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        bdry.set_open(region=edge_at(1, 0))
        bdry.set_open(region=edge_at(4, 1))
        bdry.set_open(region=edge_at(0, 5))
        bdry.set_open(region=edge_at(4, 5))
        bdry.set_open(region=edge_at(4, 4))
        bdry.set_open(region=edge_at(0, 0), merge=True)
        bdry.set_open(region=edge_at(0, 4), merge=True)

        self.assertEqual(len(bdry.open()), 4)
        self.assertEqual(len(bdry.land()), 4)
        self.assertEqual(len(bdry.interior()), 1)

        self.assertEqual(bdry.open().iloc[0]['index_id'], [6, 1, 2, 3])
        self.assertEqual(bdry.open().iloc[1]['index_id'], [5, 10, 15])
        self.assertEqual(bdry.open().iloc[2]['index_id'], [16, 21, 26, 27])
        self.assertEqual(bdry.open().iloc[3]['index_id'], [20, 25, 30, 29])


    def test_specify_boundary_on_imported_mesh_with_boundary(self):
        self.mesh.boundaries.auto_generate()

        tmpfd, tmppath = tempfile.mkstemp(suffix='.grd')
        self.mesh.write(tmppath, format='grd', overwrite=True)
        imported_mesh = Mesh.open(tmppath)
        os.close(tmpfd)
        os.unlink(tmppath)

        bdry = imported_mesh.boundaries

        bdry.set_open(region=edge_at(1, 0))

        self.assertEqual(len(bdry.open()), 1)
        self.assertEqual(len(bdry.land()), 1)
        self.assertEqual(len(bdry.interior()), 1)

        self.assertEqual(bdry.open().iloc[0]['index_id'], [1, 2, 3])


    def test_auto_find_islands_only(self):
        bdry = self.mesh.boundaries

        self.assertEqual(len(bdry.interior()), 0)

        bdry.find_islands()

        self.assertEqual(len(bdry.interior()), 1)
        self._chkEqualPermutations(bdry.interior().iloc[0]['index_id'], [12, 13, 18, 17, 12])


    def test_specify_boundary_on_mesh_with_no_boundary(self):
        bdry = self.mesh.boundaries

        with self.assertWarns(UserWarning) as w:
            bdry.set_open(region=edge_at(1, 0))
        self.assertTrue('didn\'t have prior boundary' in str(w.warning))

        self.assertEqual(len(bdry.open()), 1)
        self.assertEqual(len(bdry.land()), 0)
        self.assertEqual(len(bdry.interior()), 0)

        self.assertEqual(bdry.open().iloc[0]['index_id'], [1, 2, 3])


class WriteOpenBoundaryOrder(unittest.TestCase):
    """``Mesh.write(format='grd')`` writes the open boundaries largest
    first unless ``sort_open_boundaries=False``."""

    def setUp(self):
        # 11 x 6 nodes, x and y coords equal to the indexes, one hole
        meshdata = utils.create_rectangle_mesh(nx=11, ny=6, holes=[13])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, message='Input mesh has no CRS information'
            )
            self.mesh = Mesh(meshdata)
        # values default to 0: every boundary node is dry (land)
        self.mesh.boundaries.auto_generate()
        self.tdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tdir)

    @staticmethod
    def _box(x0, y0, x1, y1):
        return geometry.box(x0 - 0.05, y0 - 0.05, x1 + 0.05, y1 + 0.05)

    def _write(self, **kwargs):
        path = self.tdir / 'mesh.grd'
        self.mesh.write(path, format='grd', overwrite=True, **kwargs)
        return path.read_text().splitlines()

    @staticmethod
    def _open_sections(lines):
        """[node id list per open boundary] in file order, and the lines
        after the open boundary section."""
        i = next(k for k, l in enumerate(lines) if 'total number of open boundaries' in l)
        n_open = int(lines[i].split()[0])
        k, out = i + 2, []
        for _ in range(n_open):
            n = int(lines[k].split()[0])
            out.append([int(v) for v in lines[k + 1:k + 1 + n]])
            k += n + 1
        return out, lines[k:]

    def _set_small_then_large(self):
        # a short one on the left side, then a long one at the bottom
        self.mesh.boundaries.set_open(self._box(0, 1, 0, 2))
        self.mesh.boundaries.set_open(self._box(3, 0, 9, 0))
        small, large = [len(b) for b in self.mesh.boundaries.open()['index_id']]
        self.assertLess(small, large)
        return small, large

    def test_default_writes_largest_first(self):
        small, large = self._set_small_then_large()
        segs, _ = self._open_sections(self._write())
        self.assertEqual([len(s) for s in segs], [large, small])
        # node ids are 1-based; the bottom row is ids 1..11
        self.assertTrue(set(range(4, 11)) <= set(segs[0]))
        self.assertTrue(max(segs[0]) <= 11)

    def test_mesh_boundaries_unchanged(self):
        small, large = self._set_small_then_large()
        before = deepcopy(self.mesh.boundaries.data)
        self._write()
        self.assertEqual(self.mesh.boundaries.data, before)
        sizes = [len(b) for b in self.mesh.boundaries.open()['index_id']]
        self.assertEqual(sizes, [small, large])

    def test_no_sort_keeps_order(self):
        small, large = self._set_small_then_large()
        segs, _ = self._open_sections(self._write(sort_open_boundaries=False))
        self.assertEqual([len(s) for s in segs], [small, large])

    def test_land_and_island_sections_unchanged(self):
        self._set_small_then_large()
        _, rest_sorted = self._open_sections(self._write())
        _, rest_plain = self._open_sections(self._write(sort_open_boundaries=False))
        self.assertEqual(rest_sorted, rest_plain)

    def test_equal_sizes_keep_order(self):
        # two short ones of the same length (right and left sides), then
        # a long one at the bottom
        self.mesh.boundaries.set_open(self._box(10, 1, 10, 2))
        self.mesh.boundaries.set_open(self._box(0, 3, 0, 4))
        self.mesh.boundaries.set_open(self._box(3, 0, 9, 0))
        plain, _ = self._open_sections(self._write(sort_open_boundaries=False))
        n = [len(s) for s in plain]
        self.assertTrue(n[0] == n[1] < n[2])
        segs, _ = self._open_sections(self._write())
        self.assertEqual(segs, [plain[2], plain[0], plain[1]])

    def test_read_back(self):
        small, large = self._set_small_then_large()
        path = self.tdir / 'mesh.grd'
        self.mesh.write(path, format='grd', overwrite=True)
        imported = Mesh.open(path)
        sizes = [len(b['indexes']) for b in imported.boundaries.data[None].values()]
        self.assertEqual(sizes, [large, small])
        self.assertEqual(len(imported.boundaries.land()), len(self.mesh.boundaries.land()))
        self.assertEqual(len(imported.boundaries.interior()), 1)

    def test_no_open_boundaries(self):
        segs, _ = self._open_sections(self._write())
        self.assertEqual(segs, [])


class RasterInterpolation(unittest.TestCase):

    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())

        meshdata1 = utils.create_rectangle_mesh(
            nx=13, ny=5, x_extent=(-73.9, -71.1), y_extent=(40.55, 40.85),
            holes=[],
        )
        meshdata1.crs = CRS.from_user_input(4326)
        meshdata2 = utils.create_rectangle_mesh(
            nx=11, ny=7, x_extent=(-73.9, -71.1), y_extent=(40.55, 40.85),
            holes=[],
        )
        meshdata2.crs = CRS.from_user_input(4326)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning,
                message='Input mesh has no CRS information'
            )
            self.mesh1 = Mesh(meshdata1)
            self.mesh2 = Mesh(meshdata2)

        self.rast = self.tdir / 'rast.tif'

        rast_xy = np.mgrid[-74:-71:0.1, 40.9:40.5:-0.01]
        rast_z = np.ones((rast_xy.shape[1], rast_xy.shape[2], 2))
        rast_z[:, :, 1] = 2
        utils.raster_from_numpy(
            self.rast, rast_z, rast_xy, 4326
        )


    def tearDown(self):
        shutil.rmtree(self.tdir)


    def test_interpolation_io(self):
        rast = Raster(self.rast)

        self.mesh1.interpolate(rast)
        self.assertTrue(np.isclose(self.mesh1.value, 1).all())

        # TODO: Improve the assertion!
        with self.assertRaises(Exception):
            self.mesh1.interpolate(self.mesh2)


    def test_interpolation_band(self):
        rast = Raster(self.rast)

        self.mesh1.interpolate(rast)
        self.assertTrue(np.isclose(self.mesh1.value, 1).all())

        self.mesh1.interpolate(rast, band=2)
        self.assertTrue(np.isclose(self.mesh1.value, 2).all())


    # TODO Add more interpolation tests


if __name__ == '__main__':
    unittest.main()
