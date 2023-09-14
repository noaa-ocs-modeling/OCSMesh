#! python
import unittest
import tempfile
from copy import deepcopy

import numpy as np
from jigsawpy import jigsaw_msh_t
from pyproj import CRS
import re
from shapely.geometry import (
    Point,
    LineString,
    box,
    Polygon,
    MultiPolygon,
    GeometryCollection,
)

from ocsmesh import Raster, utils


class SetUp(unittest.TestCase):

    def test_cpp_version_parsing(self):
        # inline copy of code in setup.py.
        def _cpp_version(line):
            m = re.search('(\d+\.)(\d+\.)(\d+)', line)
            return line[m.start(): m.end()].split('.')

        _test_data = [
            ('c++ (GCC) 11.3.1 20221121 (Red Hat 11.3.1-4)', '11', '3', '1'),
            ('c++ (GCC) 7.3.1 20180712 (Red Hat 7.3.1-15)', '7', '3', '1'),
            ('Apple clang version 12.0.0 (clang-1200.0.32.29)', '12', '0', '0'),
            ('Apple clang version 14.0.3 (clang-1403.0.22.14.1)', '14', '0', '3'),
            ('cpp (MacPorts gcc12 12.2.0_2+stdlib_flag) 12.2.0', '12', '2', '0')
        ]
        for line in _test_data:
            major, minor, patch = _cpp_version(line[0])
            assert major == line[1]
            assert minor == line[2]
            assert patch == line[3]


class FinalizeMesh(unittest.TestCase):

    def test_cleanup_mesh_and_generate_valid_mesh(self):
        msh_t1 = utils.create_rectangle_mesh(
            nx=40, ny=40,
            holes=[50, 51],
            quads=np.hstack((
                np.arange(130, 150),
                np.arange(170, 190),
            )),
            x_extent=(-2, 2), y_extent=(-2, 2))

        msh_t2 = utils.create_rectangle_mesh(
            nx=20, ny=20,
            holes=[],
            x_extent=(-3.5, -3), y_extent=(0, 1))

        verts = msh_t1.vert2['coord']
        verts = np.vstack((verts, msh_t2.vert2['coord']))

        trias = msh_t1.tria3['index']
        trias = np.vstack((trias, msh_t2.tria3['index'] + len(msh_t1.vert2)))

        quads = msh_t1.quad4['index']
        quads = np.vstack((quads, msh_t2.quad4['index'] + len(msh_t1.vert2)))

        msh_t = utils.msht_from_numpy(
            verts, triangles=trias, quadrilaterals=quads
        )

        utils.finalize_mesh(msh_t)


    def test_cleanup_duplicate(self):

        # Create two mesh with "exact" element overlaps
        mesh_1 = utils.create_rectangle_mesh(
            nx=6, ny=6, holes=[],
            x_extent=(0, 5), y_extent=(-4, 1)
        )
        mesh_2 = utils.create_rectangle_mesh(
            nx=6, ny=6, holes=[],
            x_extent=(3, 8), y_extent=(-2, 3)
        )

        trias = deepcopy(mesh_1.tria3['index'])
        verts = deepcopy(mesh_1.vert2['coord'])
        trias = np.vstack([
            trias, deepcopy(mesh_2.tria3['index']) + len(verts)
        ])
        verts = np.vstack([
            verts, deepcopy(mesh_2.vert2['coord'])
        ])

        n_vert_pre = len(verts)
        n_tria_pre = len(trias)
        mesh_comb = utils.msht_from_numpy(
            coordinates=verts,
            triangles=trias
        )

        utils.cleanup_duplicates(mesh_comb)
        n_vert_fix = len(mesh_comb.vert2)
        n_tria_fix = len(mesh_comb.tria3)

        self.assertEqual(n_vert_pre - n_vert_fix, 12)
        self.assertEqual(n_tria_pre - n_tria_fix, 12)

        try:
            utils.get_boundary_segments(mesh_comb)
        except ValueError as e:
            self.fail(str(e))


class RemovePolygonHoles(unittest.TestCase):

    def setUp(self):
        self._main = box(0, 0, 10, 6)
        self._aux = box(-3, -3, -1, 3)
        self._hole1 = Point(3, 3).buffer(1.5)
        self._hole2 = Point(7.17, 4.13).buffer(1.43)
        self._hole1_island = Point(3.15, 3.25).buffer(1)
        self._hole1_island_hole1 = Point(3.20, 2.99).buffer(0.25)

        self._poly = Polygon(
            self._main.boundary,
            [
                self._hole1.boundary,
                self._hole2.boundary
            ]
        )
        self._island = Polygon(
            self._hole1_island.boundary,
            [
                self._hole1_island_hole1.boundary
            ]
        )
        self._multipoly1 = MultiPolygon([self._poly, self._island])
        self._multipoly2 = MultiPolygon([self._poly, self._island, self._aux])


    def test_invalid_input_raises(self):
        self.assertRaises(
            ValueError, utils.remove_holes, Point(0, 0)
        )
        self.assertRaises(
            ValueError, utils.remove_holes, LineString([[0, 0], [1, 1]])
        )
        self.assertRaises(
            ValueError, utils.remove_holes, GeometryCollection()
        )

        self.assertRaises(
            ValueError, utils.remove_holes_by_relative_size, Point(0, 0)
        )
        self.assertRaises(
            ValueError, utils.remove_holes_by_relative_size, LineString([[0, 0], [1, 1]])
        )
        self.assertRaises(
            ValueError, utils.remove_holes_by_relative_size, GeometryCollection()
        )


    def test_no_mutate_input_1(self):

        utils.remove_holes(self._poly)
        self.assertEqual(
            len(self._poly.interiors), 2
        )


    def test_no_mutate_input_2(self):

        utils.remove_holes_by_relative_size(self._poly, 1)
        self.assertEqual(
            len(self._poly.interiors), 2
        )


    def test_remove_polygon_holes(self):

        self.assertIsInstance(
            utils.remove_holes(self._poly), Polygon
        )
        self.assertTrue(
            utils.remove_holes(self._poly).is_valid
        )
        self.assertEqual(
            self._main,
            utils.remove_holes(self._main)
        )
        self.assertEqual(
            len(utils.remove_holes(self._poly).interiors),
            0
        )


    def test_remove_multipolygon_holes(self):

        self.assertIsInstance(
            utils.remove_holes(self._multipoly1), Polygon
        )
        self.assertTrue(
            utils.remove_holes(self._multipoly1).is_valid
        )
        self.assertEqual(
            len(utils.remove_holes(self._multipoly1).interiors),
            0
        )

        self.assertIsInstance(
            utils.remove_holes(self._multipoly2), MultiPolygon
        )
        self.assertTrue(
            utils.remove_holes(self._multipoly2).is_valid
        )
        self.assertEqual(
            sum(len(p.interiors) for p in utils.remove_holes(self._multipoly2).geoms),
            0
        )


    def test_remove_polygon_holes_with_size(self):
        self.assertIsInstance(
            utils.remove_holes_by_relative_size(self._poly, 1), Polygon
        )
        self.assertTrue(
            utils.remove_holes_by_relative_size(self._poly, 1).is_valid
        )
        self.assertEqual(
            self._main,
            utils.remove_holes_by_relative_size(self._main, 1)
        )
        self.assertEqual(
            len(
               utils.remove_holes_by_relative_size(
                    self._poly, 1
               ).interiors
            ),
            0
        )
        self.assertEqual(
            len(
               utils.remove_holes_by_relative_size(
                    self._poly, 0.5
               ).interiors
            ),
            0
        )
        self.assertEqual(
            len(
               utils.remove_holes_by_relative_size(
                    self._poly, 0.15
               ).interiors
            ),
            1
        )
        self.assertEqual(
            len(
               utils.remove_holes_by_relative_size(
                    self._poly, 0
               ).interiors
            ),
            2
        )



    def test_remove_multipolygon_holes_with_size(self):

        self.assertIsInstance(
            utils.remove_holes_by_relative_size(self._multipoly1, 1),
            Polygon
        )
        self.assertTrue(
            utils.remove_holes_by_relative_size(
                self._multipoly1, 1
            ).is_valid
        )
        self.assertEqual(
            len(
                utils.remove_holes_by_relative_size(
                    self._multipoly1, 1
                ).interiors
            ),
            0
        )
        self.assertEqual(
            sum(
                len(p.interiors) for p in 
                utils.remove_holes_by_relative_size(
                    self._multipoly1,
                    0
                ).geoms
            ),
            3
        )
        self.assertEqual(
            sum(
                len(p.interiors) for p in 
                utils.remove_holes_by_relative_size(
                    self._multipoly1,
                    0.1
                ).geoms
            ),
            2
        )
        self.assertEqual(
            sum(
                len(p.interiors) for p in 
                utils.remove_holes_by_relative_size(
                    self._multipoly1,
                    0.15
                ).geoms
            ),
            1
        )
        self.assertEqual(
            len(
                utils.remove_holes_by_relative_size(
                    self._multipoly1, 0.2
                ).interiors
            ),
            0
        )

        self.assertIsInstance(
            utils.remove_holes_by_relative_size(self._multipoly2, 1), MultiPolygon
        )
        self.assertTrue(
            utils.remove_holes_by_relative_size(self._multipoly2, 1).is_valid
        )
        self.assertEqual(
            sum(len(p.interiors) for p in utils.remove_holes_by_relative_size(self._multipoly2, 1).geoms),
            0
        )


class CreateRectangleMesh(unittest.TestCase):

    def test_min_input(self):
        in_nx = 20
        in_ny = 20

        out_msht = utils.create_rectangle_mesh(
            nx=in_nx, ny=in_ny, holes=[]
        )
        coo = out_msht.vert2['coord']
        x = coo[:, 0]
        y = coo[:, 1]
        tri = out_msht.tria3['index']
        quad = out_msht.quad4['index']

        self.assertIsInstance(out_msht, jigsaw_msh_t)

        self.assertEqual(coo.shape, (in_nx * in_ny, 2))
        self.assertTrue(np.all(np.logical_and(0 <= x, x < in_nx)))
        self.assertTrue(np.all(np.logical_and(0 <= y, y < in_ny)))

        self.assertEqual(tri.shape, ((in_nx-1) * (in_ny-1) * 2, 3))
        self.assertEqual(len(quad), 0)

        self.assertTrue(np.all(out_msht.value > 0))

        self.assertFalse(hasattr(out_msht, 'crs'))


    def test_extent_input(self):
        in_nx = 20
        in_ny = 20
        in_xmin = -3
        in_xmax = 2
        in_ymin = -5
        in_ymax = 4

        out_msht = utils.create_rectangle_mesh(
            nx=in_nx, ny=in_ny, holes=[],
            x_extent=(in_xmin, in_xmax), y_extent=(in_ymin, in_ymax)
        )
        coo = out_msht.vert2['coord']
        x = coo[:, 0]
        y = coo[:, 1]
        tri = out_msht.tria3['index']
        quad = out_msht.quad4['index']

        self.assertTrue(np.all(np.logical_and(in_xmin <= x, x <= in_xmax)))
        self.assertTrue(np.all(np.logical_and(in_ymin <= y, y <= in_ymax)))


    def test_1hole(self):
        in_nx = 20
        in_ny = 20
        in_holes = [11, 37]

        out_msht = utils.create_rectangle_mesh(
            nx=in_nx, ny=in_ny, holes=in_holes
        )
        coo = out_msht.vert2['coord']
        x = coo[:, 0]
        y = coo[:, 1]
        tri = out_msht.tria3['index']
        quad = out_msht.quad4['index']

        self.assertEqual(coo.shape, (in_nx * in_ny, 2))
        self.assertEqual(
            tri.shape, (((in_nx-1) * (in_ny-1) - len(in_holes)) * 2, 3)
        )


    def test_side_n_corner_holes(self):
        in_nx = 20
        in_ny = 20
        in_holes = [12, 13, 19]
        exp_isolate_from_holes = 2

        out_msht = utils.create_rectangle_mesh(
            nx=in_nx, ny=in_ny, holes=in_holes
        )
        coo = out_msht.vert2['coord']
        x = coo[:, 0]
        y = coo[:, 1]
        tri = out_msht.tria3['index']
        quad = out_msht.quad4['index']

        self.assertEqual(
            coo.shape, (in_nx * in_ny - exp_isolate_from_holes, 2)
        )
        self.assertEqual(
            tri.shape, (((in_nx-1) * (in_ny-1) - len(in_holes)) * 2, 3)
        )


    def test_combined_holes(self):
        in_nx = 20
        in_ny = 20
        in_holes = [45, 46, 64, 65]
        exp_isolate_from_holes = 1

        out_msht = utils.create_rectangle_mesh(
            nx=in_nx, ny=in_ny, holes=in_holes
        )
        coo = out_msht.vert2['coord']
        x = coo[:, 0]
        y = coo[:, 1]
        tri = out_msht.tria3['index']
        quad = out_msht.quad4['index']

        self.assertEqual(
            coo.shape, (in_nx * in_ny - exp_isolate_from_holes, 2)
        )
        self.assertEqual(
            tri.shape, (((in_nx-1) * (in_ny-1) - len(in_holes)) * 2, 3)
        )


    def test_quads(self):
        in_nx = 20
        in_ny = 20
        
        in_quads = [110, 111, 250]

        out_msht = utils.create_rectangle_mesh(
            nx=in_nx, ny=in_ny, holes=[], quads=in_quads
        )
        coo = out_msht.vert2['coord']
        x = coo[:, 0]
        y = coo[:, 1]
        tri = out_msht.tria3['index']
        quad = out_msht.quad4['index']

        self.assertEqual(coo.shape, (in_nx * in_ny, 2))
        self.assertEqual(
            tri.shape, (((in_nx-1) * (in_ny-1) - len(in_quads)) * 2, 3)
        )
        self.assertEqual(len(quad), len(in_quads))


    def test_quads_n_holes_pass_the_same(self):
        in_nx = 20
        in_ny = 20
        
        in_quads = [110]
        in_holes = in_quads

        out_msht = utils.create_rectangle_mesh(
            nx=in_nx, ny=in_ny, holes=in_holes, quads=in_quads
        )
        coo = out_msht.vert2['coord']
        x = coo[:, 0]
        y = coo[:, 1]
        tri = out_msht.tria3['index']
        quad = out_msht.quad4['index']

        self.assertEqual(coo.shape, (in_nx * in_ny, 2))
        self.assertEqual(
            tri.shape, (((in_nx-1) * (in_ny-1) - len(in_holes)) * 2, 3)
        )
        self.assertEqual(len(quad), 0)


class CreateMeshTFromNumpy(unittest.TestCase):

    def setUp(self):

        self.in_verts = [
            [0, 5],
            [0, 0],
            [2, 1],
            [3, 3],
            [2.5, 5],
            [1, 0],
            [3, 1],
            [0, 7],
            [2.5, 7],
            [0, 9],
            [2.5, 9],
        ]
        self.in_tria = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [5, 6, 2],
            [5, 2, 1],
            [2, 6, 3],
        ]
        self.in_quad = [
            [0, 7, 8, 5],
            [7, 9, 10, 8],
        ]

    def test_min_input(self):

        out_msht = utils.msht_from_numpy(
            coordinates=self.in_verts,
            triangles=self.in_tria,
        )

        self.assertIsInstance(out_msht, jigsaw_msh_t)

        self.assertTrue(
            np.all(out_msht.vert2['coord'] == np.array(self.in_verts))
        )
        self.assertTrue(
            np.all(out_msht.tria3['index'] == np.array(self.in_tria))
        )
        self.assertEqual(len(out_msht.quad4['index']), 0)
        self.assertEqual(out_msht.crs, CRS.from_epsg(4326))


    def test_quads(self):

        out_msht = utils.msht_from_numpy(
            coordinates=self.in_verts,
            triangles=self.in_tria,
            quadrilaterals=self.in_quad,
        )

        self.assertIsInstance(out_msht, jigsaw_msh_t)

        self.assertTrue(
            np.all(out_msht.vert2['coord'] == np.array(self.in_verts))
        )
        self.assertTrue(
            np.all(out_msht.tria3['index'] == np.array(self.in_tria))
        )
        self.assertTrue(
            np.all(out_msht.quad4['index'] == np.array(self.in_quad))
        )
        self.assertEqual(out_msht.crs, CRS.from_epsg(4326))


    def test_crs(self):

        out_msht_1 = utils.msht_from_numpy(
            coordinates=self.in_verts,
            triangles=self.in_tria,
            crs=None
        )
        out_msht_2 = utils.msht_from_numpy(
            coordinates=self.in_verts,
            triangles=self.in_tria,
            crs=CRS.from_user_input('esri:102008')
        )


        self.assertFalse(hasattr(out_msht_1, 'crs'))

        self.assertEqual(out_msht_2.crs, CRS.from_user_input('esri:102008'))


class CreateRasterFromNumpy(unittest.TestCase):

    def test_basic_create(self):

        in_rast_xy = np.mgrid[1:3:0.1, -1:1:0.1]
        in_rast_z = np.random.random(in_rast_xy[0].shape)

        with tempfile.NamedTemporaryFile(suffix='.tiff') as tf:
            retval = utils.raster_from_numpy(
                tf.name,
                data=in_rast_z,
                mgrid=in_rast_xy,
                crs=4326
                )

            self.assertEqual(retval, None)

            rast = Raster(tf.name)

            self.assertTrue(np.all(np.isclose(
                in_rast_xy.transpose([2,1,0]).reshape(-1, 2),
                rast.get_xy()
            )))
            self.assertTrue(np.all(in_rast_z == rast.values))
            self.assertEqual(rast.crs, CRS.from_epsg(4326))

    def test_diff_extent_x_n_y(self):
        # TODO: Test when x and y extent are different
        pass

    
    def test_data_masking(self):
        fill_value = 12
        in_rast_xy = np.mgrid[0:1:0.2, 0:1:0.2]
        in_rast_z_nomask = np.random.random(in_rast_xy[0].shape)
        in_rast_z_mask = np.ma.MaskedArray(
            in_rast_z_nomask,
            mask=np.random.random(size=in_rast_z_nomask.shape) < 0.5,
            fill_value=fill_value
        )

        with tempfile.NamedTemporaryFile(suffix='.tiff') as tf:
            utils.raster_from_numpy(
                tf.name,
                data=in_rast_z_nomask,
                mgrid=in_rast_xy,
                crs=4326
                )

            rast = Raster(tf.name)
            self.assertEqual(rast.src.nodata, None)

        with tempfile.NamedTemporaryFile(suffix='.tiff') as tf:
            utils.raster_from_numpy(
                tf.name,
                data=in_rast_z_mask,
                mgrid=in_rast_xy,
                crs=4326
                )

            rast = Raster(tf.name)
            self.assertEqual(rast.src.nodata, fill_value)


if __name__ == '__main__':
    unittest.main()
