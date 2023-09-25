#! python
import re
import tempfile
import unittest
from copy import deepcopy

import numpy as np
import geopandas as gpd
from jigsawpy import jigsaw_msh_t
from pyproj import CRS
from shapely.geometry import (
    Point,
    LineString,
    box,
    Polygon,
    MultiPolygon,
    GeometryCollection,
)
from shapely.ops import polygonize

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

    def test_values_are_assigned(self):
        out_msht = utils.msht_from_numpy(
            coordinates=self.in_verts,
            triangles=self.in_tria,
            crs=None
        )

        self.assertTrue(len(out_msht.value) == len(self.in_verts))
        self.assertTrue(np.all(out_msht.value == 0))

    def test_values_input_validation(self):
        with self.assertRaises(ValueError) as exc_1:
            utils.msht_from_numpy(
                coordinates=self.in_verts,
                triangles=self.in_tria,
                values=[1,2,3],
                crs=None
            )

        self.assertIsNotNone(
            re.search(
                'values must either be',
                str(exc_1.exception).lower()
            ),
        )


    def test_kwonly_args(self):
        with self.assertRaises(Exception) as exc_1:
            utils.msht_from_numpy(self.in_verts, self.in_tria)

        self.assertIsNotNone(
            re.search(
                'takes 1 positional argument',
                str(exc_1.exception).lower()
            ),
        )


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


class ShapeToMeshT(unittest.TestCase):

    def setUp(self):
        self.valid_input_1 = box(0, 0, 1, 1)
        self.valid_input_2 = gpd.GeoDataFrame(
            geometry=[self.valid_input_1]
        )
        self.valid_input_3 = gpd.GeoSeries(self.valid_input_1)
        # NOTE: Hole touching boundary is still valid shape for shapely
        self.valid_input_4 = Polygon(
            [
                [0, 0],
                [2, 2],
                [4, 0],
                [2, -2],
                [0, 0],
            ],
            [
                [
                    [0, 0],
                    [1, -0.5],
                    [2, 0],
                    [1, 0.5],
                    [0, 0]
                ]
            ]
        )
        self.valid_input_5 = MultiPolygon(
            [box(0, 0, 1, 1), box(2, 2, 3, 3)]
        )

        self.invalid_input_1 = Point(0, 0)
        self.invalid_input_2 = LineString([[0, 0], [1, 1]])
        # NOTE: Unlike hole touching boundary, this is invalid shape!!
        self.invalid_input_3 = Polygon(
            [
                [0, 0],
                [2, 2],
                [4, 0],
                [2, -2],
                [0, 0],
                [1, -1],
                [2, 0],
                [1, 1],
                [0, 0]
            ]
        )


    def test_old_io_validity(self):
        msht = utils.shape_to_msh_t(self.valid_input_1)
        with self.assertRaises(ValueError) as exc_1:
            utils.shape_to_msh_t(self.invalid_input_1)

        with self.assertRaises(ValueError) as exc_2:
            utils.shape_to_msh_t(self.invalid_input_2)

        with self.assertRaises(ValueError) as exc_3:
            utils.shape_to_msh_t(self.invalid_input_3)

        self.assertIsInstance(msht, jigsaw_msh_t)

        self.assertIsNotNone(
            re.search('invalid.*type', str(exc_1.exception).lower()),
        )

        self.assertIsNotNone(
            re.search('invalid.*type', str(exc_2.exception).lower()),
        )

        self.assertIsNotNone(
            re.search('invalid.*polygon', str(exc_3.exception).lower()),
        )


    def test_new_io_validity(self):
        msht_1 = utils.shape_to_msh_t_2(self.valid_input_1)
        msht_2 = utils.shape_to_msh_t_2(self.valid_input_2)
        msht_3 = utils.shape_to_msh_t_2(self.valid_input_3)

        with self.assertRaises(ValueError) as exc_1:
            utils.shape_to_msh_t_2(self.invalid_input_1)

        with self.assertRaises(ValueError) as exc_2:
            utils.shape_to_msh_t_2(self.invalid_input_2)

        with self.assertRaises(ValueError) as exc_3:
            utils.shape_to_msh_t_2(self.invalid_input_3)

        self.assertIsInstance(msht_1, jigsaw_msh_t)
        self.assertIsInstance(msht_2, jigsaw_msh_t)
        self.assertIsInstance(msht_3, jigsaw_msh_t)
        
        self.assertTrue(
            np.all(msht_1.vert2 == msht_2.vert2)
            & np.all(msht_2.vert2 == msht_3.vert2)
        )
        self.assertTrue(
            np.all(msht_1.edge2 == msht_2.edge2)
            & np.all(msht_2.edge2 == msht_3.edge2)
        )
        self.assertTrue(
            np.all(msht_1.value == msht_2.value)
            & np.all(msht_2.value == msht_3.value)
        )

        self.assertIsNotNone(
            re.search('have.*area', str(exc_1.exception).lower()),
        )

        self.assertIsNotNone(
            re.search('have.*area', str(exc_2.exception).lower()),
        )

        self.assertIsNotNone(
            re.search('invalid.*polygon', str(exc_3.exception).lower()),
        )


    def test_old_implementation(self):
        msht_1 = utils.shape_to_msh_t(self.valid_input_1)
        msht_2 = utils.shape_to_msh_t(self.valid_input_4)
        msht_3 = utils.shape_to_msh_t(self.valid_input_5)

        self.assertTrue(
            list(
                polygonize(msht_1.vert2['coord'][msht_1.edge2['index']])
            )[0].equals(self.valid_input_1)
        )
        self.assertTrue(
            list(
                polygonize(msht_2.vert2['coord'][msht_2.edge2['index']])
            )[0].equals(self.valid_input_4)
        )
        self.assertTrue(
            MultiPolygon(
                polygonize(msht_3.vert2['coord'][msht_3.edge2['index']])
            ).equals(self.valid_input_5)
        )

        # Old approach result in duplicate nodes!
        self.assertEqual(
            len(msht_2.vert2['coord']) - 1,
            len(np.unique(msht_2.vert2['coord'], axis=0))
        )


    def test_new_implementation(self):
        msht_1 = utils.shape_to_msh_t_2(self.valid_input_1)
        msht_2 = utils.shape_to_msh_t_2(self.valid_input_4)
        msht_3 = utils.shape_to_msh_t_2(self.valid_input_5)

        self.assertTrue(
            list(
                polygonize(msht_1.vert2['coord'][msht_1.edge2['index']])
            )[0].equals(self.valid_input_1)
        )
        self.assertTrue(
            list(
                polygonize(msht_2.vert2['coord'][msht_2.edge2['index']])
            )[0].equals(self.valid_input_4)
        )
        self.assertTrue(
            MultiPolygon(
                polygonize(msht_3.vert2['coord'][msht_3.edge2['index']])
            ).equals(self.valid_input_5)
        )

        # New approach removes duplicate nodes!
        self.assertEqual(
            len(msht_2.vert2['coord']),
            len(np.unique(msht_2.vert2['coord'], axis=0))
        )




class TriangulatePolygon(unittest.TestCase):


    def setUp(self):
        self.valid_input_1 = box(0, 0, 1, 1)
        self.valid_input_2 = gpd.GeoDataFrame(
            geometry=[self.valid_input_1]
        )
        self.valid_input_3 = gpd.GeoSeries(self.valid_input_1)
        # NOTE: Hole touching boundary is still valid shape for shapely
        self.valid_input_4 = Polygon(
            [
                [0, 0],
                [2, 2],
                [4, 0],
                [2, -2],
                [0, 0],
            ],
            [
                [
                    [0, 0],
                    [1, -0.5],
                    [2, 0],
                    [1, 0.5],
                    [0, 0]
                ]
            ]
        )
        self.valid_input_5 = MultiPolygon(
            [box(0, 0, 1, 1), box(2, 2, 3, 3)]
        )

        self.invalid_input_1 = Point(0, 0)
        self.invalid_input_2 = LineString([[0, 0], [1, 1]])
        # NOTE: Unlike hole touching boundary, this is invalid shape!!
        self.invalid_input_3 = Polygon(
            [
                [0, 0],
                [2, 2],
                [4, 0],
                [2, -2],
                [0, 0],
                [1, -1],
                [2, 0],
                [1, 1],
                [0, 0]
            ]
        )


    def test_io_validity(self):
        msht_1 = utils.triangulate_polygon(self.valid_input_1)
        msht_2 = utils.triangulate_polygon(self.valid_input_2)
        msht_3 = utils.triangulate_polygon(self.valid_input_3)

        with self.assertRaises(ValueError) as exc_1:
            utils.triangulate_polygon(self.invalid_input_1)

        with self.assertRaises(ValueError) as exc_2:
            utils.triangulate_polygon(self.invalid_input_2)

        with self.assertRaises(ValueError) as exc_3:
            utils.triangulate_polygon(self.invalid_input_3)

        self.assertIsInstance(msht_1, jigsaw_msh_t)
        self.assertIsInstance(msht_2, jigsaw_msh_t)
        self.assertIsInstance(msht_3, jigsaw_msh_t)
        
        self.assertTrue(
            np.all(msht_1.vert2 == msht_2.vert2)
            & np.all(msht_2.vert2 == msht_3.vert2)
        )
        self.assertTrue(
            np.all(msht_1.tria3 == msht_2.tria3)
            & np.all(msht_2.tria3 == msht_3.tria3)
        )
        self.assertTrue(
            np.all(msht_1.value == msht_2.value)
            & np.all(msht_2.value == msht_3.value)
        )

        self.assertEqual(len(msht_1.edge2), 0)
        self.assertEqual(len(msht_1.quad4), 0)
        self.assertEqual(len(msht_1.vert2), len(msht_1.value))
        self.assertTrue((msht_1.value == 0).all())

        self.assertIsNotNone(
            re.search(
                'must be convertible to polygon',
                str(exc_1.exception).lower()
            ),
        )

        self.assertIsNotNone(
            re.search(
                'must be convertible to polygon',
                str(exc_2.exception).lower()
            ),
        )

        self.assertIsNotNone(
            re.search('invalid.*polygon', str(exc_3.exception).lower()),
        )


    def test_preserves_boundaries(self):
        bx = Polygon(
            np.array([
                [0, 0], [0, 1], [1, 1], [2, 1], [3, 1],
                [4, 1], [4, 0], [3, 0], [2, 0], [1, 0]
            ])
            + np.random.random((10, 2)) * 0.49 # to avoid exactness!
        )
        msht = utils.triangulate_polygon(bx)
        bdry_lines = utils.get_boundary_segments(msht)

        # TODO: Make sure equal means equal in all vertices, not just
        # combined shape (i.e. no edge split)
        self.assertTrue(
            list(polygonize(bdry_lines))[0].equals(bx)
        )


    def test_aux_points(self):
        bx = Polygon(
            np.array([
                [0, 0], [0, 1], [1, 1], [2, 1], [3, 1],
                [4, 1], [4, 0], [3, 0], [2, 0], [1, 0]
            ])
            + np.random.random((10, 2)) * 0.49 # to avoid exactness!
        )
        aux_1 = [[1, 0.5], [2, 0.5], [3, 0.5]]
        aux_2 = [*aux_1, [10, 0.5]] # Out of domain points

        msht_1 = utils.triangulate_polygon(bx, aux_1, opts='p')
        msht_2 = utils.triangulate_polygon(bx, aux_2, opts='p')

        self.assertTrue(
            np.all([
                np.any([pt == v.tolist() for v in msht_1.vert2['coord']])
                for pt in aux_1
            ])
        )
        # Out of domain points are discarded
        self.assertFalse(
            np.all([
                np.any([pt == v.tolist() for v in msht_2.vert2['coord']])
                for pt in aux_2
            ])
        )


    def test_polygon_holes(self):
        poly = Polygon(
            [[0, 0], [4, 0], [4, 4], [0, 4]],
            [[[1, 1], [2, 0], [2, 2], [1, 2]]]
        )
        msht = utils.triangulate_polygon(poly, opts='p')
        mesh_poly = utils.get_mesh_polygons(msht)

        self.assertTrue(poly.equals(mesh_poly))


    def test_multipolygon(self):
        mpoly = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])

        msht = utils.triangulate_polygon(mpoly, opts='p')
        mesh_poly = utils.get_mesh_polygons(msht)

        self.assertTrue(mpoly.equals(mesh_poly))


    def test_polygons_touching_two_points_no_hole(self):
        poly1 = Polygon(
            [[0, 0], [0, 4], [6, 4], [6, 0], [4, 2], [2, 2], [0, 0]],
        )
        poly2 = Polygon(
            [[0, 0], [0, -4], [6, -4], [6, 0], [4, -2], [2, -2], [0, 0]],
        )
        multpoly = MultiPolygon([poly1, poly2])
        msht = utils.triangulate_polygon(multpoly, opts='p')
        mesh_poly = utils.get_mesh_polygons(msht)

        self.assertTrue(multpoly.equals(mesh_poly))


class GetMeshPolygon(unittest.TestCase):
    def test_always_returns_multipolygon(self):
        poly1 = Polygon(
            [[0, 0], [0, 4], [6, 4], [6, 0], [4, 2], [2, 2], [0, 0]],
        )
        poly2 = Polygon(
            [[0, 0], [0, -4], [6, -4], [6, 0], [4, -2], [2, -2], [0, 0]],
        )
        multpoly = MultiPolygon([poly1, poly2])

        msht_1 = utils.triangulate_polygon(poly1, opts='p')
        msht_2 = utils.triangulate_polygon(multpoly, opts='p')

        mesh_poly_1 = utils.get_mesh_polygons(msht_1)
        mesh_poly_2 = utils.get_mesh_polygons(msht_2)

        self.assertIsInstance(mesh_poly_1, MultiPolygon)
        self.assertIsInstance(mesh_poly_2, MultiPolygon)

if __name__ == '__main__':
    unittest.main()
