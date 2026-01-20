#! python
import re
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from collections.abc import Sequence
from collections import namedtuple
import warnings

import numpy as np
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import (
    Point,
    LineString,
    MultiLineString,
    box,
    Polygon,
    MultiPolygon,
    GeometryCollection,
)
from shapely.ops import polygonize,unary_union
from scipy import constants

from ocsmesh import Raster, utils, Mesh, MeshData
from ocsmesh.engines.factory import get_mesh_engine

def get_num_elements(mesh):
    n = 0
    if mesh.tria is not None: n += len(mesh.tria)
    if mesh.quad is not None: n += len(mesh.quad)
    return n

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

class QuadCleanup(unittest.TestCase):
    def setUp(self):
        self.in_verts = [
            [0, 5],
            [-.5, 1],
            [0, 3],
            [3, 2.1],
            [2.75, 2.5],
            [3, 2],
            [0, 7],
            [2.5, 7],
            [0, 9],
            [2.5, 9],
        ]
        self.in_tria = [
            [0, 1, 2],
        ]
        self.in_quad = [
            [0, 2, 1, 5],
            [6, 7, 9, 8],
            [6, 0, 4, 7],
            [0, 5, 3, 4],
        ]

    def test_clip_elements_by_index(self):
        out_meshdata = MeshData(
            coords=self.in_verts,
            tria=self.in_tria,
            quad=self.in_quad
        )
        clean_idx = utils.clip_elements_by_index(out_meshdata,
                                                 tria=[0],
                                                 quad=[1,2])
        self.assertIsInstance(clean_idx, MeshData)
        self.assertTrue(
            np.all(clean_idx.quad == np.array([[0, 2, 1, 5],
                                                         [0, 5, 3, 4]]))
        )


class FinalizeMesh(unittest.TestCase):

    def test_cleanup_mesh_and_generate_valid_mesh(self):
        meshdata1 = utils.create_rectangle_mesh(
            nx=40, ny=40,
            holes=[50, 51],
            quads=np.hstack((
                np.arange(130, 150),
                np.arange(170, 190),
            )),
            x_extent=(-2, 2), y_extent=(-2, 2))

        meshdata2 = utils.create_rectangle_mesh(
            nx=20, ny=20,
            holes=[],
            x_extent=(-3.5, -3), y_extent=(0, 1))

        verts = meshdata1.coords
        verts = np.vstack((verts, meshdata2.coords))

        trias = meshdata1.tria
        trias = np.vstack((trias, meshdata2.tria + len(meshdata1.coords)))

        quads = meshdata1.quad
        quads = np.vstack((quads, meshdata2.quad + len(meshdata1.coords)))

        meshdata = MeshData(
            verts, tria=trias, quad=quads
        )

        utils.finalize_mesh(meshdata)


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

        trias = deepcopy(mesh_1.tria)
        verts = deepcopy(mesh_1.coords)
        trias = np.vstack([
            trias, deepcopy(mesh_2.tria) + len(verts)
        ])
        verts = np.vstack([
            verts, deepcopy(mesh_2.coords)
        ])

        n_vert_pre = len(verts)
        n_tria_pre = len(trias)
        mesh_comb = MeshData(
            coords=verts,
            tria=trias
        )

        utils.cleanup_duplicates(mesh_comb)
        n_vert_fix = len(mesh_comb.coords)
        n_tria_fix = len(mesh_comb.tria)

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

        out_meshdata = utils.create_rectangle_mesh(
            nx=in_nx, ny=in_ny, holes=[]
        )
        coo = out_meshdata.coords
        x = coo[:, 0]
        y = coo[:, 1]
        tri = out_meshdata.tria
        quad = out_meshdata.quad

        self.assertIsInstance(out_meshdata, MeshData)

        self.assertEqual(coo.shape, (in_nx * in_ny, 2))
        self.assertTrue(np.all(np.logical_and(0 <= x, x < in_nx)))
        self.assertTrue(np.all(np.logical_and(0 <= y, y < in_ny)))

        self.assertEqual(tri.shape, ((in_nx-1) * (in_ny-1) * 2, 3))
        self.assertEqual(len(quad), 0)

        self.assertTrue(np.all(out_meshdata.values > 0))

        self.assertIsNone(out_meshdata.crs)


    def test_extent_input(self):
        in_nx = 20
        in_ny = 20
        in_xmin = -3
        in_xmax = 2
        in_ymin = -5
        in_ymax = 4

        out_meshdata = utils.create_rectangle_mesh(
            nx=in_nx, ny=in_ny, holes=[],
            x_extent=(in_xmin, in_xmax), y_extent=(in_ymin, in_ymax)
        )
        coo = out_meshdata.coords
        x = coo[:, 0]
        y = coo[:, 1]
        tri = out_meshdata.tria
        quad = out_meshdata.quad

        self.assertTrue(np.all(np.logical_and(in_xmin <= x, x <= in_xmax)))
        self.assertTrue(np.all(np.logical_and(in_ymin <= y, y <= in_ymax)))


    def test_1hole(self):
        in_nx = 20
        in_ny = 20
        in_holes = [11, 37]

        out_meshdata = utils.create_rectangle_mesh(
            nx=in_nx, ny=in_ny, holes=in_holes
        )
        coo = out_meshdata.coords
        x = coo[:, 0]
        y = coo[:, 1]
        tri = out_meshdata.tria
        quad = out_meshdata.quad

        self.assertEqual(coo.shape, (in_nx * in_ny, 2))
        self.assertEqual(
            tri.shape, (((in_nx-1) * (in_ny-1) - len(in_holes)) * 2, 3)
        )


    def test_side_n_corner_holes(self):
        in_nx = 20
        in_ny = 20
        in_holes = [12, 13, 19]
        exp_isolate_from_holes = 2

        out_meshdata = utils.create_rectangle_mesh(
            nx=in_nx, ny=in_ny, holes=in_holes
        )
        coo = out_meshdata.coords
        x = coo[:, 0]
        y = coo[:, 1]
        tri = out_meshdata.tria
        quad = out_meshdata.quad

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

        out_meshdata = utils.create_rectangle_mesh(
            nx=in_nx, ny=in_ny, holes=in_holes
        )
        coo = out_meshdata.coords
        x = coo[:, 0]
        y = coo[:, 1]
        tri = out_meshdata.tria
        quad = out_meshdata.quad

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

        out_meshdata = utils.create_rectangle_mesh(
            nx=in_nx, ny=in_ny, holes=[], quads=in_quads
        )
        coo = out_meshdata.coords
        x = coo[:, 0]
        y = coo[:, 1]
        tri = out_meshdata.tria
        quad = out_meshdata.quad

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

        out_meshdata = utils.create_rectangle_mesh(
            nx=in_nx, ny=in_ny, holes=in_holes, quads=in_quads
        )
        coo = out_meshdata.coords
        x = coo[:, 0]
        y = coo[:, 1]
        tri = out_meshdata.tria
        quad = out_meshdata.quad

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

        out_meshdata = MeshData(
            coords=self.in_verts,
            tria=self.in_tria,
        )

        self.assertIsInstance(out_meshdata, MeshData)

        self.assertTrue(
            np.all(out_meshdata.coords == np.array(self.in_verts))
        )
        self.assertTrue(
            np.all(out_meshdata.tria == np.array(self.in_tria))
        )
        self.assertEqual(len(out_meshdata.quad), 0)
        self.assertIsNone(out_meshdata.crs)


    def test_quads(self):

        out_meshdata = MeshData(
            coords=self.in_verts,
            tria=self.in_tria,
            quad=self.in_quad,
        )

        self.assertIsInstance(out_meshdata, MeshData)

        self.assertTrue(
            np.all(out_meshdata.coords == np.array(self.in_verts))
        )
        self.assertTrue(
            np.all(out_meshdata.tria == np.array(self.in_tria))
        )
        self.assertTrue(
            np.all(out_meshdata.quad == np.array(self.in_quad))
        )
        self.assertIsNone(out_meshdata.crs)


    def test_crs(self):

        out_meshdata_1 = MeshData(
            coords=self.in_verts,
            tria=self.in_tria,
            crs=None
        )
        out_meshdata_2 = MeshData(
            coords=self.in_verts,
            tria=self.in_tria,
            crs=CRS.from_user_input('esri:102008')
        )


        self.assertIsNone(out_meshdata_1.crs)

        self.assertEqual(out_meshdata_2.crs, CRS.from_user_input('esri:102008'))

    def test_values_are_assigned(self):
        out_meshdata = MeshData(
            coords=self.in_verts,
            tria=self.in_tria,
            crs=None
        )

        self.assertTrue(len(out_meshdata.values) == len(self.in_verts))
        self.assertTrue(np.all(out_meshdata.values == 0))

    def test_values_input_validation(self):
        with self.assertRaises(ValueError) as exc_1:
            MeshData(
                coords=self.in_verts,
                tria=self.in_tria,
                values=[1,2],
                crs=None
            )

        self.assertIsNotNone(
            re.search(
                'length .* does not match number of nodes',
                str(exc_1.exception).lower()
            ),
        )


class CreateRasterFromNumpy(unittest.TestCase):

    def test_basic_create(self):

        in_rast_xy = np.mgrid[1:3:0.1, -1:1:0.1]
        in_rast_z = np.random.random(in_rast_xy[0].shape)

        with tempfile.TemporaryDirectory() as tdir:
#        with tempfile.NamedTemporaryFile(suffix='.tiff', mode='w') as tf:
            retval = utils.raster_from_numpy(
                Path(tdir) / 'test_rast.tiff',
#                tf.name,
                data=in_rast_z,
                mgrid=in_rast_xy,
                crs=4326
                )

            self.assertEqual(retval, None)

            rast = Raster(Path(tdir) / 'test_rast.tiff')
#            rast = Raster(tf.name)

            self.assertTrue(np.all(np.isclose(
                in_rast_xy.transpose([2,1,0]).reshape(-1, 2),
                rast.get_xy()
            )))
            self.assertTrue(np.all(in_rast_z == rast.values))
            self.assertEqual(rast.crs, CRS.from_epsg(4326))
            del rast

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

#        with tempfile.NamedTemporaryFile(suffix='.tiff') as tf:
        with tempfile.TemporaryDirectory() as tdir:
            tf_name = Path(tdir) / 'tiff1.tiff'
            utils.raster_from_numpy(
#                tf.name,
                tf_name,
                data=in_rast_z_nomask,
                mgrid=in_rast_xy,
                crs=4326
                )

            rast = Raster(tf_name)
            self.assertEqual(rast.src.nodata, None)
            del rast

#        with tempfile.NamedTemporaryFile(suffix='.tiff') as tf:
            tf_name = Path(tdir) / 'tiff2.tiff'
            utils.raster_from_numpy(
#                tf.name,
                tf_name,
                data=in_rast_z_mask,
                mgrid=in_rast_xy,
                crs=4326
                )

            rast = Raster(tf_name)
            self.assertEqual(rast.src.nodata, fill_value)
            del rast


    def test_multiband_raster_data(self):
        nbands = 5
        in_data = np.ones((3, 4, nbands))
        for i in range(nbands):
            in_data[:, :, i] *= i
        in_rast_xy = np.mgrid[-74:-71:1, 40.5:40.9:0.1]
#        with tempfile.NamedTemporaryFile(suffix='.tiff') as tf:
        with tempfile.TemporaryDirectory() as tdir:
            tf_name = Path(tdir) / 'tiff3.tiff'
            utils.raster_from_numpy(
                tf_name,
#                tf.name,
                data=in_data,
                mgrid=in_rast_xy,
                crs=4326
                )
            rast = Raster(tf_name)
#            rast = Raster(tf.name)
            self.assertEqual(rast.count, nbands)
            for i in range(nbands):
                with self.subTest(band_number=i):
                    self.assertTrue(
                        (rast.get_values(band=i+1) == i).all()
                    )
            del rast


    def test_multiband_raster_invalid_io(self):
        in_data = np.ones((3, 4, 5, 6))
        in_rast_xy = np.mgrid[-74:-71:1, 40.5:40.9:0.1]
        with tempfile.NamedTemporaryFile(suffix='.tiff') as tf:
            with self.assertRaises(ValueError) as cm:
                utils.raster_from_numpy(
                    tf.name,
                    data=in_data,
                    mgrid=in_rast_xy,
                    crs=4326
                    )
            exc = cm.exception
            self.assertRegex(str(exc).lower(), '.*dimension.*')


class GetMeshPolygon(unittest.TestCase):
    def test_always_returns_multipolygon(self):
        poly1 = Polygon(
            [[0, 0], [0, 4], [6, 4], [6, 0], [4, 2], [2, 2], [0, 0]],
        )
        poly2 = Polygon(
            [[0, 0], [0, -4], [6, -4], [6, 0], [4, -2], [2, -2], [0, 0]],
        )
        multpoly = MultiPolygon([poly1, poly2])

        engine = get_mesh_engine('triangle')
        meshdata_1 = engine.generate(gpd.GeoSeries(poly1))
        meshdata_2 = engine.generate(gpd.GeoSeries(multpoly))

        mesh_poly_1 = utils.get_mesh_polygons(meshdata_1)
        mesh_poly_2 = utils.get_mesh_polygons(meshdata_2)

        self.assertIsInstance(mesh_poly_1, MultiPolygon)
        self.assertIsInstance(mesh_poly_2, MultiPolygon)


class RepartitionFeature(unittest.TestCase):
    def _chk_segment_validity(
        self,
        results,
        n_segment,
        fix_len,
        last_len,
    ):
        with self.subTest('Invalid number of segments'):
            self.assertEqual(len(results), n_segment)

        with self.subTest('Invalid segment length'):
            self.assertTrue(all(len(i.coords) == fix_len for i in results[:-1]))
            self.assertEqual(len(results[-1].coords), last_len)

        for i, (seg1, seg2) in enumerate(zip(results[:-1], results[1:])):
            with self.subTest('Invalid end nodes', segment_1=i):
                crd1 = list(seg1.coords)
                crd2 = list(seg2.coords)
                self.assertEqual(crd1[-1], crd2[0])


    def test_io_types(self):
        lstr1 = LineString([[i, i] for i in range(100)])
        res1 = utils.repartition_features(lstr1, 20)
        self.assertIsInstance(res1, Sequence)
        self.assertTrue(
            all([isinstance(i, LineString) for i in res1])
        )

        wrong_in_1 = box(0, 0, 1, 1)
        with self.assertRaises(ValueError) as exc_1:
            utils.repartition_features(wrong_in_1, 20)
        self.assertIsNotNone(
            re.search(
                'must be a LineString',
                str(exc_1.exception)
            ),
        )

        wrong_in_2 = MultiLineString(
            [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]
        )
        with self.assertRaises(ValueError) as exc_2:
            utils.repartition_features(wrong_in_2, 20)
        self.assertIsNotNone(
            re.search(
                'must be a LineString',
                str(exc_2.exception)
            ),
        )

        wrong_in_3 = 20.0
        with self.assertRaises(ValueError) as exc_3:
            utils.repartition_features(lstr1, wrong_in_3)
        self.assertIsNotNone(
            re.search(
                'must be an integer',
                str(exc_3.exception)
            ),
        )


    def test_validity(self):
        # NOTE: In counts, be careful about having shared nodes.
        # The first 4-5 tests are for easy hand verification
        lstr = LineString([[i, i] for i in range(10)])
        res = utils.repartition_features(lstr, 2)
        self._chk_segment_validity(
            res, n_segment=9, fix_len=2, last_len=2
        )

        lstr = LineString([[i, i] for i in range(10)])
        res = utils.repartition_features(lstr, 3)
        self._chk_segment_validity(
            res, n_segment=5, fix_len=3, last_len=2
        )

        lstr = LineString([[i, i] for i in range(11)])
        res = utils.repartition_features(lstr, 3)
        self._chk_segment_validity(
            res, n_segment=5, fix_len=3, last_len=3
        )

        lstr = LineString([[i, i] for i in range(47)])
        res = utils.repartition_features(lstr, 5)
        self._chk_segment_validity(
            res, n_segment=12, fix_len=5, last_len=3
        )

        lstr = LineString([[i, i] for i in range(100)])
        res = utils.repartition_features(lstr, 7)
        self._chk_segment_validity(
            res, n_segment=17, fix_len=7, last_len=4
        )

        lstr = LineString([[i, i] for i in range(100000)])
        res = utils.repartition_features(lstr, 200)
        self._chk_segment_validity(
            res, n_segment=503, fix_len=200, last_len=102
        )

        lstr = LineString([[i, i] for i in range(100000)])
        res = utils.repartition_features(lstr, 201)
        self._chk_segment_validity(
            res, n_segment=500, fix_len=201, last_len=200
        )


class TransformLineString(unittest.TestCase):
    def _chk_segments(self, lstr_in, result, fix_dist):
        with self.subTest('Invalid number of segments!'):
            exp_nseg = (
                lstr_in.length // fix_dist
                + 1 * bool(lstr_in.length % fix_dist)
            )
            self.assertEqual(exp_nseg, len(result.coords) - 1)

        with self.subTest('Invalid segment length!'):
            out_crd = np.array(result.coords)
            out_lens = ((out_crd[1:] - out_crd[:-1]) ** 2).sum(axis=1) ** 0.5
            np.testing.assert_allclose(out_lens[:-1], fix_dist)


    def _chk_begin_end(self, lstr_in, result):
        with self.subTest('Beginning and end points must remain the same!'):
            in_crd = list(lstr_in.coords)
            out_crd = list(result.coords)
            self.assertEqual(in_crd[0], out_crd[0])
            self.assertEqual(in_crd[-1], out_crd[-1])


    def test_io_types(self):
        lstr1 = LineString([[i, i] for i in range(100)])
        res1 = utils.transform_linestring(lstr1, 2)
        self.assertIsInstance(res1, LineString)

        wrong_in_1 = box(0, 0, 1, 1)
        with self.assertRaises(ValueError) as exc_1:
            utils.repartition_features(wrong_in_1, 2)
        self.assertIsNotNone(
            re.search(
                'must be a LineString',
                str(exc_1.exception)
            ),
        )

        wrong_in_2 = MultiLineString(
            [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]
        )
        with self.assertRaises(ValueError) as exc_2:
            utils.repartition_features(wrong_in_2, 2)
        self.assertIsNotNone(
            re.search(
                'must be a LineString',
                str(exc_2.exception)
            ),
        )


    def test_validity(self):
        lstr1 = LineString([[i, i] for i in range(50)])
        res1 = utils.transform_linestring(lstr1, 3)
        self._chk_segments(lstr1, res1, 3)
        self._chk_begin_end(lstr1, res1)

        lstr2 = LineString([[i, i] for i in range(50)])
        res2 = utils.transform_linestring(lstr2, 0.3)
        self._chk_segments(lstr2, res2, 0.3)
        self._chk_begin_end(lstr2, res2)

        # TODO: Add test cases of non-straight linestring


# ==========================================
# NEW TESTS (Physics, Topo, Projections, etc)
# ==========================================

class TestPhysicsCalculations(unittest.TestCase):
    """Tests for physical parameter estimations"""

    def test_can_velocity_be_approximated(self):
        # depth (-1) <= -abs(2) --> -1 <= -2 --> False.
        # Shallow water returns False in this implementation.
        self.assertFalse(utils.can_velocity_be_approximated_by_linear_wave_theory(-1.0,
                                                                                  2.0))

        # Deep water: -5 <= -2 --> True
        self.assertTrue(utils.can_velocity_be_approximated_by_linear_wave_theory(-5.0,
                                                                                 2.0))

    def test_estimate_particle_velocity(self):
        h = -100.0
        amp = 2.0
        # Deep: u = amp * sqrt(g/|h|)
        expected_v = amp * np.sqrt(constants.g / abs(h))
        calc_v = utils.estimate_particle_velocity_from_depth(h, amp)
        self.assertAlmostEqual(float(calc_v), expected_v)

    def test_approximate_courant_number(self):
    # If not, this test will pass IF we feed it 1-element arrays instead of floats.
        h = np.array([-100.0])
        dt = 10.0
        dx = np.array([100.0])
        amp = 2.0

        c_num = utils.approximate_courant_number_for_depth(h, dt, dx, amp)
        self.assertGreater(c_num[0], 0)


class TestMeshTopologyOps(unittest.TestCase):
    """Tests for skewed elements, ordering, quad conversion, etc."""

    def test_order_mesh_ccw(self):
        # Create a single triangle defined Clockwise (CW)
        # (0,0) -> (0,1) -> (1,0) is CW
        coords = np.array([[0,0], [0,1], [1,0]])
        tria = np.array([[0, 1, 2]])

        mesh = MeshData(coords=coords, tria=tria)

        # verify initial order is effectively CW or at least not CCW relative to std
        # Area = 0.5 * (x1(y2-y3) + ...) -> 0.5 * (0(1-0) + 0(0-0) + 1(0-1)) = -0.5

        ordered_mesh = utils.order_mesh(mesh)

        # Check new connectivity
        new_tria = ordered_mesh.tria[0]
        p1, p2, p3 = coords[new_tria]

        # Calculate signed area
        area = 0.5 * (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
        self.assertGreater(area, 0, "Triangle should be ordered CCW (positive area)")

    def test_cleanup_skewed_elements(self):
        # Create a good triangle and a very thin (skewed) triangle
        coords = np.array([
            [0,0], [1,0], [0,1],       # Good
            [2,0], [3,0], [2.001, 0.1] # Bad (Very small angle)
        ])
        tria = np.array([[0,1,2], [3,4,5]])
        mesh = MeshData(coords=coords, tria=tria)

        cleaned = utils.cleanup_skewed_el(mesh, lw_bound_tri=10) # Min angle 10 deg

        self.assertEqual(len(cleaned.tria), 1)
        # Should keep the first one
        np.testing.assert_array_equal(cleaned.tria[0], [0,1,2])

    def test_quads_from_tri(self):
        # Two triangles forming a square
        # 3 -- 2
        # | \  |
        # |  \ |
        # 0 -- 1
        coords = np.array([[0,0], [1,0], [1,1], [0,1]])
        tria = np.array([
            [0, 1, 2], # Bottom-Right
            [0, 2, 3]  # Top-Left
        ])
        mesh = MeshData(coords=coords, tria=tria)

        quad_mesh = utils.quads_from_tri(mesh)

        self.assertEqual(len(quad_mesh.tria), 0)
        self.assertEqual(len(quad_mesh.quad), 1)
        # Sort indices to compare connectivity set
        self.assertTrue(set(quad_mesh.quad[0]) == {0,1,2,3})

    def test_cleanup_concave_quads(self):
        # Create a "boomerang" shape quad (concave)
        # 3     2
        #  \   /
        #    1
        #    |
        #    0
        coords = np.array([[0,0], [0,1], [1,2], [-1,2]])
        # Connect 0-2-1-3 ? No standard quad cycle
        # Let's define cycle: 0 -> 2 -> 1 -> 3 (Crosses itself or concave)
        # Let's try 0 -> 1 -> 2 -> 3? 
        # 0(0,0), 1(0,1), 2(1,2), 3(-1,2)
        # Hull area of these points is triangle 0,2,3 -> area != quad area if concave
        quad = np.array([[0, 2, 1, 3]]) 

        mesh = MeshData(coords=coords, quad=quad)

        cleaned = utils.cleanup_concave_quads(mesh)
        self.assertEqual(len(cleaned.quad), 0)


class TestProjections(unittest.TestCase):
    def test_reproject(self):
        coords = np.array([[-75.0, 40.0]])
        mesh = MeshData(coords=coords, crs=CRS.from_epsg(4326))
        utils.reproject(mesh, "EPSG:3857")
        self.assertEqual(mesh.crs, CRS.from_epsg(3857))
        self.assertGreater(abs(mesh.coords[0,0]), 1000)

    def test_estimate_mesh_utm(self):
        coords = np.array([[-75.0, 40.0], [-74.0, 41.0]])
        mesh = MeshData(coords=coords, crs=CRS.from_epsg(4326))

        utm_crs = utils.estimate_mesh_utm(mesh)
        self.assertIsNotNone(utm_crs)

        # Zone 18N (EPSG 32618)
        self.assertTrue(utm_crs.to_string().upper().count("18") > 0 or "ZONE=18" in utm_crs.to_wkt().upper())

    def test_project_to_utm(self):
        coords = np.array([[-75.0, 40.0]])
        mesh = MeshData(coords=coords, crs=CRS.from_epsg(4326))
        utils.project_to_utm(mesh)
        self.assertTrue(mesh.crs.is_projected)


class TestGeometricResampling(unittest.TestCase):

    def test_resample_geom_by_hfun(self):
        """Test the complex resampling logic using a KDTree lookup"""
        # 1. Geometry: A 10x10 square at (0,0)
        square = box(0, 0, 10, 10)
        shape_series = gpd.GeoSeries([square], crs="EPSG:4326")

        # 2. Hfun: A mesh covering the area with constant value 1.0
        # If Hfun is 1.0, and perimeter is 40, we expect roughly 40 points
        h_coords = np.array([
            [-1, -1], [11, -1], [11, 11], [-1, 11]
        ])
        h_vals = np.array([1.0, 1.0, 1.0, 1.0])
        hfun_mesh = MeshData(coords=h_coords, values=h_vals, crs="EPSG:4326")

        # 3. Run Resample
        resampled_series = utils.resample_geom_by_hfun(shape_series, hfun_mesh)

        resampled_poly = resampled_series.iloc[0]

        # Check it is still a polygon
        self.assertIsInstance(resampled_poly, Polygon)

        # Check number of vertices
        # Perimeter 40 / size 1.0 = 40 segments + 1 closing point = 41 points
        # The algorithm walks, so it might vary slightly depending on interpolation logic
        n_pts = len(resampled_poly.exterior.coords)
        self.assertTrue(35 < n_pts < 45, f"Expected ~41 points, got {n_pts}")

class TestClipping(unittest.TestCase):
    def setUp(self):
        self.mesh = utils.create_rectangle_mesh(
            nx=5, ny=5,
            holes=[],
            x_extent=(0, 4),
            y_extent=(0, 4)
        )
        # Mesh has elements of size 1.0 x 1.0
        # Total elements = 16 quads = 32 triangles

    def test_clip_mesh_by_shape_box(self):
        # Make the box STRICTLY larger than the first element (0,0 -> 1,1)
        # Box from -0.1 to 1.1 covers the 0-1 element completely with margin
        clip_box = box(-0.1, -0.1, 1.1, 1.1)

        clipped = utils.clip_mesh_by_shape(self.mesh, clip_box, fit_inside=True)

        # Should strictly have fewer elements than original
        self.assertLess(get_num_elements(clipped), get_num_elements(self.mesh))
        # Should have at least the one element we captured
        self.assertGreater(get_num_elements(clipped), 0)

    def test_clip_mesh_inverse(self):
        # Punch a hole clearly in the middle of an element (centroid)
        # Element at (0,0)-(1,1) has center (0.5, 0.5)
        hole = box(-0.1, -0.1, 1.1, 1.1)

        punched = utils.clip_mesh_by_shape(self.mesh, hole, inverse=True)

        # This shoulddestroy at least the 2 triangles in that first square
        self.assertLess(get_num_elements(punched), get_num_elements(self.mesh))


class TestGraphHelpers(unittest.TestCase):

    def test_get_boundary_edges(self):
        # Single triangle (0,0), (1,0), (0,1)
        coords = np.array([[0,0], [1,0], [0,1]])
        tria = np.array([[0,1,2]])
        mesh = MeshData(coords=coords, tria=tria)

        edges = utils.get_boundary_edges(mesh)
        # All 3 edges are boundary
        self.assertEqual(len(edges), 3)

        # Two triangles sharing an edge
        coords2 = np.array([[0,0], [1,0], [0,1], [1,1]])
        tria2 = np.array([[0,1,2], [1,3,2]]) # shared edge 1-2
        mesh2 = MeshData(coords=coords2, tria=tria2)

        edges2 = utils.get_boundary_edges(mesh2)
        # Total edges 5. Shared 1. Boundary 4.
        self.assertEqual(len(edges2), 4)



if __name__ == '__main__':
    unittest.main()
