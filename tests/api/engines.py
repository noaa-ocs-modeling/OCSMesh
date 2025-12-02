#! python
import re
import unittest

import numpy as np
import geopandas as gpd
from shapely.geometry import (
    Point,
    LineString,
    box,
    Polygon,
    MultiPolygon,
)
from shapely.ops import polygonize

from ocsmesh import utils, MeshData
from ocsmesh.engines.factory import get_mesh_engine


class TriangulatePolygon(unittest.TestCase):


    def setUp(self):
        self.tr_noopts = get_mesh_engine('triangle')
        self.tr_popts = get_mesh_engine('triangle', opts='p')

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
        meshdata_1 = self.tr_noopts.generate(gpd.GeoSeries(self.valid_input_1))
        meshdata_2 = self.tr_noopts.generate(self.valid_input_2.geometry)
        meshdata_3 = self.tr_noopts.generate(self.valid_input_3)

        with self.assertRaises(ValueError) as exc_1:
            self.tr_noopts.generate(gpd.GeoSeries(self.invalid_input_1))

        with self.assertRaises(ValueError) as exc_2:
            self.tr_noopts.generate(gpd.GeoSeries(self.invalid_input_2))

        with self.assertRaises(ValueError) as exc_3:
            self.tr_noopts.generate(gpd.GeoSeries(self.invalid_input_3))

        self.assertIsInstance(meshdata_1, MeshData)
        self.assertIsInstance(meshdata_2, MeshData)
        self.assertIsInstance(meshdata_3, MeshData)
        
        self.assertTrue(
            np.all(meshdata_1.coords == meshdata_2.coords)
            & np.all(meshdata_2.coords == meshdata_3.coords)
        )
        self.assertTrue(
            np.all(meshdata_1.tria == meshdata_2.tria)
            & np.all(meshdata_2.tria == meshdata_3.tria)
        )
        self.assertTrue(
            np.all(meshdata_1.values == meshdata_2.values)
            & np.all(meshdata_2.values == meshdata_3.values)
        )

        self.assertEqual(len(meshdata_1.quad), 0)
        self.assertEqual(len(meshdata_1.coords), len(meshdata_1.values))
        self.assertTrue((meshdata_1.values == 0).all())

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
        meshdata = self.tr_noopts.generate(gpd.GeoSeries(bx))
        bdry_lines = utils.get_boundary_segments(meshdata)

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
        aux_1 = MeshData(coords=[[1, 0.5], [2, 0.5], [3, 0.5]])
        aux_2 = MeshData(coords=[*aux_1.coords, [10, 0.5]]) # Out of domain points

        meshdata_1 = self.tr_popts.generate(gpd.GeoSeries(bx), seed=aux_1)
        meshdata_2 = self.tr_popts.generate(gpd.GeoSeries(bx), seed=aux_2)

        self.assertTrue(
            np.all([
                np.any([pt == v.tolist() for v in meshdata_1.coords])
                for pt in aux_1.coords
            ])
        )
        # Out of domain points are discarded
        self.assertFalse(
            np.all([
                np.any([pt == v.tolist() for v in meshdata_2.coords])
                for pt in aux_2.coords
            ])
        )


    def test_polygon_holes(self):
        poly = Polygon(
            [[0, 0], [4, 0], [4, 4], [0, 4]],
            [[[1, 1], [2, 0], [2, 2], [1, 2]]]
        )
        meshdata = self.tr_popts.generate(gpd.GeoSeries(poly))
        mesh_poly = utils.get_mesh_polygons(meshdata)

        self.assertTrue(poly.equals(mesh_poly))


    def test_multipolygon(self):
        mpoly = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])

        meshdata = self.tr_popts.generate(gpd.GeoSeries(mpoly))
        mesh_poly = utils.get_mesh_polygons(meshdata)

        self.assertTrue(mpoly.equals(mesh_poly))


    def test_polygons_touching_two_points_no_hole(self):
        poly1 = Polygon(
            [[0, 0], [0, 4], [6, 4], [6, 0], [4, 2], [2, 2], [0, 0]],
        )
        poly2 = Polygon(
            [[0, 0], [0, -4], [6, -4], [6, 0], [4, -2], [2, -2], [0, 0]],
        )
        multpoly = MultiPolygon([poly1, poly2])
        meshdata = self.tr_popts.generate(gpd.GeoSeries(multpoly))
        mesh_poly = utils.get_mesh_polygons(meshdata)

        self.assertTrue(multpoly.equals(mesh_poly))


