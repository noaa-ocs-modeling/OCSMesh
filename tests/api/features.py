#! python
import unittest
import tempfile
from pathlib import Path

import geopandas as gpd
from pyproj import CRS
from shapely import geometry

from ocsmesh.features.linefeature import LineFeature

class LineFeatureCapabilities(unittest.TestCase):

    def setUp(self):
        self._crs_1 = CRS.from_epsg(4326)
        self._crs_2 = CRS.from_string('nad83')
        self._lines_1 = geometry.LineString(
            [ 
                [0, 0], [1, 1], [2, 2]
            ]
        )
        self._lines_2 = geometry.LineString(
            [
                [1, 0], [2, 1]
            ]
        )
        self._mult_line_1 = geometry.MultiLineString(
            [self._lines_1, self._lines_2]
        )


    def test_ctor_args_1(self):
        lf = LineFeature(self._lines_1)
        shape, crs = lf.get_multiline()
        self.assertIsInstance(shape, geometry.MultiLineString)
        self.assertEqual(crs, self._crs_1)


    def test_ctor_args_2(self):
        lf = LineFeature(self._mult_line_1, self._crs_2)
        shape, crs = lf.get_multiline()
        self.assertIsInstance(shape, geometry.MultiLineString)
        self.assertEqual(crs, self._crs_2)


    def test_ctor_args_3(self):
        lf = LineFeature(shape=self._lines_1, shape_crs=self._crs_2)
        shape, crs = lf.get_multiline()
        self.assertIsInstance(shape, geometry.MultiLineString)
        self.assertEqual(crs, self._crs_2)


    def test_ctor_args_4(self):

        with tempfile.TemporaryDirectory() as tdir:

            shape_path = tdir

            gdf = gpd.GeoDataFrame(geometry=[self._mult_line_1], crs=self._crs_2)
            gdf.to_file(shape_path)

            lf = LineFeature(shapefile=shape_path)
            shape, crs = lf.get_multiline()
            self.assertIsInstance(shape, geometry.MultiLineString)
            self.assertEqual(crs, self._crs_2)


    def test_ctor_args_5(self):

        with tempfile.TemporaryDirectory() as tdir:

            # NOTE: Path is file this time
            shape_path = Path(tdir) / 'test_shape.shp'

            gdf = gpd.GeoDataFrame(geometry=[self._mult_line_1], crs=self._crs_2)
            gdf.to_file(shape_path)

            lf = LineFeature(shapefile=shape_path)
            shape, crs = lf.get_multiline()
            self.assertIsInstance(shape, geometry.MultiLineString)
            self.assertEqual(crs, self._crs_2)


    def test_ctor_args_6(self):

        with tempfile.TemporaryDirectory() as tdir:

            shape_path = tdir

            # NOTE: This time shapefile doesn't have CRS
            gdf = gpd.GeoDataFrame(geometry=[self._mult_line_1])
            gdf.to_file(shape_path)

            lf = LineFeature(shapefile=shape_path)
            with self.assertRaises(ValueError):
                lf.get_multiline()


if __name__ == '__main__':
    unittest.main()
