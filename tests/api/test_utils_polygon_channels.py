"""utils.get_polygon_channels with single Polygon results.

The overlay inside can return a Polygon row instead of a MultiPolygon
(e.g. a single creek left in a small tile); iterating ``.geoms`` on it
raised AttributeError and failed a whole channel detection run.
"""

import unittest
from unittest import mock

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, box

from ocsmesh import utils


class TestPolygonChannels(unittest.TestCase):

    def setUp(self):
        # a 10 x 10 lake with a 1 m wide, 10 m long creek
        self.creek = box(10, 4, 20, 5)
        self.poly = box(0, 0, 10, 10).union(self.creek)

    def test_multipolygon_overlay(self):
        out = utils.get_polygon_channels(self.poly, width=2)
        self.assertIsInstance(out, MultiPolygon)
        self.assertGreaterEqual(out.area, self.creek.area)

    def test_polygon_overlay(self):
        def single_polygon(*_args, **_kwargs):
            return gpd.GeoDataFrame(geometry=gpd.GeoSeries([self.creek]))
        with mock.patch.object(utils.gpd, 'overlay', single_polygon):
            out = utils.get_polygon_channels(self.poly, width=2)
        self.assertIsInstance(out, MultiPolygon)
        self.assertAlmostEqual(out.area, self.creek.area)
        self.assertTrue(out.equals(MultiPolygon([self.creek])))

    def test_small_polygon_is_dropped(self):
        tiny = Polygon([(0, 0), (0.5, 0), (0.5, 0.5)])
        def tiny_only(*_args, **_kwargs):
            return gpd.GeoDataFrame(geometry=gpd.GeoSeries([tiny]))
        with mock.patch.object(utils.gpd, 'overlay', tiny_only):
            self.assertIsNone(utils.get_polygon_channels(self.poly, width=2))


if __name__ == '__main__':
    unittest.main()
