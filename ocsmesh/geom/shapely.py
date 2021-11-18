from typing import Union

from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon

from ocsmesh.geom.base import BaseGeom


class ShapelyGeom(BaseGeom):
    """ Base class for geoms based on shapely objects """


class PolygonGeom(ShapelyGeom):

    def __init__(self, polygon: Polygon, crs: Union[CRS, str]):
        assert isinstance(polygon, Polygon)
        self._polygon = polygon
        super().__init__(crs)

    def get_multipolygon(self, **kwargs):
        return MultiPolygon([self._polygon])

    @property
    def polygon(self):
        return self._polygon

    @property
    def crs(self):
        return self._crs


class MultiPolygonGeom(ShapelyGeom):

    def __init__(self, multipolygon: MultiPolygon, crs: Union[CRS, str]):
        assert isinstance(multipolygon, MultiPolygon)
        self._multipolygon = multipolygon
        super().__init__(crs)

    def get_multipolygon(self, **kwargs):
        return self._multipolygon

    @property
    def multipolygon(self):
        return self._multipolygon
