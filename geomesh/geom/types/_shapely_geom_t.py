
from shapely.geometry import Polygon, MultiPolygon

from ._base import _BaseGeomType


class _ShapelyGeomType(_BaseGeomType):
    """ Base class for geoms based on shapely objects """


class _PolygonGeomType(_ShapelyGeomType):

    __slots__ = ["__polygon"]

    def __init__(self, polygon):
        self._polygon = polygon

    def get_multipolygon(self):
        return MultiPolygon([self._polygon])

    @property
    def _geom(self):
        return self.__polygon

    @property
    def _polygon(self):
        return self.__polygon

    @_polygon.setter
    def _polygon(self, polygon):
        assert isinstance(polygon, Polygon)
        self.__polygon = polygon


class _MultiPolygonGeomType(_ShapelyGeomType):

    __slots__ = ["__polygon"]

    def __init__(self, polygon):
        self._polygon = polygon

    def get_multipolygon(self):
        return self._multipolygon

    @property
    def _geom(self):
        return self.__multipolygon

    @property
    def _multipolygon(self):
        return self.__multipolygon

    @_multipolygon.setter
    def _multipolygon(self, multipolygon):
        assert isinstance(multipolygon, MultiPolygon)
        self.__multipolygon = multipolygon
