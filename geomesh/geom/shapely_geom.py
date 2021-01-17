from shapely.geometry import Polygon, MultiPolygon

from geomesh.geom.base import BaseGeom


class ShapelyGeom(BaseGeom):
    """ Base class for geoms based on shapely objects """


class PolygonGeom(ShapelyGeom):


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


class MultiPolygonGeom(ShapelyGeom):


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
