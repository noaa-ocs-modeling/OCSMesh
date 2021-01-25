from shapely.geometry import Polygon, MultiPolygon

from geomesh.geom.base import BaseGeom


class ShapelyGeom(BaseGeom):
    """ Base class for geoms based on shapely objects """


class PolygonGeom(ShapelyGeom):

    def __init__(self, polygon: Polygon):
        assert isinstance(polygon, Polygon)
        self._polygon = polygon

    def get_multipolygon(self):
        return MultiPolygon([self._polygon])

    @property
    def polygon(self):
        return self._polygon


class MultiPolygonGeom(ShapelyGeom):

    def __init__(self, multipolygon: MultiPolygon):
        assert isinstance(multipolygon, MultiPolygon)
        self._multipolygon = multipolygon

    def get_multipolygon(self):
        return self._multipolygon

    @property
    def multipolygon(self):
        return self._multipolygon
