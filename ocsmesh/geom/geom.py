from shapely.geometry import MultiPolygon, Polygon  # type: ignore[import]

from ocsmesh.geom.base import BaseGeom
from ocsmesh.geom.collector import GeomCollector
from ocsmesh.geom.mesh import MeshGeom
from ocsmesh.geom.raster import RasterGeom
from ocsmesh.geom.shapely import MultiPolygonGeom, PolygonGeom
from ocsmesh.mesh.base import BaseMesh
from ocsmesh.raster import Raster


class Geom(BaseGeom):
    """
    Factory class that creates and returns correct object type
    based on the input type
    """

    def __new__(cls, geom, **kwargs):
        """
        Input parameters
        ----------------
        geom:
            Object to use as input to compute the output mesh hull.
        """

        if isinstance(geom, Raster):  # pylint: disable=R1705
            return RasterGeom(geom, **kwargs)

        elif isinstance(geom, BaseMesh):
            return MeshGeom(geom, **kwargs)

        elif isinstance(geom, Polygon):
            return PolygonGeom(geom, **kwargs)

        elif isinstance(geom, MultiPolygon):
            return MultiPolygonGeom(geom, **kwargs)

        elif isinstance(geom, (list, tuple)):
            return GeomCollector(geom, **kwargs)

        raise TypeError(
            f"Argument geom must be of type {BaseGeom} or a derived type, "
            f"not type {type(geom)}."
        )

    @staticmethod
    def is_valid_type(geom):
        return isinstance(geom, BaseGeom)

    def get_multipolygon(self, **kwargs) -> MultiPolygon:
        raise NotImplementedError
