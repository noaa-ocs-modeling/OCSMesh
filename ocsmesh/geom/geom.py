"""This module defines geometry factory class

Instead of importing specific geometry types, users can import
factory `Geom` object from this module which creates the correct
geometry type based on the input arguments passed to it
"""
from typing import Union, Any

from shapely.geometry import Polygon, MultiPolygon  # type: ignore[import]

from ocsmesh.raster import Raster
from ocsmesh.mesh.base import BaseMesh
from ocsmesh.geom.base import BaseGeom
from ocsmesh.geom.raster import RasterGeom
from ocsmesh.geom.mesh import MeshGeom
from ocsmesh.geom.shapely import PolygonGeom, MultiPolygonGeom
from ocsmesh.geom.collector import GeomCollector, CanCreateGeom

GeomType = Union[
        RasterGeom,
        MeshGeom,
        PolygonGeom,
        MultiPolygonGeom,
        GeomCollector
        ]

class Geom(BaseGeom):
    """Geometry object factory

    Factory class that creates and returns concrete geometry object
    based on the input types.

    Methods
    -------
    is_valid_type(geom)
        Static method to check if an object is a valid geometry type

    Notes
    -----
    The object created when using this class is not an instance of
    this class or a subclass of it. It is a subclass of BaseGeom
    instead.
    """

    def __new__(
            cls,
            geom: CanCreateGeom,
            **kwargs: Any
            ) -> GeomType:
        """
        Parameters
        ----------
        geom : CanCreateGeom
            Object to create the domain geometry from. The type of
            this object determines the created geometry object type
        **kwargs : dict, optional
            Keyword arguments passed to the constructor of the
            correct geometry type
        """

        if isinstance(geom, Raster): # pylint: disable=R1705
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
            f'Argument geom must be of type {BaseGeom} or a derived type, '
            f'not type {type(geom)}.')

    @staticmethod
    def is_valid_type(geom: Any) -> bool:
        """Checks if an object is a valid geometry type"""
        return isinstance(geom, BaseGeom)

    def get_multipolygon(self, **kwargs: Any) -> MultiPolygon:

        # FIXME: Need to override the superclass method here to avoid
        # instantiation of abstract class error
        raise NotImplementedError
