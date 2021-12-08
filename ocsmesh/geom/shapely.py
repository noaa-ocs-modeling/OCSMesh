"""This module defines `shapely` object based geometry
"""

from typing import Union, Any

from pyproj import CRS
from shapely.geometry import Polygon, MultiPolygon

from ocsmesh.geom.base import BaseGeom


class ShapelyGeom(BaseGeom):
    """Base class for geoms based on shapely objects"""


class PolygonGeom(ShapelyGeom):
    """Geometry based on `shapely.geometry.Polygon`.

    Attributes
    ----------
    polygon : Polygon
        Reference to the underlying `shapely` object.
    crs : crs
        CRS associated with the geometry during construction.

    Methods
    -------
    get_multipolygon(**kwargs)
        Returns `shapely` object representation of the geometry
    msh_t(**kwargs)
        Returns the `jigsawpy` vertex-edge representation of the geometry
    """

    def __init__(self, polygon: Polygon, crs: Union[CRS, str]) -> None:
        """Initialize a `shapely` based geometry object

        Parameters
        ----------
        polygon : Polygon
            Input polygon to be used for geometry object
        crs : CRS or str
            The CRS of the input `Polygon`
        """

        assert isinstance(polygon, Polygon)
        self._polygon = polygon
        super().__init__(crs)

    def get_multipolygon(self, **kwargs : Any) -> MultiPolygon:
        """Returns a `MultiPolygon` from the underlying `shapely` geometry

        Parameters
        ----------
        **kwargs : dict, optional
            Currently unused for this class, needed for generic API
            support

        Returns
        -------
        MultiPolygon
            Multipolygon created from the input single polygon
        """

        return MultiPolygon([self._polygon])

    @property
    def polygon(self):
        """Read-only attribute referencing the underlying `Polygon`"""
        return self._polygon

    @property
    def crs(self):
        """Read-only attribute returning the CRS associated with the geometry"""
        return self._crs


class MultiPolygonGeom(ShapelyGeom):

    def __init__(
            self,
            multipolygon: MultiPolygon,
            crs: Union[CRS, str]
            ) -> None:
        """Initialize a `shapely` based geometry object

        Parameters
        ----------
        multipolygon : MultiPolygon
            Input multipolygon to be used for geometry object
        crs : CRS or str
            The CRS of the input `Polygon`
        """

        assert isinstance(multipolygon, MultiPolygon)
        self._multipolygon = multipolygon
        super().__init__(crs)

    def get_multipolygon(self, **kwargs):
        """Returns the underlying `shapely` MultiPolygon

        Parameters
        ----------
        **kwargs : dict, optional
            Currently unused for this class, needed for generic API
            support

        Returns
        -------
        MultiPolygon
            Underlying `Multipolygon`
        """

        return self._multipolygon

    @property
    def multipolygon(self):
        """Read-only attribute referencing the underlying `MultiPolygon`"""
        return self._multipolygon
