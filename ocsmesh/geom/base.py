"""This module defines the base class for all geometry (domain) types
"""

from abc import ABC, abstractmethod
from typing import Any, Union

import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon

from ocsmesh.crs import CRS as CRSDescriptor
from ocsmesh import utils


class BaseGeom(ABC):
    """Abstract base class used to construct OCSMesh "geom" objects.

    This abstract class defines the interface that all geometry
    objects are expected to have in OCSMesh. All geometry type
    classes need to inherit this base class.

    Attributes
    ----------
    crs : CRSDescriptor
        Representing the CRS of the geometry's underlying object.
    multipolygon : MultiPolygon
        Lazily calculated `shapely` (multi)polygon of the geometry

    Methods
    -------
    get_multipolygon(**kwargs)
        Returns `shapely` object representation of the geometry

    Notes
    -----
    A "geom" object represents the domain of meshing (i.e.
    simulation). This domain can be represented either as a `shapely`
    `MultiPolygon` object
    """

    _crs = CRSDescriptor()

    def __init__(self, crs: Union[CRS, str, int]) -> None:
        self._crs = crs

    @property
    def multipolygon(self) -> MultiPolygon:
        """Read-only attribute for `shapely` representation of the geometry

        This read-only attribute is calculated lazily and has the same
        value as calling `get_multipolygon` without any arguments.
        """

        return self.get_multipolygon()

    def msh_t(self, **kwargs: Any) -> 'jigsaw_msh_t':
        raise NotImplementedError(
            "Deprecated for new internal mesh structure and multiple mesh engine support!"
        )


    def geoseries(self, **kwargs: Any) -> gpd.GeoSeries
        gs = gpd.GeoSeries(get_multipolygon(**kwargs), crs=self.crs)
        utm_crs = utils.estimate_bounds_utm(gs.total_bounds, gs.crs)
        if utm_crs is not None:
            gs = gs.to_crs(utm_crs)

        return gs


    @abstractmethod
    def get_multipolygon(self, **kwargs: Any) -> MultiPolygon:
        """Calculate the `shapely` representation of the geometry

        This abstract method defines the expected API for a geometry
        object.

        Raises
        ------
        NotImplementedError
            If method is not redefined in the subclass
        """

        raise NotImplementedError

    @property
    def crs(self) -> CRS:
        """Read-only attribute for CRS of the input geometry"""
        return self._crs
