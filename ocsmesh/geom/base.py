"""This module defines the base class for all geometry (domain) types
"""

from abc import ABC, abstractmethod
from typing import Any, Union

import geopandas as gpd
from pyproj import CRS
import shapely
from shapely.geometry import MultiPolygon

from ocsmesh.crs import CRS as CRSDescriptor
from ocsmesh import utils, geodesy


def geoseries_metric_crs(gs: gpd.GeoSeries, metric_crs='auto'):
    """Metric CRS of the shapes of ``gs`` (see ``ocsmesh.geodesy``):
    an explicit ``metric_crs``, else chosen from their vertices when
    ``gs`` is geographic; None when ``gs`` has no CRS or is metric."""

    crs_explicit = geodesy.explicit(metric_crs)
    if crs_explicit is not None:
        return crs_explicit
    if gs.crs is None or not gs.crs.is_geographic or gs.is_empty.all():
        return None
    xy = shapely.get_coordinates(gs.values)
    x0, y0, x1, y1 = gs.total_bounds
    if x1 - x0 > 180:
        x0, x1 = geodesy.lon_arc(xy[:, 0])
    utm_crs = utils.estimate_bounds_utm((x0, y0, x1, y1), gs.crs)
    return geodesy.pick_metric_crs(
        xy[:, 0], xy[:, 1], metric_crs, utm_crs=utm_crs, geo_crs=gs.crs)


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

    def msh_t(self, **kwargs: Any) -> 'MeshData':
        raise NotImplementedError(
            "Deprecated for new internal mesh structure and multiple mesh engine support!"
        )


    def geoseries(self, metric_crs: Union[str, CRS, None] = 'auto',
                  **kwargs: Any) -> gpd.GeoSeries:
        """The geometry as a GeoSeries in its metric CRS.

        ``metric_crs`` ('auto', 'utm' or a CRS, see
        ``ocsmesh.geodesy``) is chosen from the polygon vertices when
        the geometry is geographic; other keyword arguments go to
        ``get_multipolygon``.
        """
        gs = gpd.GeoSeries(self.get_multipolygon(**kwargs), crs=self.crs)
        target = geoseries_metric_crs(gs, metric_crs)
        if target is not None and not target.equals(gs.crs):
            gs = gs.to_crs(target)

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
