"""This module defines raster based geometry class
"""

import os
from typing import Union, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from shapely.geometry import MultiPolygon
from pyproj import CRS

from ocsmesh.geom.base import BaseGeom
from ocsmesh.raster import Raster


class SourceRaster:
    """Descriptor class used for referencing the source `Raster` object."""

    def __set__(self, obj, val: Union[Raster, str, os.PathLike]):

        if isinstance(val, (str, os.PathLike)):  # type: ignore[misc]
            val = Raster(val)

        if not isinstance(val, Raster):
            raise TypeError(
                f'Argument raster must be of type {Raster}, '
                f'not type {type(val)}.')
        obj.__dict__['source_raster'] = val

    def __get__(self, obj, objtype=None) -> Raster:
        return obj.__dict__['source_raster']


class RasterGeom(BaseGeom):
    """Raster based geometry.

    Create geometry based on a raster object. All the calculations
    are done on the raster image.

    Attributes
    ----------
    raster
        Reference to the source raster object
    crs
        CRS of the input raster

    Methods
    -------
    msh_t(**kwargs)
        Returns the `jigsawpy` vertex-edge representation of the geometry
    get_multipolygon(zmin=None, zmax=None)
        Returns `shapely` object representation of the geometry

    Notes
    -----
    This is the main object to use for handling DEM data. The objects
    of this class hold a reference to the original raster object and
    use that handle to extract data using `Raster` API.
    """

    _source_raster = SourceRaster()

    def __init__(
            self,
            raster: Union[Raster, str, os.PathLike],
            zmin: Optional[float] = None,
            zmax: Optional[float] = None,
        ) -> None:
        """Initialize a raster based geometry object.

        Parameters
        ----------
        raster : Raster or str or path-like
            Input raster for generating the geometry, i.e. the
            simulation domain
        zmin : float or None, default=None
            Minimum elevation cut-off for calculating polygon
        zmax : float or None, default=None
            Maximum elevation cut-off for calculating polygon
        """

        self._source_raster = raster
        self._zmin = zmin
        self._zmax = zmax

    def get_multipolygon(  # type: ignore[override]
            self,
            zmin: Optional[float] = None,
            zmax: Optional[float] = None
            ) -> MultiPolygon:
        """Returns the `shapely` representation of the geometry

        Calculates and returns the `MultiPolygon` representation of
        the geometry.

        Parameters
        ----------
        zmin : float or None, default=None
            Minimum elevation cut-off for calculating polygon, overrides
            value provided during object construction
        zmax : float or None, default=None
            Maximum elevation cut-off for calculating polygon, overrides
            value provided during object construction

        Returns
        -------
        MultiPolygon
            Calculated polygon from raster data based on the minimum
            and maximum elevations of interest.
        """

        zmin = self._zmin if zmin is None else zmin
        zmax = self._zmax if zmax is None else zmax

        if zmin is None and zmax is None:
            return MultiPolygon([self.raster.get_bbox()])

        return self.raster.get_multipolygon(zmin=zmin, zmax=zmax)


    @property
    def raster(self) -> Raster:
        """Read-only attribute for reference to the source raster"""
        return self._source_raster

    @property
    def crs(self) -> CRS:
        """Read-only attribute returning the CRS of the source raster"""
        return self.raster.crs

    def make_plot(
            self,
            ax: Optional[Axes] = None,
            show: bool = False
            ) -> Axes:
        """Create a plot from the boundaries of the geometry polygon

        Parameters
        ----------
        ax : Axes or None, default=None
            An existing axes to draw the geomtry polygon on
        show : bool, default=False
            Whether to call `pyplot.show()` after drawing the polygon

        Returns
        -------
        Axes
            The axes onto which the polygon boundaries are drawn
        """

        # TODO: Consider the ellipsoidal case. Refer to commit
        # dd087257c15692dd7d8c8e201d251ab5e66ff67f on main branch for
        # ellipsoidal ploting routing (removed).
        for polygon in self.multipolygon:
            plt.plot(*polygon.exterior.xy, color='k')
            for interior in polygon.interiors:
                plt.plot(*interior.xy, color='r')
        if show:
            plt.show()

        return plt.gca()
