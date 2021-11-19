import os
from typing import Union

import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon

from ocsmesh.geom.base import BaseGeom
from ocsmesh.raster import Raster


class SourceRaster:
    """Descriptor class used for referencing a :class:`ocsmesh.Raster`
    object."""

    def __set__(self, obj, val: Union[Raster, str, os.PathLike]):

        if isinstance(val, (str, os.PathLike)):  # type: ignore[misc]
            val = Raster(val)

        if not isinstance(val, Raster):
            raise TypeError(
                f"Argument raster must be of type {Raster}, " f"not type {type(val)}."
            )
        obj.__dict__["source_raster"] = val

    def __get__(self, obj, val):
        return obj.__dict__["source_raster"]


class RasterGeom(BaseGeom):

    _source_raster = SourceRaster()

    def __init__(
        self,
        raster: Union[Raster, str, os.PathLike],
        zmin=None,
        zmax=None,
    ):
        """
        Input parameters
        ----------------
        raster:
            Input object used to compute the output mesh hull.
        """
        self._source_raster = raster
        self._zmin = zmin
        self._zmax = zmax

    def get_multipolygon(  # type: ignore[override]
        self, zmin: float = None, zmax: float = None
    ) -> MultiPolygon:
        """Returns the shapely.geometry.MultiPolygon object that represents
        the hull of the raster given optional zmin and zmax contraints.
        """
        zmin = self._zmin if zmin is None else zmin
        zmax = self._zmax if zmax is None else zmax

        if zmin is None and zmax is None:
            return MultiPolygon([self.raster.get_bbox()])

        return self.raster.get_multipolygon(zmin=zmin, zmax=zmax)

    @property
    def raster(self):
        return self._source_raster

    @property
    def crs(self):
        return self.raster.crs

    def make_plot(self, ax=None, show=False):

        # TODO: Consider the ellipsoidal case. Refer to commit
        # dd087257c15692dd7d8c8e201d251ab5e66ff67f on main branch for
        # ellipsoidal ploting routing (removed).
        for polygon in self.multipolygon:
            plt.plot(*polygon.exterior.xy, color="k")
            for interior in polygon.interiors:
                plt.plot(*interior.xy, color="r")
        if show:
            plt.show()

        return plt.gca()
