import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from shapely import ops
from shapely.geometry import Polygon, MultiPolygon

from ocsmesh.geom.base import BaseGeom
from ocsmesh.raster import Raster
from ocsmesh import utils


class SourceRaster:
    '''Descriptor class used for referencing a :class:`ocsmesh.Raster`
    object.'''

    def __set__(self, obj, val: Union[Raster, str, os.PathLike]):

        if isinstance(val, (str, os.PathLike)):  # type: ignore[misc]
            val = Raster(val)

        if not isinstance(val, Raster):
            raise TypeError(
                f'Argument raster must be of type {Raster}, '
                f'not type {type(val)}.')
        obj.__dict__['source_raster'] = val

    def __get__(self, obj, val):
        return obj.__dict__['source_raster']


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
            self, zmin: float = None, zmax: float = None) -> MultiPolygon:
        """Returns the shapely.geometry.MultiPolygon object that represents
        the hull of the raster given optional zmin and zmax contraints.
        """
        zmin = self._zmin if zmin is None else zmin
        zmax = self._zmax if zmax is None else zmax

        if zmin is None and zmax is None:
            return MultiPolygon([self.raster.get_bbox()])

        polygon_collection = []
        for window in self.raster.iter_windows():
            x, y, z = self.raster.get_window_data(window, band=1)
            new_mask = np.full(z.mask.shape, 0)
            new_mask[np.where(z.mask)] = -1
            new_mask[np.where(~z.mask)] = 1

            if zmin is not None:
                new_mask[np.where(z < zmin)] = -1

            if zmax is not None:
                new_mask[np.where(z > zmax)] = -1

            if np.all(new_mask == -1):  # or not new_mask.any():
                continue

            fig, ax = plt.subplots()
            ax.contourf(x, y, new_mask, levels=[0, 1])
            plt.close(fig)
            polygon_collection.extend(
                list(utils.get_multipolygon_from_pathplot(ax)))

        union_result = ops.unary_union(polygon_collection)
        if isinstance(union_result, Polygon):
            union_result = MultiPolygon([union_result])
        return union_result


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
            plt.plot(*polygon.exterior.xy, color='k')
            for interior in polygon.interiors:
                plt.plot(*interior.xy, color='r')
        if show:
            plt.show()

        return plt.gca()
