import os
from typing import Union

from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
from shapely import ops
from shapely.geometry import (
    Polygon, MultiPolygon, LinearRing)

from geomesh.geom.base import BaseGeom
from geomesh.raster import Raster


class SourceRaster:
    '''Descriptor class used for referencing a :class:`geomesh.Raster`
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

            else:
                fig, ax = plt.subplots()
                ax.contourf(x, y, new_mask, levels=[0, 1])
                plt.close(fig)
                polygon_collection.extend(
                    [polygon for polygon in get_multipolygon_from_axes(ax)])
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

    def triplot(
        self,
        show=False,
        linewidth=0.07,
        color='black',
        alpha=0.5,
        **kwargs
    ):
        plt.triplot(
            self.triangulation,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            **kwargs
            )
        if show:
            plt.axis('scaled')
            plt.show()


def get_multipolygon_from_axes(ax):
    # extract linear_rings from plot
    linear_ring_collection = list()
    for path_collection in ax.collections:
        for path in path_collection.get_paths():
            polygons = path.to_polygons(closed_only=True)
            for linear_ring in polygons:
                if linear_ring.shape[0] > 3:
                    linear_ring_collection.append(
                        LinearRing(linear_ring))
    if len(linear_ring_collection) > 1:
        # reorder linear rings from above
        areas = [Polygon(linear_ring).area
                 for linear_ring in linear_ring_collection]
        idx = np.where(areas == np.max(areas))[0][0]
        polygon_collection = list()
        outer_ring = linear_ring_collection.pop(idx)
        path = Path(np.asarray(outer_ring.coords), closed=True)
        while len(linear_ring_collection) > 0:
            inner_rings = list()
            for i, linear_ring in reversed(
                    list(enumerate(linear_ring_collection))):
                xy = np.asarray(linear_ring.coords)[0, :]
                if path.contains_point(xy):
                    inner_rings.append(linear_ring_collection.pop(i))
            polygon_collection.append(Polygon(outer_ring, inner_rings))
            if len(linear_ring_collection) > 0:
                areas = [Polygon(linear_ring).area
                         for linear_ring in linear_ring_collection]
                idx = np.where(areas == np.max(areas))[0][0]
                outer_ring = linear_ring_collection.pop(idx)
                path = Path(np.asarray(outer_ring.coords), closed=True)
        multipolygon = MultiPolygon(polygon_collection)
    else:
        multipolygon = MultiPolygon(
            [Polygon(linear_ring_collection.pop())])
    return multipolygon
