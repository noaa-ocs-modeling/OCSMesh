import os
from typing import Union

from matplotlib.path import Path  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import mpl_toolkits.mplot3d as m3d  # type: ignore[import]
import numpy as np  # type: ignore[import]
from shapely import ops  # type: ignore[import]
from shapely.geometry import (   # type: ignore[import]
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

    def __init__(self, raster: Union[Raster, str, os.PathLike]):
        """
        Input parameters
        ----------------
        raster:
            Input object used to compute the output mesh hull.
        """
        self._source_raster = raster

    def get_multipolygon(self, zmin=None, zmax=None):
        # TODO: Is it better to partition the raster in windows and parallelize
        # the computation, or is it better to do the windows in serial for all
        # use cases?
        # The current implementation uses serial processing over the raster
        # windows. This computation uses considerable amounts of memory.
        # Be aware of memory constraints if parallelizing this function.
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
                ax = plt.contourf(x, y, new_mask, levels=[0, 1])
                plt.close(plt.gcf())
                polygon_collection.extend(
                    [polygon for polygon in get_multipolygon_from_axes(ax)])
        return ops.unary_union(polygon_collection)

    @property
    def raster(self):
        return self._source_raster

    @property
    def crs(self):
        return self.raster.crs

    def make_plot(
        self,
        ax=None,
        show=False,
    ):

        # TODO: This function doesn't work due to disabling ellipsoid

        # spherical plot
        if self._ellipsoid is not None:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for polygon in self.multipolygon:
                coords = np.asarray(polygon.exterior.coords)
                x, y, z = self._geodetic_to_geocentric(
                    self._ellipsoids[self._ellipsoid.upper()],
                    coords[:, 1],
                    coords[:, 0],
                    0.
                    )
                ax.add_collection3d(
                    m3d.art3d.Line3DCollection([np.vstack([x, y, z]).T]),
                    )
        # planar plot
        else:
            for polygon in self.multipolygon:
                plt.plot(*polygon.exterior.xy, color='k')
                for interior in polygon.interiors:
                    plt.plot(*interior.xy, color='r')
        if show:
            if self._ellipsoid is None:
                plt.gca().axis('scaled')
            else:
                radius = self._ellipsoids[self._ellipsoid.upper()][0]
                # ax.set_aspect('equal')
                ax.set_xlim3d([-radius, radius])
                ax.set_xlabel("X")
                ax.set_ylim3d([-radius, radius])
                ax.set_ylabel("Y")
                ax.set_zlim3d([-radius, radius])
                ax.set_zlabel("Z")

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
