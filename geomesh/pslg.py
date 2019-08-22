import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from netCDF4 import Dataset
from osgeo import osr, gdal, ogr
from geomesh import gdal_tools

pyenv_prefix = "/".join(sys.executable.split('/')[:-2])
if os.getenv('SRTM15_PATH') is not None:
    nc = pathlib.Path(os.getenv('SRTM15_PATH'))
else:
    nc = pathlib.Path(pyenv_prefix + '/lib/SRTM15+V2.nc')


class PlanarStraightLineGraph:

    def __init__(self, SpatialReference=3395):
        self.__container = {'gdal_strings': list(),
                            'multipolygons': list()}
        self.__nc = Dataset(nc)
        self.__MultiPolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        self._SpatialReference = SpatialReference

    def make_plot(self, show=False):
        for _MultiPolygon in self.MultiPolygons:
            for _Polygon in _MultiPolygon:
                for _LinearRing in _Polygon:
                    array = np.asarray(_LinearRing.GetPoints())
                    plt.plot(array[:, 0], array[:, 1])
        plt.gca().axis('scaled')
        plt.show()

    def add_Dataset(self, Dataset):
        self.__container['gdal_strings'].append(Dataset)
        self.__container['multipolygons'].append(
            self.__get_MultiPolygon_from_Dataset(Dataset))

    def __get_MultiPolygon_from_Dataset(self, Dataset):
        # semi_perimeter = 3*self.h0/np.sqrt(2.)
        # area = np.sqrt(3*semi_perimeter*(semi_perimeter-self.h0))
        ds = gdal.Open(Dataset)
        ds = gdal.Warp(
            '', ds, dstSRS=self.SpatialReference, format='VRT',
            xRes=0.5*self.h0, yRes=0.5*self.h0)
        x, y, z = gdal_tools.get_arrays(ds)
        _Polygon = ogr.Geometry(ogr.wkbPolygon)
        _Polygon.AssignSpatialReference(self.SpatialReference)
        # return empty polygon if tile is fully internal/external
        if np.all(np.logical_and(z >= self.zmin, z <= self.zmax)) \
                or (np.all(z >= self.zmax) or np.all(z <= self.zmin)):
            return _Polygon
        # compute polygon using matplotlib
        ax = plt.contourf(x, y, z, levels=[self.zmin, self.zmax])
        plt.close(plt.gcf())
        _x, _y = np.meshgrid(x, y)
        xy = np.vstack([_x.flatten(), _y.flatten()]).T
        _z = z.flatten()
        # the output of contourf is a multipolygon.
        # clean up the secondary polygons to yield only the principal one.
        paths = list()
        for PathCollection in ax.collections:
            for path in PathCollection.get_paths():
                polygons = path.to_polygons(closed_only=True)
                for i, polygon in enumerate(polygons):
                    # xmin = np.min(polygon[:, 0])
                    # xmax = np.max(polygon[:, 0])
                    # ymin = np.min(polygon[:, 1])
                    # ymax = np.max(polygon[:, 1])
                    # xbool = np.logical_and(xy[:, 0] >= xmin,
                    #                        xy[:, 0] <= xmax)
                    # ybool = np.logical_and(xy[:, 1] >= ymin,
                    #                        xy[:, 1] <= ymax)
                    # _idx = np.where(np.logical_and(xbool, ybool))
                    _path = Path(polygon, closed=True)
                    # idx = np.where(_path.contains_points(xy[_idx]))
                    # if np.all(np.logical_and(
                    #         _z[_idx][idx] > 0.,
                    #         _z[_idx][idx] < self.zmax)):
                    #     pass
                    # elif np.all(np.logical_and(
                    #         _z[_idx][idx] < 0.,
                    #         _z[_idx][idx] > self.zmin)):
                    #     pass
                    # elif not np.any(_z[_idx][idx] >= 0.):
                    #     pass
                    # else:
                    #     paths.append(_path)

                    paths.append(_path)
        # now we need to sort into separate polygons/holes.
        rings = list()
        for path in paths:
            _LinearRing = ogr.Geometry(ogr.wkbLinearRing)
            _LinearRing.AssignSpatialReference(self.SpatialReference)
            for x, y in path.vertices:
                _LinearRing.AddPoint_2D(x, y)
            _LinearRing.CloseRings()
            rings.append(_LinearRing)
        areas = [_.GetArea() for _ in rings]
        multipolygon = list()
        while len(rings) > 0:
            _idx = np.where(np.max(areas) == areas)[0]
            path = paths[_idx[0]]
            _idxs = np.where([path.contains_point(_.vertices[0, :])
                              for _ in paths])[0]
            # try:
            #     if _idxs[0] != _idx[0]:
            _idxs = np.hstack([_idx, _idxs])
            # except IndexError:
            #     break
            polygon = list()
            for _idx in np.flip(_idxs):
                polygon.insert(0, rings.pop(_idx))
                paths.pop(_idx)
                areas.pop(_idx)
            multipolygon.append(polygon)
        _MultiPolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        _MultiPolygon.AssignSpatialReference(self.SpatialReference)
        for _LinearRings in multipolygon:
            _Polygon = ogr.Geometry(ogr.wkbPolygon)
            _Polygon.AssignSpatialReference(self.SpatialReference)
            for _LinearRing in _LinearRings:
                _Polygon.AddGeometry(_LinearRing)
            _MultiPolygon.AddGeometry(_Polygon)
        return _MultiPolygon

    @property
    def SpatialReference(self):
        return self.__MultiPolygon.GetSpatialReference()

    @property
    def MultiPolygon(self):
        return self.__MultiPolygon

    @property
    def MultiPolygons(self):
        for polygon in self.__container['multipolygons']:
            yield polygon

    @property
    def gdal_strings(self):
        for gdal_string in self.__container['gdal_strings']:
            yield gdal_string

    @property
    def h0(self):
        try:
            return self.__h0
        except AttributeError:
            raise AttributeError('Must set h0 attribute.')

    @property
    def zmin(self):
        try:
            return self.__zmin
        except AttributeError:
            raise AttributeError('Must set zmin attribute')

    @property
    def zmax(self):
        try:
            return self.__zmax
        except AttributeError:
            raise AttributeError('Must set zmax attribute')

    @property
    def ndim(self):
        return 2

    @property
    def mshID(self):
        return "euclidean-mesh"

    @property
    def vert2(self):
        vert2 = list()
        for _MultiPolygon in self.MultiPolygons:
            for _Polygon in _MultiPolygon:
                for _LinearRing in _Polygon:
                    vert2 = [*vert2, *_LinearRing.GetPoints()[:-1]]
        return vert2

    @property
    def edge2(self):
        i = 0
        edge2 = list()
        for _MultiPolygon in self.MultiPolygons:
            for _Polygon in _MultiPolygon:
                for _LinearRing in _Polygon:
                    _np = len(_LinearRing.GetPoints()[:-1])
                    for j in range(_np-1):
                        edge2.append((j+i, j+i+1))
                    edge2.append((edge2[-1][1], edge2[i][0]))
                    i = j+2
                for line in np.asarray(edge2):
                    print(line)
                BREAKME
        raise NotImplementedError

    @SpatialReference.setter
    def SpatialReference(self, SpatialReference):
        raise NotImplementedError

    @h0.setter
    def h0(self, h0):
        self.__h0 = float(h0)

    @zmin.setter
    def zmin(self, zmin):
        self.__zmin = float(zmin)

    @zmax.setter
    def zmax(self, zmax):
        self.__zmax = float(zmax)

    @property
    def _SpatialReference(self):
        return self.__SpatialReference

    @_SpatialReference.setter
    def _SpatialReference(self, SpatialReference):
        if isinstance(SpatialReference, int):
            EPSG = SpatialReference
            SpatialReference = osr.SpatialReference()
            SpatialReference.ImportFromEPSG(EPSG)
        try:
            self.__MultiPolygon.AssignSpatialReference(SpatialReference)
        except TypeError:
            raise TypeError(
                'Input must be an int corresponding to the espg code'
                + ' or osgeo.osr.SpatialReference instance.')


# @property
# def vert2(self):
#     vert2 = self.outer_vertices
#     for i, array in enumerate(self.inner_vertices):
#         vert2 = np.vstack([vert2, array])
#     return vert2

# @property
# def edge2(self):
#     edge2 = self.outer_wire_network_edges
#     i = edge2[-1, 0] + 1
#     for _i, _ in enumerate(self.inner_vertices):
#         edge2 = np.vstack([edge2, self.inner_wire_network_edges[_i]+i])
#         i = edge2[-1, 0] + 1
#     return edge2

# @property
# def outer_wire_network_edges(self):
#     if not hasattr(self, "__outer_wire_network_edges"):
#         outer_edges = list()
#         for i, _ in enumerate(self.outer_vertices):
#             if i != len(self.outer_vertices)-1:
#                 outer_edges.append((i, i+1))
#             else:
#                 outer_edges.append((i, 0))
#         self.__outer_wire_network_edges = np.asarray(outer_edges)
#     return self.__outer_wire_network_edges

# @property
# def inner_wire_network_edges(self):
#     if not hasattr(self, "__inner_wire_network_edges"):
#         inner_edges = list()
#         for inner_vertices in self.inner_vertices:
#             _inner_edges = list()
#             for i, _ in enumerate(inner_vertices):
#                 if i != len(inner_vertices)-1:
#                     _inner_edges.append((i, i+1))
#                 else:
#                     _inner_edges.append((i, 0))
#             inner_edges.append(np.asarray(_inner_edges))
#         self.__inner_wire_network_edges = inner_edges
#     return self.__inner_wire_network_edges

# @classmethod
#     def from_Dataset(cls, Dataset, zmin, zmax):
#         SpatialReference = gdal_tools.get_SpatialReference(Dataset)
#         if SpatialReference is None:
#             raise TypeError('Input Dataset must be spatially referenced.')
#         paths = []
#         if not isinstance(Dataset, gdal.Dataset):
#             raise TypeError(
#                 'Input must be of type {}'.format(gdal.Dataset))
#         x, y, z = gdal_tools.get_arrays(Dataset)
#         ax = plt.contourf(x, y, z, levels=[zmin, zmax])
#         plt.close(plt.gcf())
#         _x, _y = np.meshgrid(x, y)
#         xy = np.vstack([_x.flatten(), _y.flatten()]).T
#         _z = z.flatten()
#         # contour filtering
#         for PathCollection in ax.collections:
#             for Path in PathCollection.get_paths():
#                 polygons = Path.to_polygons(closed_only=True)
#                 if len(polygons) > 1:
#                     for i, polygon in enumerate(polygons):
#                         xmin = np.min(polygon[:, 0])
#                         xmax = np.max(polygon[:, 0])
#                         ymin = np.min(polygon[:, 1])
#                         ymax = np.max(polygon[:, 1])
#                         xbool = np.logical_and(xy[:, 0] >= xmin,
#                                                xy[:, 0] <= xmax)
#                         ybool = np.logical_and(xy[:, 1] >= ymin,
#                                                xy[:, 1] <= ymax)
#                         _idx = np.where(np.logical_and(xbool, ybool))
#                         path = _Path(polygon, closed=True)
#                         idx = np.where(path.contains_points(xy[_idx]))
#                         if i == 0:
#                             if np.any(_z[_idx][idx] >= 0):
#                                 paths.append(polygon)
#                                 _outer_elim = False
#                             else:
#                                 _outer_elim = True
#                         else:
#                             if not _outer_elim:
#                                 if np.all(_z[_idx][idx] >= zmax):
#                                     pass
#                                 elif np.all(_z[_idx][idx] <= zmin):
#                                     pass
#                                 else:
#                                     paths.append(polygon)
#         return cls(SpatialReference, *paths)