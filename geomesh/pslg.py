import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as _Path
from osgeo import osr, gdal, ogr
from geomesh._lib import _get_cache_dir
from geomesh import gdal_tools


class PlanarStraightLineGraph:

    def __init__(self, SpatialReference=3395):
        self.__container = {'gdal_strings': list(),
                            'polygons': list()}
        self.__MultiPolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        self._SpatialReference = SpatialReference

    def make_plot(self, show=False):
        plt.plot(self.outer_vertices[:, 0], self.outer_vertices[:, 1])
        for vertices in self.inner_vertices:
            plt.scatter(vertices[:, 0], vertices[:, 1])
        if show:
            plt.show()
        return plt.gca()

    def generate(self):
        # This function is a prototype and does two things: calculates the
        # contours and also cleans up the result. These two operations should
        # probably be separated in the future.
        res = self.h0/np.sqrt(2.)
        for ds in self.__Datasets:
            ds = self.__open_ogr_string(ds)
            ds = gdal.Warp(
                '', ds, dstSRS=self.SpatialReference, format='VRT', xRes=res,
                yRes=res)
            x, y, z = gdal_tools.get_arrays(ds)
            ax = plt.contourf(x, y, z, levels=[self.zmin, self.zmax])
            plt.close(plt.gcf())
            _x, _y = np.meshgrid(x, y)
            xy = np.vstack([_x.flatten(), _y.flatten()]).T
            _z = z.flatten()
            paths = list()
            # the output of contourf is a multipolygon.
            # clean up the secondary polygons to yield only the principal one.
            for PathCollection in ax.collections:
                for Path in PathCollection.get_paths():
                    polygons = Path.to_polygons(closed_only=True)
                    for i, polygon in enumerate(polygons):
                        xmin = np.min(polygon[:, 0])
                        xmax = np.max(polygon[:, 0])
                        ymin = np.min(polygon[:, 1])
                        ymax = np.max(polygon[:, 1])
                        xbool = np.logical_and(xy[:, 0] >= xmin,
                                               xy[:, 0] <= xmax)
                        ybool = np.logical_and(xy[:, 1] >= ymin,
                                               xy[:, 1] <= ymax)
                        _idx = np.where(np.logical_and(xbool, ybool))
                        path = _Path(polygon, closed=True)
                        idx = np.where(path.contains_points(xy[_idx]))
                        # hacky way for polygon cleanup
                        if i == 0:
                            if np.any(_z[_idx][idx] >= 0):
                                paths.append(polygon)
                                _outer_elim = False
                            else:
                                _outer_elim = True
                        else:
                            if not _outer_elim:
                                if np.all(_z[_idx][idx] >= self.zmax):
                                    pass
                                elif np.all(_z[_idx][idx] <= self.zmin):
                                    pass
                                else:
                                    paths.append(polygon)
            _Polygon = ogr.Geometry(ogr.wkbPolygon)
            for path in paths:
                _LinearRing = ogr.Geometry(ogr.wkbLinearRing)
                for x, y in path.vertices:
                    _LinearRing.AddPoint_2D(x, y)
                _Polygon.AddGeometry(_LinearRing)
            _Polygon.CloseRings()
            self.__polygon_collection.append(_Polygon)
        for polygon in self.__polygon_collection:
            print(polygon)

    def add_Dataset(self, Dataset):
        self.__container['gdal_strings'].append(Dataset)

    def __add_polygon(self, outer_vertices, *inner_vertices):
        raise NotImplementedError
        _Polygon = ogr.Geometry(ogr.wkbPolygon)
        _LinearRing = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in outer_vertices:
            _LinearRing.AddPoint_2D(x, y)
        _Polygon.AddGeometry(_LinearRing)
        for vertices in inner_vertices:
            _LinearRing = ogr.Geometry(ogr.wkbLinearRing)
            for x, y in vertices:
                _LinearRing.AddPoint_2D(x, y)
            _Polygon.AddGeometry(_LinearRing)
        _Polygon.CloseRings()
        assert _Polygon.IsValid()
        self.__MultiPolygon.AddGeometry(_Polygon)

    def __open_ogr_string(self, string):
        return gdal.Open(string)

    @property
    def SpatialReference(self):
        return self.__MultiPolygon.GetSpatialReference()

    @property
    def MultiPolygon(self):
        return self.__MultiPolygon

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