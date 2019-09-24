import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.path import Path
# from copy import deepcopy
import numpy as np
from scipy.interpolate import griddata
# from scipy.spatial import cKDTree
from osgeo import ogr
from jigsawpy import jigsaw_msh_t
from pysheds.grid import Grid

from geomesh.pslg import PlanarStraightLineGraph
# from geomesh.dataset_collection import DatasetCollection
from geomesh import gdal_tools


class SizeFunction:

    def __init__(
        self,
        pslg,
        hmin=None,
        hmax=None,
        SpatialReference=3395
    ):
        self._pslg = pslg
        self._hmin = hmin
        self._hmax = hmax
        self._SpatialReference = SpatialReference

    def set_shoreline(self, target_size):
        """
        shoreline is defined as the countour line at the zero level.
        """
        self.add_contour(target_size, 0)

    def set_ocean_boundary(self, i, target_size):
        """
        ocean boundary is defined as any point lying that lies on any of the
        outer rings of the multipolygon defining the planar straight line
        graph, and that has an elevation < 0.
        """
        ocean_boundary = (float(target_size), self.ocean_boundaries[i][1])
        self._ocean_boundaries[i] = ocean_boundary

    def set_land_boundary(self, i, target_size):
        """
        land boundary is defined as any point lying that lies on any of the
        outer rings of the multipolygon defining the pslg, and that has an
        elevation > 0.
        """
        land_boundary = (float(target_size), self.land_boundaries[i][1])
        self._land_boundaries[i] = land_boundary

    def set_inner_boundary(self, i, j, target_size):
        """
        An inner boundary is defined as any ring contained inside of of the
        outer rings of the multipolygon that defines the pslg. Inner rings
        do not contain other inner rings on the current implementation.
        """
        inner_boundary = (float(target_size), self.inner_boundaries[i][j][1])
        self._inner_boundaries[i][j] = inner_boundary

    def set_ocean_boundaries(self, target_size):
        for i, _ in enumerate(self.ocean_boundaries):
            self.set_ocean_boundary(i, target_size)

    def set_land_boundaries(self, target_size):
        for i, _ in enumerate(self.land_boundaries):
            self.set_land_boundary(i, target_size)

    def set_inner_boundaries(self, target_size):
        for i, inner_boundaries in enumerate(self.inner_boundaries):
            for j, inner_boundary in enumerate(inner_boundaries):
                self.set_inner_boundary(i, j, target_size)

    def add_watershed(self, target_size, pour_point, **kwargs):
        # default dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        # dirmap => (N, NE, E, SE, S, SW, W, NW)
        raise NotImplementedError
        for dem in self.pslg:
            grid = Grid.from_raster(dem.path, 'dem')
            grid.fill_depressions(data='dem', out_name='flooded_dem')
            grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
            dirmap = (360., 45., 90., 135., 180., 225., 270., 315.)
            grid.flowdir(
                data='inflated_dem',
                out_name='dir',
                dirmap=dirmap,
                routing='dinf',
                pits=0,
                flats=-1
                )
            plt.imshow(
                grid.view('dir'),
                interpolation='nearest',
                cmap='twilight_shifted',
                vmin=0.,
                vmax=360.
                )
            values = grid.view('dem')
            idx = np.where(values == np.max(values))
            grid.catchment(
                data='dir',
                x=idx[1][0],
                y=idx[0][0],
                out_name='catch',
                recursionlimit=15000,
                xytype='index'
                )
            grid.clip_to('catch')
            plt.imshow(grid.view('catch'))
            plt.show()
            plt.close(plt.gcf())
            grid.clip_to('dir')
            grid.accumulation(data='catch', out_name='acc')
            branches = grid.extract_river_network('catch', 'acc')
            for branch in branches['features']:
                line = np.asarray(branch['geometry']['coordinates'])
                plt.plot(line[:, 0], line[:, 1])
            plt.show()
        self._add_feature()

    def add_contour(self, target_size, level):
        self.add_contours(target_size, levels=[float(level)])

    def add_contours(self, target_size, levels):
        _MultiLineString = ogr.Geometry(ogr.wkbMultiLineString)
        _MultiLineString.AssignSpatialReference(self.SpatialReference)
        for ds in self.pslg:
            x, y, elevation = ds.get_arrays(self.SpatialReference)
            ax = plt.contour(x, y, elevation, levels=levels)
            plt.close(plt.gcf())
            for line_collection in ax.collections:
                for path in line_collection.get_paths():
                    _LineString = ogr.Geometry(ogr.wkbLineString)
                    _LineString.AssignSpatialReference(self.SpatialReference)
                    for (x, y), _ in path.iter_segments():
                        if self.SpatialReference.IsProjected():
                            _LineString.AddPoint(x, y, ds.get_value(x, y))
                        elif self.SpatialReference.IsGeographic():
                            _LineString.AddPoint(y, x, ds.get_value(x, y))
                        else:
                            raise Exception('duck-typing error.')
                    _MultiLineString.AddGeometry(_LineString)
        self._add_feature(target_size, _MultiLineString)

    def set_mpl_tri_mask(self):
        self.mpl_tri.set_mask(self.mpl_tri_mask)

    def make_plot(self, axes=None, show=False, masked=True):
        if axes is None:
            axes = plt.figure().add_subplot(111)
        if masked:
            self.set_mpl_tri_mask()
        ax = axes.tricontourf(self.mpl_tri, self.values)
        plt.colorbar(ax)
        axes.axis('scaled')
        if show:
            plt.show()

    def _add_feature(self, feature_size, MultiLineString):
        del(self._points)
        self._features.append((feature_size, MultiLineString))

    def _check_ocean_boundaries(self):
        sizes = [target_size for target_size, _ in self.ocean_boundaries]
        if np.all(sizes is None):
            raise AttributeError('Ocean boundary target sizes not set.')
        for i, size in enumerate(sizes):
            if size is None:
                msg = ('Target size for ocean boundary {} '.format(i))
                msg += 'not set.'
                raise AttributeError(msg)

    def _check_land_boundaries(self):
        sizes = [target_size for target_size, _ in self.land_boundaries]
        if np.all(sizes is None):
            raise AttributeError('land boundary target sizes not set.')
        for i, size in enumerate(sizes):
            if size is None:
                msg = ('Target size for land boundary {} '.format(i))
                msg += 'not set.'
                raise AttributeError(msg)

    def _check_inner_boundaries(self):
        sizes = list()
        for inner_boundaries in self.inner_boundaries:
            _inner_sizes = list()
            for target_size, _ in inner_boundaries:
                _inner_sizes.append(target_size)
            sizes.append(_inner_sizes)
        _target_size_flat = np.asarray(sizes).flatten()
        if np.all(_target_size_flat) is None:
            raise AttributeError('Target size for inner boundaries not set.')
        elif np.any(_target_size_flat) is None:
            for i, _ in enumerate(self.inner_boundaries):
                for j, (target_size, _) in enumerate(_):
                    if target_size is None:
                        msg = 'Must set target size for land boundary '
                        msg += '({}, {})'.format(i, j)
                        raise AttributeError(msg)

    @property
    def ocean_boundaries(self):
        return tuple(self._ocean_boundaries)

    @property
    def land_boundaries(self):
        return tuple(self._land_boundaries)

    @property
    def inner_boundaries(self):
        return tuple(self._inner_boundaries)

    @property
    def planar_straight_line_graph(self):
        return self.pslg

    @property
    def pslg(self):
        return self._pslg

    @property
    def hmin(self):
        return self._hmin

    @property
    def hmax(self):
        return self._hmax

    @property
    def SpatialReference(self):
        return self._SpatialReference

    @property
    def mpl_tri(self):
        return self._mpl_tri

    @property
    def mpl_tri_mask(self):
        return self._mpl_tri_mask

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def xy(self):
        return self._xy

    @property
    def elements(self):
        return self._elements

    @property
    def values(self):
        return self._values

    @property
    def elevation(self):
        return self._elevation

    @property
    def points(self):
        return self._points

    @property
    def features(self):
        return tuple(self._features)

    @property
    def scaling(self):
        return self._scaling

    @property
    def vert2(self):
        vert2 = list()
        for x, y, z in self.points:
            vert2.append(((x, y), 0))
        return np.asarray(vert2, dtype=jigsaw_msh_t.VERT2_t)

    @property
    def tria3(self):
        tria3 = list()
        triangles = self.mpl_tri.triangles[np.where(~self.mpl_tri_mask)]
        mpl_tri = Triangulation(self.x, self.y, triangles=triangles)
        for indices in mpl_tri.triangles:
            tria3.append((tuple(indices), 0))
        return np.asarray(tria3, dtype=jigsaw_msh_t.TRIA3_t)

    @property
    def hfun_value(self):
        return np.asarray(self.values.tolist(), dtype=jigsaw_msh_t.REALS_t)

    @property
    def hfun(self):
        hfun = jigsaw_msh_t()
        hfun.vert2 = self.vert2
        hfun.tria3 = self.tria3
        hfun.value = self.hfun_value
        hfun.ndim = 2
        hfun.mshID = "euclidean-mesh"
        return hfun

    @property
    def _pslg(self):
        return self.__pslg

    @property
    def _hmin(self):
        return self.__hmin

    @property
    def _hmax(self):
        return self.__hmax

    @property
    def _x(self):
        return self.mpl_tri.x

    @property
    def _y(self):
        return self.mpl_tri.y

    @property
    def _xy(self):
        return np.vstack([self.x, self.y]).T

    @property
    def _elements(self):
        return self.mpl_tri.triangles

    @property
    def _values(self):
        try:
            return self.__values
        except AttributeError:
            values = np.full((self.elevation.size,), np.nan)
            self._check_ocean_boundaries()
            self._check_land_boundaries()
            self._check_inner_boundaries()
            for target_size, idxs in self.ocean_boundaries:
                values[idxs] = target_size
            for target_size, idxs in self.land_boundaries:
                values[idxs] = target_size
            for inner_boundaries in self.inner_boundaries:
                for target_size, idxs in inner_boundaries:
                    values[idxs] = target_size
            initial_i = self.pslg.values.size
            for target_size, _MultiLineString in self.features:
                final_i = initial_i
                for _LineString in _MultiLineString:
                    final_i += _LineString.GetPointCount()
                values[initial_i:final_i] = target_size
                initial_i = final_i
            idxs = np.where(np.isnan(values))
            not_idxs = np.where(~np.isnan(values))
            pad_values = griddata(
                (self.x[not_idxs], self.y[not_idxs]),
                values[not_idxs],
                (self.x[idxs], self.y[idxs]),
                method='nearest')
            values[idxs] = pad_values
            self._values = values
            return self.__values

    @property
    def _elevation(self):
        return self.points[:, 2]

    @property
    def _points(self):
        try:
            return self.__points
        except AttributeError:
            self._points = self.pslg.points
            return self.__points

    @property
    def _SpatialReference(self):
        return self.__SpatialReference

    @property
    def _mpl_tri(self):
        try:
            return self.__mpl_tri
        except AttributeError:
            self._mpl_tri = self.points
            return self.__mpl_tri

    @property
    def _mpl_tri_mask(self):
        try:
            return self.__mpl_tri_mask
        except AttributeError:
            self._mpl_tri_mask = np.full(
                (self.mpl_tri.triangles.shape[0],), True)
            return self.__mpl_tri_mask

    @property
    def _ocean_boundaries(self):
        try:
            return self.__ocean_boundaries
        except AttributeError:
            self._ocean_boundaries = list()
            return self.__ocean_boundaries

    @property
    def _land_boundaries(self):
        try:
            return self.__land_boundaries
        except AttributeError:
            self._land_boundaries = list()
            return self.__land_boundaries

    @property
    def _inner_boundaries(self):
        try:
            return self.__inner_boundaries
        except AttributeError:
            self._inner_boundaries = list()
            return self.__inner_boundaries

    @property
    def _features(self):
        try:
            return self.__features
        except AttributeError:
            self._features = list()
            return self.__features

    @property
    def _scaling(self):
        try:
            return self.__scaling
        except AttributeError:
            self._scaling = "absolute"
            return self.__scaling

    @property
    def _hfun(self):
        try:
            return self.__hfun
        except AttributeError:
            self._hfun = jigsaw_msh_t()
            return self.__hfun

    @scaling.setter
    def scaling(self, scaling):
        self._scaling = scaling

    @_pslg.setter
    def _pslg(self, pslg):
        assert isinstance(pslg, PlanarStraightLineGraph)
        self.__pslg = pslg

    @_hmin.setter
    def _hmin(self, hmin):
        if hmin is not None:
            hmin = float(hmin)
            assert hmin > 0. and hmin < float("inf")
        self.__hmin = hmin

    @_features.setter
    def _features(self, features):
        del(self._points)
        self.__features = features

    @_hmax.setter
    def _hmax(self, hmax):
        if hmax is not None:
            hmax = float(hmax)
            assert hmax > 0. and hmax < float("inf")
        self.__hmax = hmax

    @_mpl_tri.setter
    def _mpl_tri(self, points):
        self.__mpl_tri = Triangulation(points[:, 0], points[:, 1])

    @_points.setter
    def _points(self, points):
        for target_size, _MultiLineString in self.features:
            for _LineString in _MultiLineString:
                _points = np.asarray(_LineString.GetPoints())
                points = np.vstack([points, _points])
        self.__points = points

    @_values.setter
    def _values(self, values):
        self.__values = values

    @_mpl_tri_mask.setter
    def _mpl_tri_mask(self, mask):
        centroid_x = np.sum(
            self.mpl_tri.x[self.mpl_tri.triangles], axis=1) / 3
        centroid_y = np.sum(
            self.mpl_tri.y[self.mpl_tri.triangles], axis=1) / 3
        centroids = np.vstack([centroid_x, centroid_y]).T
        for _Polygon in self.pslg.MultiPolygon:
            for i, _LinearRing in enumerate(_Polygon):
                mpl_path = Path(
                    np.asarray(_LinearRing.GetPoints())[:, :2],
                    closed=True)
                if i == 0:
                    mask = np.logical_and(
                        mask, ~mpl_path.contains_points(centroids))
                else:
                    mask = np.logical_or(
                        mask, mpl_path.contains_points(centroids))
        self.__mpl_tri_mask = mask

    @_ocean_boundaries.setter
    def _ocean_boundaries(self, ocean_boundaries):
        for boundary in self.pslg.ocean_boundaries:
            ocean_boundaries.append((None, boundary))
        self.__ocean_boundaries = ocean_boundaries

    @_land_boundaries.setter
    def _land_boundaries(self, land_boundaries):
        for boundary in self.pslg.land_boundaries:
            land_boundaries.append((None, boundary))
        self.__land_boundaries = land_boundaries

    @_inner_boundaries.setter
    def _inner_boundaries(self, inner_boundaries):
        for inner_boundaries in self.pslg.inner_boundaries:
            _inner_boundaries = list()
            for inner_boundary in inner_boundaries:
                _inner_boundaries.append((None, inner_boundary))
            inner_boundaries.append(_inner_boundaries)
        self.__inner_boundaries = inner_boundaries

    @_SpatialReference.setter
    def _SpatialReference(self, SpatialReference):
        SpatialReference = gdal_tools.sanitize_SpatialReference(
            SpatialReference)
        self.__SpatialReference = SpatialReference

    @_scaling.setter
    def _scaling(self, scaling):
        assert scaling in ["absolute", "relative"]
        self.__scaling = scaling

    @_points.deleter
    def _points(self):
        try:
            del(self.__points)
            del(self._values)
            del(self._mpl_tri)
        except AttributeError:
            pass

    @_values.deleter
    def _values(self):
        try:
            del(self.__values)
        except AttributeError:
            pass

    @_mpl_tri.deleter
    def _mpl_tri(self):
        try:
            del(self.__mpl_tri)
            del(self._mpl_tri_mask)
        except AttributeError:
            pass

    @_mpl_tri_mask.deleter
    def _mpl_tri_mask(self):
        try:
            del(self.__mpl_tri_mask)
        except AttributeError:
            pass
